//! R model formulas (`~ a + b + a:b`, `~ 0 + a`, `~ a * b * c`, `~ poly(x, 2) + log(age)`,
//! ...) and the model matrices R builds from them
//!
//! A formula variable is a column of the sample metadata or an R expression of
//! columns (`factor(x)`, `log(age)`, `I(x^2)`, `poly(x, 2)`, `ns(x, df = 3)`, ...,
//! evaluated as R's `model.frame()` does, see [`super::model_frame`]). Columns are
//! factors unless they are numeric (continuous) columns of the sample metadata, or
//! logical (all values TRUE/FALSE). Factors use treatment contrasts and ordered
//! factors polynomial contrasts, as `model.matrix()` does with R's default
//! `contrasts` option.

use std::collections::HashMap;

use ndarray::Array2;

use super::design::{check_full_rank, DesignInfo};
use super::model_frame::{Column, FrameValue, ModelFrame};
use super::rfuns::format_real15;
use crate::data::SampleMetadata;
use crate::error::{DeseqError, Result};

/// The right-hand side of a model formula
/// R equivalent: terms.formula() (variables, term.labels, order, intercept, offset)
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    /// The variables (R: attr(terms(formula), "variables")) in order of first
    /// appearance, including any removed by `-` and offsets, as R deparses them:
    /// `batch`, `log(age)`, `poly(x, 2)`, `` `my var` `` is `my var`
    pub variables: Vec<String>,
    /// The expression of each variable
    pub(crate) exprs: Vec<Ex>,
    /// The terms as bit sets over `variables`, ordered by degree and, within a degree,
    /// by their position in the formula (R: terms(keep.order = FALSE))
    pub terms: Vec<u64>,
    /// Whether the model has an intercept (R: attr(terms(formula), "intercept"))
    pub intercept: bool,
    /// The variables that are `offset(...)` terms (R: attr(terms, "offset")); they are
    /// not columns of the model matrix
    pub offsets: Vec<usize>,
    /// The names in the formula, in order of first appearance (R: all.vars(formula)):
    /// the metadata columns it uses
    pub all_vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A name; `true` if it was backquoted
    Name(String, bool),
    Number(String),
    Str(String),
    Op(&'static str),
    /// `%op%`, with the text between the percent signs
    Special(String),
}

fn formula_error(text: &str, reason: impl std::fmt::Display) -> DeseqError {
    DeseqError::InvalidDesignMatrix {
        reason: format!("invalid model formula '{}': {}", text.trim(), reason),
    }
}

/// Split a formula into tokens; a leading `~` is optional
fn tokenize(text: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = text.trim().chars().collect();
    let mut i = 0;
    if chars.first() == Some(&'~') {
        i = 1;
    }
    let mut tokens = Vec::new();
    const TWO: [&str; 7] = ["==", "!=", "<=", ">=", "&&", "||", "::"];
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if let Some(n) = next {
            let pair: String = [c, n].iter().collect();
            if let Some(op) = TWO.iter().find(|op| **op == pair) {
                i += 2;
                if *op == "::" && chars.get(i) == Some(&':') {
                    i += 1;
                }
                tokens.push(Token::Op(op));
                continue;
            }
        }
        match c {
            '+' | '-' | '*' | '/' | ':' | '^' | '(' | ')' | ',' | '=' | '<' | '>' | '!' | '&' | '|' => {
                i += 1;
                tokens.push(Token::Op(match c {
                    '+' => "+",
                    '-' => "-",
                    '*' => "*",
                    '/' => "/",
                    ':' => ":",
                    '^' => "^",
                    '(' => "(",
                    ')' => ")",
                    ',' => ",",
                    '=' => "=",
                    '<' => "<",
                    '>' => ">",
                    '!' => "!",
                    '&' => "&",
                    _ => "|",
                }));
            }
            '%' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '%' {
                    i += 1;
                }
                if i == chars.len() {
                    return Err(formula_error(text, "unterminated %operator%"));
                }
                tokens.push(Token::Special(chars[start..i].iter().collect()));
                i += 1;
            }
            '`' | '"' | '\'' => {
                i += 1;
                let mut s = String::new();
                loop {
                    match chars.get(i) {
                        None => return Err(formula_error(text, "unterminated quoted name or string")),
                        Some(&q) if q == c => {
                            i += 1;
                            break;
                        }
                        Some('\\') => {
                            let e = chars.get(i + 1).copied().ok_or_else(|| formula_error(text, "bad escape"))?;
                            s.push(match e {
                                'n' => '\n',
                                't' => '\t',
                                other => other,
                            });
                            i += 2;
                        }
                        Some(&ch) => {
                            s.push(ch);
                            i += 1;
                        }
                    }
                }
                tokens.push(if c == '`' { Token::Name(s, true) } else { Token::Str(s) });
            }
            c if c.is_ascii_digit() || (c == '.' && next.is_some_and(|d| d.is_ascii_digit())) => {
                let start = i;
                if c == '0' && matches!(next, Some('x' | 'X')) {
                    i += 2;
                    while i < chars.len() && chars[i].is_ascii_hexdigit() {
                        i += 1;
                    }
                } else {
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                        i += 1;
                        if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                            i += 1;
                        }
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                if i < chars.len() && chars[i] == 'L' {
                    i += 1;
                }
                if i < chars.len() && chars[i] == 'i' {
                    return Err(formula_error(text, "complex numbers are not supported"));
                }
                tokens.push(Token::Number(chars[start..i].iter().collect()));
            }
            c if c.is_alphabetic() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '.' || chars[i] == '_') {
                    i += 1;
                }
                tokens.push(Token::Name(chars[start..i].iter().collect(), false));
            }
            '~' => return Err(formula_error(text, "a response (left-hand side) is not allowed; write '~ a + b'")),
            other => return Err(formula_error(text, format!("unexpected character '{}'", other))),
        }
    }
    Ok(tokens)
}

/// An R expression (the parse tree of the formula)
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Ex {
    Sym(String),
    /// A number; `true` for an integer literal (`2L`)
    Num(f64, bool),
    Str(String),
    Lgl(bool),
    Null,
    Na,
    /// A call `f(args)`; `f` may be `pkg::f`
    Call(String, Vec<(Option<String>, Ex)>),
    Unary(&'static str, Box<Ex>),
    /// A binary operator: `+ - * / ^ : == != < > <= >= & && | ||` or `%op%`
    Binary(String, Box<Ex>, Box<Ex>),
    Paren(Box<Ex>),
}

/// Whether `name` is a syntactic R name (deparse() writes other names in backquotes)
fn is_syntactic(name: &str) -> bool {
    !name.is_empty() && make_names(name) == name && name != "..."
}

/// R's deparse() of an expression (width.cutoff = 500, backticks for non-syntactic
/// names), which gives the names of model frame variables and model matrix columns
/// R equivalent: deparse2() of model.frame() / the term labels of terms.formula()
pub(crate) fn deparse(e: &Ex) -> String {
    match e {
        Ex::Sym(s) => {
            if is_syntactic(s) {
                s.clone()
            } else {
                format!("`{}`", s)
            }
        }
        Ex::Num(v, integer) => {
            let s = format_real15(*v);
            if *integer { format!("{}L", s) } else { s }
        }
        Ex::Str(s) => {
            let mut out = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        Ex::Lgl(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        Ex::Null => "NULL".to_string(),
        Ex::Na => "NA".to_string(),
        Ex::Call(f, args) => {
            let parts: Vec<String> = args
                .iter()
                .map(|(name, a)| match name {
                    Some(n) => {
                        let n = if is_syntactic(n) { n.clone() } else { format!("`{}`", n) };
                        format!("{} = {}", n, deparse(a))
                    }
                    None => deparse(a),
                })
                .collect();
            format!("{}({})", f, parts.join(", "))
        }
        Ex::Unary(op, a) => format!("{}{}", op, deparse(a)),
        Ex::Binary(op, a, b) => {
            let spaced = !matches!(op.as_str(), "/" | "^" | ":" | "%%" | "%/%");
            if spaced {
                format!("{} {} {}", deparse(a), op, deparse(b))
            } else {
                format!("{}{}{}", deparse(a), op, deparse(b))
            }
        }
        Ex::Paren(a) => format!("({})", deparse(a)),
    }
}

/// Recursive-descent parser with R's operator precedence (highest first): `::`,
/// function calls, `^` (right to left), unary `-`/`+`, `:`, `%op%`, `*` `/`,
/// binary `+` `-`, comparisons, `!`, `&` `&&`, `|` `||`
struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    text: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_op(&self) -> Option<&'static str> {
        match self.peek() {
            Some(Token::Op(op)) => Some(op),
            _ => None,
        }
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn binary_left(&mut self, ops: &[&str], sub: fn(&mut Self) -> Result<Ex>) -> Result<Ex> {
        let mut left = sub(self)?;
        while let Some(op) = self.peek_op().filter(|op| ops.contains(op)) {
            self.pos += 1;
            let right = sub(self)?;
            left = Ex::Binary(op.to_string(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn or(&mut self) -> Result<Ex> {
        self.binary_left(&["|", "||"], Self::and)
    }

    fn and(&mut self) -> Result<Ex> {
        self.binary_left(&["&", "&&"], Self::not)
    }

    fn not(&mut self) -> Result<Ex> {
        if self.peek_op() == Some("!") {
            self.pos += 1;
            return Ok(Ex::Unary("!", Box::new(self.not()?)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Result<Ex> {
        self.binary_left(&["==", "!=", "<", ">", "<=", ">="], Self::sum)
    }

    fn sum(&mut self) -> Result<Ex> {
        self.binary_left(&["+", "-"], Self::product)
    }

    fn product(&mut self) -> Result<Ex> {
        self.binary_left(&["*", "/"], Self::special)
    }

    fn special(&mut self) -> Result<Ex> {
        let mut left = self.colon()?;
        while let Some(Token::Special(op)) = self.peek().cloned() {
            self.pos += 1;
            let right = self.colon()?;
            left = Ex::Binary(format!("%{}%", op), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn colon(&mut self) -> Result<Ex> {
        self.binary_left(&[":"], Self::unary)
    }

    fn unary(&mut self) -> Result<Ex> {
        match self.peek_op() {
            Some("-") => {
                self.pos += 1;
                Ok(Ex::Unary("-", Box::new(self.unary()?)))
            }
            Some("+") => {
                self.pos += 1;
                Ok(Ex::Unary("+", Box::new(self.unary()?)))
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Ex> {
        let base = self.postfix()?;
        if self.peek_op() == Some("^") {
            self.pos += 1;
            let exponent = self.unary()?;
            return Ok(Ex::Binary("^".to_string(), Box::new(base), Box::new(exponent)));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> Result<Ex> {
        let e = self.primary()?;
        if self.peek_op() == Some("(") {
            let name = match &e {
                Ex::Sym(s) => s.clone(),
                Ex::Call(f, args) if args.is_empty() && f.contains("::") => f.clone(),
                _ => return Err(formula_error(self.text, format!("attempt to apply non-function '{}'", deparse(&e)))),
            };
            self.pos += 1;
            let args = self.arguments()?;
            return Ok(Ex::Call(name, args));
        }
        if let Ex::Call(f, args) = &e {
            if args.is_empty() && f.contains("::") {
                return Err(formula_error(self.text, format!("'{}' must be called", f)));
            }
        }
        Ok(e)
    }

    fn arguments(&mut self) -> Result<Vec<(Option<String>, Ex)>> {
        let mut args = Vec::new();
        if self.peek_op() == Some(")") {
            self.pos += 1;
            return Ok(args);
        }
        loop {
            // name = value
            let named = match (self.tokens.get(self.pos), self.tokens.get(self.pos + 1)) {
                (Some(Token::Name(n, _) | Token::Str(n)), Some(Token::Op("="))) => Some(n.clone()),
                _ => None,
            };
            if named.is_some() {
                self.pos += 2;
            }
            if matches!(self.peek_op(), Some(",") | Some(")")) {
                return Err(formula_error(self.text, "empty function arguments are not supported"));
            }
            let value = self.or()?;
            args.push((named, value));
            match self.next() {
                Some(Token::Op(",")) => continue,
                Some(Token::Op(")")) => return Ok(args),
                _ => return Err(formula_error(self.text, "missing ')' after function arguments")),
            }
        }
    }

    fn primary(&mut self) -> Result<Ex> {
        match self.next() {
            Some(Token::Name(name, quoted)) => {
                if self.peek_op() == Some("::") {
                    self.pos += 1;
                    match self.next() {
                        Some(Token::Name(f, _)) => return Ok(Ex::Call(format!("{}::{}", name, f), Vec::new())),
                        _ => return Err(formula_error(self.text, "expected a name after '::'")),
                    }
                }
                if !quoted {
                    match name.as_str() {
                        "TRUE" => return Ok(Ex::Lgl(true)),
                        "FALSE" => return Ok(Ex::Lgl(false)),
                        "NULL" => return Ok(Ex::Null),
                        "NA" | "NA_real_" | "NA_integer_" | "NA_character_" => return Ok(Ex::Na),
                        "Inf" => return Ok(Ex::Num(f64::INFINITY, false)),
                        "NaN" => return Ok(Ex::Num(f64::NAN, false)),
                        _ => {}
                    }
                }
                Ok(Ex::Sym(name))
            }
            Some(Token::Number(n)) => {
                let (body, integer) = match n.strip_suffix('L') {
                    Some(b) => (b, true),
                    None => (n.as_str(), false),
                };
                let v = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
                    u64::from_str_radix(h, 16).map(|v| v as f64).ok()
                } else {
                    crate::io::tximport::r_strtod(body)
                };
                v.map(|v| Ex::Num(v, integer)).ok_or_else(|| formula_error(self.text, format!("invalid number '{}'", n)))
            }
            Some(Token::Str(s)) => Ok(Ex::Str(s)),
            Some(Token::Op("(")) => {
                let inner = self.or()?;
                match self.next() {
                    Some(Token::Op(")")) => Ok(Ex::Paren(Box::new(inner))),
                    _ => Err(formula_error(self.text, "missing ')'")),
                }
            }
            Some(t) => Err(formula_error(self.text, format!("unexpected {}", token_text(&t)))),
            None => Err(formula_error(self.text, "unexpected end of formula")),
        }
    }
}

fn token_text(t: &Token) -> String {
    match t {
        Token::Name(n, _) => format!("'{}'", n),
        Token::Number(n) => format!("'{}'", n),
        Token::Str(s) => format!("string \"{}\"", s),
        Token::Op(op) => format!("'{}'", op),
        Token::Special(s) => format!("'%{}%'", s),
    }
}

/// Builds the term list of an expression
/// R equivalent: EncodeVars() and the term algebra of model.c (PlusTerms, DeleteTerms,
/// InteractTerms, CrossTerms, InTerms, NestTerms, PowerTerms, TrimRepeats)
struct Encoder<'a> {
    variables: Vec<String>,
    exprs: Vec<Ex>,
    intercept: bool,
    /// false inside the right-hand side of `-` (R: parity)
    parity: bool,
    text: &'a str,
}

/// R: TrimRepeats() -- drops empty and repeated terms, keeping first occurrences
fn trim_repeats(terms: Vec<u64>) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::with_capacity(terms.len());
    for t in terms {
        if t != 0 && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// R: InteractTerms() -- every left term with every right term, left outer
fn interact(left: &[u64], right: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(left.len() * right.len());
    for &l in left {
        for &r in right {
            out.push(l | r);
        }
    }
    trim_repeats(out)
}

impl Encoder<'_> {
    /// R: InstallVar() -- variables are identified by their deparsed text
    fn var_bit(&mut self, e: &Ex) -> Result<u64> {
        let label = deparse(e);
        let idx = match self.variables.iter().position(|v| *v == label) {
            Some(i) => i,
            None => {
                self.variables.push(label);
                self.exprs.push(e.clone());
                self.variables.len() - 1
            }
        };
        if idx >= 64 {
            return Err(formula_error(self.text, "more than 64 variables"));
        }
        Ok(1u64 << idx)
    }

    fn encode(&mut self, e: &Ex) -> Result<Vec<u64>> {
        match e {
            Ex::Num(v, _) => {
                // R: isOne / isZero set the intercept (reversed under `-`)
                if *v == 1.0 {
                    self.intercept = self.parity;
                } else if *v == 0.0 {
                    self.intercept = !self.parity;
                } else {
                    return Err(formula_error(self.text, format!("invalid term '{}'", deparse(e))));
                }
                Ok(Vec::new())
            }
            Ex::Str(_) | Ex::Lgl(_) | Ex::Null | Ex::Na => {
                Err(formula_error(self.text, format!("invalid term '{}' (R: invalid model formula)", deparse(e))))
            }
            Ex::Paren(a) => self.encode(a),
            Ex::Unary("+", a) => self.encode(a),
            Ex::Unary("-", b) => self.delete(Vec::new(), b),
            Ex::Binary(op, a, b) => match op.as_str() {
                "+" => {
                    let mut left = self.encode(a)?;
                    left.extend(self.encode(b)?);
                    Ok(trim_repeats(left))
                }
                "-" => {
                    let left = self.encode(a)?;
                    self.delete(left, b)
                }
                ":" => {
                    let left = self.encode(a)?;
                    let right = self.encode(b)?;
                    Ok(interact(&left, &right))
                }
                "*" => {
                    // R: CrossTerms() -- a + b + a:b
                    let left = self.encode(a)?;
                    let right = self.encode(b)?;
                    let mut out = left.clone();
                    out.extend(right.iter().copied());
                    out.extend(interact(&left, &right));
                    Ok(trim_repeats(out))
                }
                "%in%" => {
                    // R: InTerms() -- every left term with all right variables
                    let left = self.encode(a)?;
                    let right = self.encode(b)?;
                    let all = right.iter().fold(0u64, |acc, &t| acc | t);
                    Ok(trim_repeats(left.iter().map(|&l| l | all).collect()))
                }
                "/" => {
                    // R: NestTerms() -- a / b = a + b %in% a
                    let left = self.encode(a)?;
                    let right = self.encode(b)?;
                    let all = left.iter().fold(0u64, |acc, &t| acc | t);
                    let mut out = left.clone();
                    out.extend(right.iter().map(|&r| r | all));
                    Ok(trim_repeats(out))
                }
                "^" => {
                    // R: PowerTerms() -- all interactions up to the given order; the power is
                    // asInteger() of a number (truncated) and must be at least 2
                    let n = match &**b {
                        Ex::Num(n, _) if n.is_finite() && n.trunc() >= 2.0 => n.trunc() as usize,
                        _ => return Err(formula_error(self.text, "invalid power in formula")),
                    };
                    let left = self.encode(a)?;
                    let mut out = left.clone();
                    for _ in 1..n {
                        let mut next = out.clone();
                        next.extend(interact(&out, &left));
                        out = trim_repeats(next);
                    }
                    Ok(out)
                }
                _ => Ok(vec![self.var_bit(e)?]),
            },
            _ => Ok(vec![self.var_bit(e)?]),
        }
    }

    /// R: DeleteTerms()
    fn delete(&mut self, left: Vec<u64>, b: &Ex) -> Result<Vec<u64>> {
        self.parity = !self.parity;
        let right = self.encode(b);
        self.parity = !self.parity;
        let right = right?;
        Ok(left.into_iter().filter(|t| !right.contains(t)).collect())
    }
}

/// R's all.vars(): the names of an expression, in order of first appearance, without
/// function names
fn collect_vars(e: &Ex, out: &mut Vec<String>) {
    match e {
        Ex::Sym(s) => {
            if !out.contains(s) {
                out.push(s.clone());
            }
        }
        Ex::Call(_, args) => args.iter().for_each(|(_, a)| collect_vars(a, out)),
        Ex::Unary(_, a) | Ex::Paren(a) => collect_vars(a, out),
        Ex::Binary(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        _ => {}
    }
}

impl Formula {
    /// Parse the right-hand side of an R formula, e.g. `"~ batch + condition"`,
    /// `"~ genotype * treatment"`, `"~ 0 + condition"`, `"~ (a + b + c)^2"`,
    /// `"~ batch + poly(age, 2)"`
    /// R equivalent: terms(as.formula(text))
    ///
    /// Supported: `+`, `-`, `:`, `*`, `/`, `%in%`, `^`, parentheses, `0`/`1`/`- 1`
    /// for the intercept, backquoted names, `offset(...)` and the functions that
    /// [`super::model_frame`] evaluates. `.` is kept as a name; it is not a metadata
    /// column, so designs with it fail as DESeqDataSet() fails.
    pub fn parse(text: &str) -> Result<Formula> {
        let tokens = tokenize(text)?;
        if tokens.is_empty() {
            return Err(formula_error(text, "empty formula (use '~ 1' for an intercept-only model)"));
        }
        let mut parser = Parser { tokens, pos: 0, text };
        let expr = parser.or()?;
        if parser.pos != parser.tokens.len() {
            return Err(formula_error(text, format!("unexpected {}", token_text(&parser.tokens[parser.pos]))));
        }
        let mut enc = Encoder { variables: Vec::new(), exprs: Vec::new(), intercept: true, parity: true, text };
        let terms = enc.encode(&expr)?;
        // R: terms.formula(keep.order = FALSE) sorts terms by degree, keeping the
        // formula order within a degree; terms with an offset are removed
        let offsets: Vec<usize> = (0..enc.variables.len()).filter(|&i| enc.variables[i].starts_with("offset(")).collect();
        let offset_bits = offsets.iter().fold(0u64, |acc, &i| acc | 1u64 << i);
        let mut terms: Vec<u64> = trim_repeats(terms).into_iter().filter(|t| t & offset_bits == 0).collect();
        terms.sort_by_key(|t| t.count_ones());
        let mut all_vars = Vec::new();
        collect_vars(&expr, &mut all_vars);
        Ok(Formula { variables: enc.variables, exprs: enc.exprs, terms, intercept: enc.intercept, offsets, all_vars })
    }

    /// The variables of a term, in `variables` order
    pub fn term_variables(&self, term: u64) -> Vec<&str> {
        (0..self.variables.len()).filter(|&i| term >> i & 1 == 1).map(|i| self.variables[i].as_str()).collect()
    }

    /// R: attr(terms(formula), "term.labels")
    pub fn term_labels(&self) -> Vec<String> {
        self.terms.iter().map(|&t| self.term_variables(t).join(":")).collect()
    }

    /// Whether any term is an interaction (R: any(attr(terms, "order") > 1))
    pub fn has_interactions(&self) -> bool {
        self.terms.iter().any(|t| t.count_ones() > 1)
    }

    /// The formula as R prints its terms, e.g. `~ batch + condition + batch:condition`
    pub fn display(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.intercept {
            parts.push("0".to_string());
        }
        parts.extend(self.term_labels());
        parts.extend(self.offsets.iter().map(|&i| self.variables[i].clone()));
        if parts.is_empty() {
            parts.push("1".to_string());
        }
        format!("~ {}", parts.join(" + "))
    }
}

/// R's make.names() for one name
/// R equivalent: make.names(name)
pub fn make_names(name: &str) -> String {
    const RESERVED: [&str; 19] = [
        "if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE", "FALSE", "NULL", "Inf", "NaN",
        "NA", "NA_integer_", "NA_real_", "NA_character_", "NA_complex_", "in",
    ];
    let mut chars = name.chars();
    let first = chars.next();
    let need_prefix = match first {
        None => true,
        Some('.') => chars.next().is_some_and(|c| c.is_ascii_digit()),
        Some(c) => !c.is_alphabetic(),
    };
    let mut out = String::with_capacity(name.len() + 2);
    if need_prefix {
        out.push('X');
    }
    out.extend(name.chars().map(|c| if c.is_alphanumeric() || c == '.' || c == '_' { c } else { '.' }));
    if RESERVED.contains(&out.as_str()) || out == "..." || is_dot_dot_number(&out) {
        out.push('.');
    }
    out
}

/// `..1`, `..2`, ... are reserved in R as well
fn is_dot_dot_number(s: &str) -> bool {
    s.len() > 2 && s.starts_with("..") && s[2..].chars().all(|c| c.is_ascii_digit())
}

/// A model matrix with R's column names
#[derive(Debug, Clone)]
pub struct ModelMatrix {
    pub matrix: Array2<f64>,
    /// R: colnames(model.matrix(...)), e.g. `(Intercept)`, `batchb2`, `batchb2:conditionB`,
    /// `poly(x, 2)1`
    pub column_names: Vec<String>,
    /// Index into `formula.terms` of each column; `None` for the intercept
    pub column_terms: Vec<Option<usize>>,
}

/// R's model matrix of a formula
/// R equivalent: model.matrix(formula, colData) with treatment contrasts for factors and
/// contr.poly for ordered factors (modelmatrix() in model.c)
///
/// Columns: `(Intercept)` if present, then each term in `formula.terms` order. A factor
/// in a term is coded by contrasts when the term without it is the empty term or is
/// contained in a preceding term, and by indicators of all levels otherwise; without an
/// intercept, the first factor of the first term containing one is coded by indicators
/// (R: TermCode() and the no-intercept rule of modelmatrix()). A numeric variable is one
/// column named by the variable, a matrix variable (poly(), ns(), bs()) one column per
/// matrix column, named variable + column name (a one-column matrix by the variable
/// alone). Interaction columns are products, the first variable varying fastest, named
/// `a1:b2`.
pub fn model_matrix(formula: &Formula, frame: &ModelFrame) -> Result<ModelMatrix> {
    let n = frame.n_samples;
    let nvar = formula.variables.len();
    // R: TermCode() -- 1 = contrasts, 2 = indicators of every level
    let mut codes: Vec<Vec<u8>> = Vec::with_capacity(formula.terms.len());
    for (j, &term) in formula.terms.iter().enumerate() {
        let mut term_codes = vec![0u8; nvar];
        for (i, code) in term_codes.iter_mut().enumerate() {
            if term >> i & 1 == 0 {
                continue;
            }
            let without = term & !(1u64 << i);
            *code = if without == 0 || formula.terms[..j].iter().any(|&t| without & !t == 0) { 1 } else { 2 };
        }
        codes.push(term_codes);
    }
    if !formula.intercept {
        'found: for (j, &term) in formula.terms.iter().enumerate() {
            for i in 0..nvar {
                if term >> i & 1 == 1 && frame.values[i].n_levels() > 1 {
                    codes[j][i] = 2;
                    break 'found;
                }
            }
        }
    }

    let mut names: Vec<String> = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut column_terms: Vec<Option<usize>> = Vec::new();
    if formula.intercept {
        names.push("(Intercept)".to_string());
        columns.push(vec![1.0; n]);
        column_terms.push(None);
    }
    for (j, &term) in formula.terms.iter().enumerate() {
        let mut term_cols: Vec<(String, Vec<f64>)> = vec![(String::new(), vec![1.0; n])];
        for i in 0..nvar {
            if term >> i & 1 == 0 {
                continue;
            }
            let label = &formula.variables[i];
            let pieces: Vec<(String, Vec<f64>)> = match &frame.values[i] {
                FrameValue::Numeric(v) => vec![(label.clone(), v.clone())],
                FrameValue::Matrix { columns, names } => {
                    if columns.len() == 1 {
                        vec![(label.clone(), columns[0].clone())]
                    } else {
                        columns.iter().zip(names).map(|(c, nm)| (format!("{}{}", label, nm), c.clone())).collect()
                    }
                }
                FrameValue::Factor { codes: fc, levels, ordered } => {
                    if codes[j][i] == 1 && *ordered {
                        let (contr, cnames) = super::rfuns::contr_poly(levels.len())
                            .map_err(|e| DeseqError::InvalidDesignMatrix { reason: format!("{}: {}", label, e) })?;
                        (0..contr.ncols())
                            .map(|k| (format!("{}{}", label, cnames[k]), fc.iter().map(|&c| contr[[c, k]]).collect()))
                            .collect()
                    } else {
                        let skip = if codes[j][i] == 1 { 1 } else { 0 };
                        (skip..levels.len())
                            .map(|l| {
                                (format!("{}{}", label, levels[l]), fc.iter().map(|&c| if c == l { 1.0 } else { 0.0 }).collect())
                            })
                            .collect()
                    }
                }
            };
            let mut next = Vec::with_capacity(term_cols.len() * pieces.len());
            for (pname, pvals) in &pieces {
                for (cname, cvals) in &term_cols {
                    let name = if cname.is_empty() { pname.clone() } else { format!("{}:{}", cname, pname) };
                    // `+ 0.0` turns the -0 of a negative value times 0 into 0, as R's
                    // nOrMoreInCell(), which pastes the rows, does not tell them apart
                    next.push((name, cvals.iter().zip(pvals).map(|(a, b)| a * b + 0.0).collect()));
                }
            }
            term_cols = next;
        }
        for (name, vals) in term_cols {
            names.push(name);
            columns.push(vals);
            column_terms.push(Some(j));
        }
    }

    let mut matrix = Array2::zeros((n, columns.len()));
    for (k, col) in columns.iter().enumerate() {
        for (s, &v) in col.iter().enumerate() {
            matrix[[s, k]] = v;
        }
    }
    Ok(ModelMatrix { matrix, column_names: names, column_terms })
}

/// DESeq2's coefficient names (resultsNames) for the columns of a model matrix
/// R equivalent: fitNbinomGLMs(): `(Intercept)` -> `Intercept`, make.names(), and, for
/// designs with an intercept (renameCols = hasIntercept), renameModelMatrixColumns():
/// the columns `make.names(<factor><level>)` of the colData factors among
/// all.vars(design) become `<factor>_<level>_vs_<reference>`. Columns of expressions
/// (`factor(x)2`, `relevel(cond, "B")A`, ...) and of logical or numeric columns keep
/// their make.names() names.
pub fn deseq_coefficient_names(formula: &Formula, column_names: &[String], frame: &ModelFrame) -> Vec<String> {
    let mut names: Vec<String> = column_names
        .iter()
        .map(|n| if n == "(Intercept)" { "Intercept".to_string() } else { n.clone() })
        .map(|n| make_names(&n))
        .collect();
    if formula.intercept {
        let original = names.clone();
        let mut renames: Vec<(usize, String)> = Vec::new();
        for v in &formula.all_vars {
            let Some(Column::Factor { levels: lv, .. }) = frame.columns.get(v) else { continue };
            for l in &lv[1..] {
                let from = make_names(&format!("{}{}", v, l));
                if let Some(idx) = original.iter().position(|n| *n == from) {
                    renames.push((idx, make_names(&format!("{}_{}_vs_{}", v, l, lv[0]))));
                }
            }
        }
        for (idx, to) in renames {
            names[idx] = to;
        }
    }
    names
}

/// The checks DESeqDataSet() makes of the design formula's variables
/// R equivalent: DESeqDataSet(): all.vars(design) must be colData columns; a design
/// with a single variable needs two values; colData factors need two levels
fn check_design_variables(formula: &Formula, frame: &ModelFrame) -> Result<()> {
    if let [v] = formula.all_vars.as_slice() {
        if frame.columns.get(v).is_some_and(|c| c.is_constant()) {
            return Err(DeseqError::InvalidDesignMatrix {
                reason: "design has a single variable, with all samples having the same value.\n  use instead a \
                         design of '~ 1'. estimateSizeFactors, rlog and the VST can then be used"
                    .to_string(),
            });
        }
    }
    let single: Vec<&str> = formula
        .all_vars
        .iter()
        .filter(|v| matches!(frame.columns.get(*v), Some(c @ Column::Factor { .. }) if c.is_constant()))
        .map(|v| v.as_str())
        .collect();
    if !single.is_empty() {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!(
                "design contains one or more variables with all samples having the same value,\n  remove these \
                 variables from the design ({})",
                single.join(", ")
            ),
        });
    }
    Ok(())
}

/// Model matrix and design information of a formula
/// R equivalent: DESeqDataSet(design = formula) (its checks of the design variables,
/// model.matrix() and checkFullRank()) and the column naming of fitNbinomGLMs()
pub fn formula_design(
    formula: &Formula,
    metadata: &SampleMetadata,
    reference_levels: &HashMap<String, String>,
) -> Result<(Array2<f64>, DesignInfo)> {
    let frame = ModelFrame::new(formula, metadata, reference_levels)?;
    check_design_variables(formula, &frame)?;
    let frame = frame.evaluate(formula)?;
    let ModelMatrix { matrix, column_names, column_terms } = model_matrix(formula, &frame)?;
    if matrix.ncols() == 0 {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!("the design formula '{}' has no columns", formula.display()),
        });
    }
    check_full_rank(&matrix)?;
    let coef_names = deseq_coefficient_names(formula, &column_names, &frame);

    // The colData factors of the design (R: levels(colData(object)[[v]])): the
    // contrasts of results() and R's default contrast use these
    let factor_levels = |v: &str| match frame.columns.get(v) {
        Some(Column::Factor { levels, .. }) => Some(levels.clone()),
        _ => None,
    };
    // R: the default contrast variable is the last variable of the design if it is a
    // factor; the dataset's main variable is the last factor
    let design_variable = formula.all_vars.iter().rev().find(|v| factor_levels(v).is_some()).cloned().unwrap_or_default();
    let levels = factor_levels(&design_variable).unwrap_or_default();
    let reference_levels: HashMap<String, String> = formula
        .all_vars
        .iter()
        .filter_map(|v| factor_levels(v).map(|lv| (v.clone(), lv[0].clone())))
        .collect();

    // Main-effect columns of each metadata column used as a plain variable
    let mut factor_columns: HashMap<String, Vec<usize>> = HashMap::new();
    let mut continuous_columns: HashMap<String, usize> = HashMap::new();
    for (col, term) in column_terms.iter().enumerate() {
        let Some(j) = *term else { continue };
        let t = formula.terms[j];
        if t.count_ones() != 1 {
            continue;
        }
        let i = t.trailing_zeros() as usize;
        let Ex::Sym(v) = &formula.exprs[i] else { continue };
        match frame.columns.get(v) {
            Some(Column::Factor { .. }) => factor_columns.entry(v.clone()).or_default().push(col),
            Some(Column::Numeric(_)) => {
                continuous_columns.insert(v.clone(), col);
            }
            _ => {}
        }
    }

    let info = DesignInfo {
        coef_names,
        design_variable,
        reference_level: levels.first().cloned().unwrap_or_default(),
        reference_levels,
        levels,
        factor_columns,
        continuous_columns,
        interaction_columns: HashMap::new(),
        has_intercept: formula.intercept,
        design_vars: formula.all_vars.clone(),
        interaction_present: formula.has_interactions(),
    };
    Ok((matrix, info))
}

/// The model matrix of a reduced formula for the LRT
/// R equivalent: model.matrix(reduced, colData) in nbinomLRT(), after checkLRT():
/// every variable of the reduced formula must be in the full formula, and the reduced
/// model must have fewer columns ("less than one degree of freedom"). R does not
/// check the rank of the reduced model matrix; a rank-deficient one is fitted from
/// fitNbinomGLMs()'s start for that case (see [`crate::testing::likelihood_ratio_test_reduced`]).
pub fn reduced_design(
    reduced: &Formula,
    full_variables: &[String],
    full_columns: usize,
    metadata: &SampleMetadata,
    reference_levels: &HashMap<String, String>,
) -> Result<Array2<f64>> {
    let missing: Vec<&str> = reduced.all_vars.iter().filter(|v| !full_variables.contains(v)).map(|v| v.as_str()).collect();
    if !missing.is_empty() {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!(
                "the following variables in the reduced formula not in the full formula: {}",
                missing.join(", ")
            ),
        });
    }
    let frame = ModelFrame::new(reduced, metadata, reference_levels)?.evaluate(reduced)?;
    let matrix = model_matrix(reduced, &frame)?.matrix;
    if matrix.ncols() == 0 {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!("the reduced formula '{}' has no columns", reduced.display()),
        });
    }
    if full_columns <= matrix.ncols() {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: "less than one degree of freedom, perhaps full and reduced models are not in the correct order"
                .to_string(),
        });
    }
    // R: stopifnot(all(colSums(abs(modelMatrix)) > 0)) in fitNbinomGLMs()
    if (0..matrix.ncols()).any(|j| matrix.column(j).iter().all(|&v| v == 0.0)) {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!(
                "the reduced model matrix of '{}' has a column of zeros (R: all(colSums(abs(modelMatrix)) > 0) is not TRUE)",
                reduced.display()
            ),
        });
    }
    Ok(matrix)
}

#[cfg(test)]
#[path = "formula_tests.rs"]
mod tests;
