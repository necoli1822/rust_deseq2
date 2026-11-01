//! R model formulas (`~ a + b + a:b`, `~ 0 + a`, `~ a * b * c`, ...) and the model
//! matrices R builds from them
//!
//! Variables are factors unless they are numeric (continuous) columns of the
//! sample metadata. Factors use treatment contrasts, as `model.matrix()` does with
//! R's default `contrasts` option.

use std::collections::HashMap;

use ndarray::Array2;

use super::design::{check_full_rank, DesignInfo};
use crate::data::SampleMetadata;
use crate::error::{DeseqError, Result};

/// The right-hand side of a model formula
/// R equivalent: terms.formula() (variables, term.labels, order, intercept)
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    /// The variables in order of first appearance, including any removed by `-`
    /// (R: all.vars(formula))
    pub variables: Vec<String>,
    /// The terms as bit sets over `variables`, ordered by degree and, within a degree,
    /// by their position in the formula (R: terms(keep.order = FALSE))
    pub terms: Vec<u64>,
    /// Whether the model has an intercept (R: attr(terms(formula), "intercept"))
    pub intercept: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Name(String),
    Number(String),
    Plus,
    Minus,
    Star,
    Slash,
    Colon,
    Caret,
    In,
    LParen,
    RParen,
}

fn formula_error(text: &str, reason: impl std::fmt::Display) -> DeseqError {
    DeseqError::InvalidDesignMatrix {
        reason: format!("invalid model formula '{}': {}", text.trim(), reason),
    }
}

/// Split a formula into tokens; a leading `~` is optional
fn tokenize(text: &str) -> Result<Vec<Token>> {
    let mut chars = text.trim().chars().peekable();
    if chars.peek() == Some(&'~') {
        chars.next();
    }
    let mut tokens = Vec::new();
    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            }
            '+' | '-' | '*' | '/' | ':' | '^' | '(' | ')' => {
                chars.next();
                tokens.push(match c {
                    '+' => Token::Plus,
                    '-' => Token::Minus,
                    '*' => Token::Star,
                    '/' => Token::Slash,
                    ':' => Token::Colon,
                    '^' => Token::Caret,
                    '(' => Token::LParen,
                    _ => Token::RParen,
                });
            }
            '%' => {
                chars.next();
                let op: String = chars.by_ref().take_while(|&c| c != '%').collect();
                if op == "in" {
                    tokens.push(Token::In);
                } else {
                    return Err(formula_error(text, format!("operator %{}% is not supported", op)));
                }
            }
            '`' => {
                chars.next();
                let mut name = String::new();
                loop {
                    match chars.next() {
                        Some('`') => break,
                        Some(c) => name.push(c),
                        None => return Err(formula_error(text, "unterminated backquoted name")),
                    }
                }
                tokens.push(Token::Name(name));
            }
            c if c.is_ascii_digit() => {
                let mut num = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Number(num));
            }
            c if c.is_alphabetic() || c == '.' => {
                let mut name = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_alphanumeric() || d == '.' || d == '_' {
                        name.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Name(name));
            }
            '~' => return Err(formula_error(text, "a response (left-hand side) is not allowed; write '~ a + b'")),
            other => return Err(formula_error(text, format!("unexpected character '{}'", other))),
        }
    }
    Ok(tokens)
}

/// Expression tree of the formula
#[derive(Debug, Clone)]
enum Expr {
    Var(String),
    Number(String),
    Plus(Box<Expr>, Box<Expr>),
    Minus(Option<Box<Expr>>, Box<Expr>),
    Colon(Box<Expr>, Box<Expr>),
    Star(Box<Expr>, Box<Expr>),
    Slash(Box<Expr>, Box<Expr>),
    In(Box<Expr>, Box<Expr>),
    Power(Box<Expr>, Box<Expr>),
}

/// Recursive-descent parser with R's operator precedence (highest first):
/// `^`, unary `-`/`+`, `:`, `%in%`, `*` `/`, binary `+` `-`
struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    text: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn sum(&mut self) -> Result<Expr> {
        let mut left = self.product()?;
        while let Some(op @ (Token::Plus | Token::Minus)) = self.peek().cloned() {
            self.pos += 1;
            let right = self.product()?;
            left = if op == Token::Plus {
                Expr::Plus(Box::new(left), Box::new(right))
            } else {
                Expr::Minus(Some(Box::new(left)), Box::new(right))
            };
        }
        Ok(left)
    }

    fn product(&mut self) -> Result<Expr> {
        let mut left = self.in_op()?;
        while let Some(op @ (Token::Star | Token::Slash)) = self.peek().cloned() {
            self.pos += 1;
            let right = self.in_op()?;
            left = if op == Token::Star {
                Expr::Star(Box::new(left), Box::new(right))
            } else {
                Expr::Slash(Box::new(left), Box::new(right))
            };
        }
        Ok(left)
    }

    fn in_op(&mut self) -> Result<Expr> {
        let mut left = self.colon()?;
        while self.peek() == Some(&Token::In) {
            self.pos += 1;
            let right = self.colon()?;
            left = Expr::In(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn colon(&mut self) -> Result<Expr> {
        let mut left = self.unary()?;
        while self.peek() == Some(&Token::Colon) {
            self.pos += 1;
            let right = self.unary()?;
            left = Expr::Colon(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Token::Minus) => {
                self.pos += 1;
                Ok(Expr::Minus(None, Box::new(self.unary()?)))
            }
            Some(Token::Plus) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr> {
        let base = self.atom()?;
        if self.peek() == Some(&Token::Caret) {
            self.pos += 1;
            let exponent = self.unary()?;
            return Ok(Expr::Power(Box::new(base), Box::new(exponent)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Expr> {
        match self.next() {
            Some(Token::Name(name)) => {
                if self.peek() == Some(&Token::LParen) {
                    return Err(formula_error(
                        self.text,
                        format!(
                            "functions such as {}() are not supported; add the transformed variable to the metadata",
                            name
                        ),
                    ));
                }
                Ok(Expr::Var(name))
            }
            Some(Token::Number(n)) => Ok(Expr::Number(n)),
            Some(Token::LParen) => {
                let inner = self.sum()?;
                match self.next() {
                    Some(Token::RParen) => Ok(inner),
                    _ => Err(formula_error(self.text, "missing ')'")),
                }
            }
            Some(t) => Err(formula_error(self.text, format!("unexpected {:?}", t))),
            None => Err(formula_error(self.text, "unexpected end of formula")),
        }
    }
}

/// Builds the term list of an expression
/// R equivalent: EncodeVars() and the term algebra of model.c (PlusTerms, DeleteTerms,
/// InteractTerms, CrossTerms, InTerms, NestTerms, PowerTerms, TrimRepeats)
struct Encoder<'a> {
    variables: Vec<String>,
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
    fn var_bit(&mut self, name: &str) -> Result<u64> {
        let idx = match self.variables.iter().position(|v| v == name) {
            Some(i) => i,
            None => {
                self.variables.push(name.to_string());
                self.variables.len() - 1
            }
        };
        if idx >= 64 {
            return Err(formula_error(self.text, "more than 64 variables"));
        }
        Ok(1u64 << idx)
    }

    fn encode(&mut self, e: &Expr) -> Result<Vec<u64>> {
        match e {
            Expr::Var(name) => Ok(vec![self.var_bit(name)?]),
            Expr::Number(n) => {
                // R: isOne / isZero set the intercept (reversed under `-`)
                match n.parse::<f64>() {
                    Ok(v) if v == 1.0 => self.intercept = self.parity,
                    Ok(v) if v == 0.0 => self.intercept = !self.parity,
                    _ => return Err(formula_error(self.text, format!("invalid term '{}'", n))),
                }
                Ok(Vec::new())
            }
            Expr::Plus(a, b) => {
                let mut left = self.encode(a)?;
                left.extend(self.encode(b)?);
                Ok(trim_repeats(left))
            }
            Expr::Minus(a, b) => {
                // R: DeleteTerms()
                let left = match a {
                    Some(a) => self.encode(a)?,
                    None => Vec::new(),
                };
                self.parity = !self.parity;
                let right = self.encode(b);
                self.parity = !self.parity;
                let right = right?;
                Ok(left.into_iter().filter(|t| !right.contains(t)).collect())
            }
            Expr::Colon(a, b) => {
                let left = self.encode(a)?;
                let right = self.encode(b)?;
                Ok(interact(&left, &right))
            }
            Expr::Star(a, b) => {
                // R: CrossTerms() -- a + b + a:b
                let left = self.encode(a)?;
                let right = self.encode(b)?;
                let mut out = left.clone();
                out.extend(right.iter().copied());
                out.extend(interact(&left, &right));
                Ok(trim_repeats(out))
            }
            Expr::In(a, b) => {
                // R: InTerms() -- every left term with all right variables
                let left = self.encode(a)?;
                let right = self.encode(b)?;
                let all = right.iter().fold(0u64, |acc, &t| acc | t);
                Ok(trim_repeats(left.iter().map(|&l| l | all).collect()))
            }
            Expr::Slash(a, b) => {
                // R: NestTerms() -- a / b = a + b %in% a
                let left = self.encode(a)?;
                let right = self.encode(b)?;
                let all = left.iter().fold(0u64, |acc, &t| acc | t);
                let mut out = left.clone();
                out.extend(right.iter().map(|&r| r | all));
                Ok(trim_repeats(out))
            }
            Expr::Power(a, b) => {
                // R: PowerTerms() -- all interactions up to the given order
                let n = match &**b {
                    Expr::Number(n) => n.parse::<f64>().ok().filter(|v| *v >= 1.0 && v.fract() == 0.0),
                    _ => None,
                }
                .ok_or_else(|| formula_error(self.text, "the exponent of '^' must be a positive integer"))?
                    as usize;
                let left = self.encode(a)?;
                let mut out = left.clone();
                for _ in 1..n {
                    let mut next = out.clone();
                    next.extend(interact(&out, &left));
                    out = trim_repeats(next);
                }
                Ok(out)
            }
        }
    }
}

impl Formula {
    /// Parse the right-hand side of an R formula, e.g. `"~ batch + condition"`,
    /// `"~ genotype * treatment"`, `"~ 0 + condition"`, `"~ (a + b + c)^2"`
    /// R equivalent: terms(as.formula(text))
    ///
    /// Supported: `+`, `-`, `:`, `*`, `/`, `%in%`, `^`, parentheses, `0`/`1`/`- 1`
    /// for the intercept and backquoted names. Functions (`log(x)`, `I(x^2)`,
    /// `factor(x)`, ...) and `.` are rejected.
    pub fn parse(text: &str) -> Result<Formula> {
        let tokens = tokenize(text)?;
        if tokens.is_empty() {
            return Err(formula_error(text, "empty formula (use '~ 1' for an intercept-only model)"));
        }
        let mut parser = Parser { tokens, pos: 0, text };
        let expr = parser.sum()?;
        if parser.pos != parser.tokens.len() {
            return Err(formula_error(text, format!("unexpected {:?}", parser.tokens[parser.pos])));
        }
        let mut enc = Encoder { variables: Vec::new(), intercept: true, parity: true, text };
        let terms = enc.encode(&expr)?;
        if enc.variables.iter().any(|v| v == ".") {
            return Err(formula_error(text, "'.' (all other columns) is not supported"));
        }
        // R: terms.formula(keep.order = FALSE) sorts terms by degree, keeping the
        // formula order within a degree
        let mut terms = trim_repeats(terms);
        terms.sort_by_key(|t| t.count_ones());
        Ok(Formula { variables: enc.variables, terms, intercept: enc.intercept })
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

    /// The formula as R prints it, e.g. `~ batch + condition + batch:condition`
    pub fn display(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.intercept {
            parts.push("0".to_string());
        }
        parts.extend(self.term_labels());
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

/// The coding of the variables of a formula
#[derive(Debug, Clone)]
pub struct VariableCoding {
    /// Levels of each factor variable, reference level first (R: levels(colData[[v]]))
    pub levels: HashMap<String, Vec<String>>,
}

impl VariableCoding {
    /// Levels of the factors of `formula`: sorted as R's factor() sorts them, with a
    /// reference level from `reference_levels` moved to the front (R: relevel())
    pub fn new(formula: &Formula, metadata: &SampleMetadata, reference_levels: &HashMap<String, String>) -> Result<Self> {
        let mut levels = HashMap::new();
        for v in &formula.variables {
            if metadata.has_continuous(v) {
                continue;
            }
            if !metadata.has_condition(v) {
                return Err(DeseqError::InvalidDesignMatrix {
                    reason: format!("all variables in the design formula must be columns in the metadata: '{}' is not", v),
                });
            }
            let mut lv = metadata.get_levels(v)?;
            if let Some(reference) = reference_levels.get(v) {
                let pos = lv.iter().position(|l| l == reference).ok_or_else(|| DeseqError::InvalidInput {
                    reason: format!("Reference level '{}' does not exist in factor '{}'. Available levels: {:?}", reference, v, lv),
                })?;
                let r = lv.remove(pos);
                lv.insert(0, r);
            }
            levels.insert(v.clone(), lv);
        }
        Ok(Self { levels })
    }

    pub fn is_factor(&self, variable: &str) -> bool {
        self.levels.contains_key(variable)
    }
}

/// A model matrix with R's column names
#[derive(Debug, Clone)]
pub struct ModelMatrix {
    pub matrix: Array2<f64>,
    /// R: colnames(model.matrix(...)), e.g. `(Intercept)`, `batchb2`, `batchb2:conditionB`
    pub column_names: Vec<String>,
    /// Index into `formula.terms` of each column; `None` for the intercept
    pub column_terms: Vec<Option<usize>>,
}

/// R's model matrix of a formula
/// R equivalent: model.matrix(formula, colData) with treatment contrasts
///
/// Columns: `(Intercept)` if present, then each term in `formula.terms` order. A factor
/// in a term is coded by contrasts (non-reference levels) when the term without it is
/// the empty term or is contained in a preceding term, and by indicators of all levels
/// otherwise; without an intercept, the first factor of the first term containing one
/// is coded by indicators (R: TermCode() and the no-intercept rule of modelmatrix() in
/// model.c). Interaction columns are products, the first variable varying fastest,
/// named `a1:b2`.
pub fn model_matrix(formula: &Formula, metadata: &SampleMetadata, coding: &VariableCoding) -> Result<ModelMatrix> {
    let n = metadata.n_samples();
    for v in &formula.variables {
        if let Some(lv) = coding.levels.get(v) {
            if lv.len() < 2 {
                return Err(DeseqError::InvalidDesignMatrix {
                    reason: format!(
                        "the factor '{}' in the design formula has a single level ('{}'); contrasts can be \
                         applied only to factors with 2 or more levels",
                        v,
                        lv.first().map(|s| s.as_str()).unwrap_or("")
                    ),
                });
            }
        } else if !metadata.has_continuous(v) {
            return Err(DeseqError::InvalidDesignMatrix {
                reason: format!("all variables in the design formula must be columns in the metadata: '{}' is not", v),
            });
        }
    }

    // R: TermCode() -- 1 = contrasts, 2 = indicators of every level
    let mut codes: Vec<Vec<u8>> = Vec::with_capacity(formula.terms.len());
    for (j, &term) in formula.terms.iter().enumerate() {
        let mut term_codes = vec![0u8; formula.variables.len()];
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
            for (i, v) in formula.variables.iter().enumerate() {
                if term >> i & 1 == 1 && coding.is_factor(v) {
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
        for (i, v) in formula.variables.iter().enumerate() {
            if term >> i & 1 == 0 {
                continue;
            }
            let pieces: Vec<(String, Vec<f64>)> = match coding.levels.get(v) {
                None => vec![(v.clone(), (0..n).map(|s| metadata.get_continuous_value(v, s)).collect::<Result<Vec<f64>>>()?)],
                Some(lv) => {
                    let values = metadata.condition(v).expect("checked above");
                    let skip = if codes[j][i] == 1 { 1 } else { 0 };
                    lv[skip..]
                        .iter()
                        .map(|l| (format!("{}{}", v, l), values.iter().map(|x| if x == l { 1.0 } else { 0.0 }).collect()))
                        .collect()
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
/// main-effect columns `<factor><level>` become `<factor>_<level>_vs_<reference>`
pub fn deseq_coefficient_names(formula: &Formula, column_names: &[String], coding: &VariableCoding) -> Vec<String> {
    let mut names: Vec<String> = column_names
        .iter()
        .map(|n| if n == "(Intercept)" { "Intercept".to_string() } else { n.clone() })
        .map(|n| make_names(&n))
        .collect();
    if formula.intercept {
        let original = names.clone();
        let mut renames: Vec<(usize, String)> = Vec::new();
        for v in &formula.variables {
            let Some(lv) = coding.levels.get(v) else { continue };
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

/// Model matrix and design information of a formula
/// R equivalent: getModelMatrix() / fitNbinomGLMs() column naming, with checkFullRank()
/// and the checks of designAndArgChecker()
pub fn formula_design(
    formula: &Formula,
    metadata: &SampleMetadata,
    reference_levels: &HashMap<String, String>,
) -> Result<(Array2<f64>, DesignInfo)> {
    let coding = VariableCoding::new(formula, metadata, reference_levels)?;
    let ModelMatrix { matrix, column_names, column_terms } = model_matrix(formula, metadata, &coding)?;
    if matrix.ncols() == 0 {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!("the design formula '{}' has no columns", formula.display()),
        });
    }
    check_full_rank(&matrix)?;
    let coef_names = deseq_coefficient_names(formula, &column_names, &coding);

    // R: the default contrast variable is the last variable of the design
    let design_variable = formula
        .variables
        .iter()
        .rev()
        .find(|v| coding.is_factor(v))
        .cloned()
        .unwrap_or_default();
    let levels = coding.levels.get(&design_variable).cloned().unwrap_or_default();
    let reference_levels: HashMap<String, String> =
        coding.levels.iter().map(|(v, lv)| (v.clone(), lv[0].clone())).collect();

    // Main-effect columns of each variable
    let mut factor_columns: HashMap<String, Vec<usize>> = HashMap::new();
    let mut continuous_columns: HashMap<String, usize> = HashMap::new();
    for (col, term) in column_terms.iter().enumerate() {
        let Some(j) = *term else { continue };
        let vars = formula.term_variables(formula.terms[j]);
        if let [v] = vars.as_slice() {
            if coding.is_factor(v) {
                factor_columns.entry(v.to_string()).or_default().push(col);
            } else {
                continuous_columns.insert(v.to_string(), col);
            }
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
        design_vars: formula.variables.clone(),
        interaction_present: formula.has_interactions(),
    };
    Ok((matrix, info))
}

/// The model matrix of a reduced formula for the LRT
/// R equivalent: model.matrix(reduced, colData) in nbinomLRT(), after checkLRT():
/// every variable of the reduced formula must be in the full formula, and the reduced
/// model must have fewer columns ("less than one degree of freedom")
pub fn reduced_design(
    reduced: &Formula,
    full_variables: &[String],
    full_columns: usize,
    metadata: &SampleMetadata,
    reference_levels: &HashMap<String, String>,
) -> Result<Array2<f64>> {
    let missing: Vec<&str> = reduced.variables.iter().filter(|v| !full_variables.contains(v)).map(|v| v.as_str()).collect();
    if !missing.is_empty() {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!(
                "the following variables in the reduced formula not in the full formula: {}",
                missing.join(", ")
            ),
        });
    }
    let coding = VariableCoding::new(reduced, metadata, reference_levels)?;
    let matrix = model_matrix(reduced, metadata, &coding)?.matrix;
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
    // R fits a rank-deficient reduced model anyway (with a different start), which is
    // not a nested model test; it is an error here.
    check_full_rank(&matrix).map_err(|e| DeseqError::InvalidDesignMatrix {
        reason: format!("reduced formula '{}': {}", reduced.display(), e),
    })?;
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The colData of the R check (validation notes): batch b1/b2 alternating,
    /// condition A/B/C in blocks of 4, genotype WT/KO (WT reference), numeric x and z
    fn metadata() -> SampleMetadata {
        let ids: Vec<String> = (1..=12).map(|i| format!("s{}", i)).collect();
        let mut m = SampleMetadata::new(ids);
        m.add_condition("batch", (0..12).map(|i| if i % 2 == 0 { "b1" } else { "b2" }.to_string()).collect()).unwrap();
        m.add_condition("condition", (0..12).map(|i| ["A", "B", "C"][i / 4].to_string()).collect()).unwrap();
        m.add_condition("genotype", (0..12).map(|i| if i < 6 { "WT" } else { "KO" }.to_string()).collect()).unwrap();
        let x = [-0.626, 0.184, -0.836, 1.595, 0.33, -0.82, 0.487, 0.738, 0.576, -0.305, 1.512, 0.39];
        let z = [-0.621, -2.215, 1.125, -0.045, -0.016, 0.944, 0.821, 0.594, 0.919, 0.782, 0.075, -1.989];
        for (name, v) in [("x", x), ("z", z)] {
            m.add_condition(name, v.iter().map(|a| a.to_string()).collect()).unwrap();
            m.add_continuous(name, v.to_vec()).unwrap();
        }
        m
    }

    /// resultsNames, rank and sum(mm * row index) as printed by R 4.5.3 / DESeq2 1.50.2
    /// for model.matrix(f, colData) with the renaming of fitNbinomGLMs()
    fn check(f: &str, expected_names: &str, rank: usize, weighted_sum: f64) {
        let meta = metadata();
        let refs = HashMap::from([("genotype".to_string(), "WT".to_string())]);
        let formula = Formula::parse(f).unwrap();
        let coding = VariableCoding::new(&formula, &meta, &refs).unwrap();
        let ModelMatrix { matrix: mm, column_names, .. } = model_matrix(&formula, &meta, &coding).unwrap();
        let names = deseq_coefficient_names(&formula, &column_names, &coding);
        assert_eq!(names.join(","), expected_names, "names of {}", f);
        assert_eq!(super::super::design::qr_rank(&mm), rank, "rank of {}", f);
        let s: f64 = (0..mm.nrows()).map(|i| (i + 1) as f64 * mm.row(i).sum()).sum();
        assert!((s - weighted_sum).abs() < 1e-9, "sum of {}: {} vs {}", f, s, weighted_sum);
    }

    #[test]
    fn test_model_matrix_matches_r() {
        check("~ batch + condition", "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A", 4, 188.0);
        let inter = "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,batchb2.conditionB,batchb2.conditionC";
        check("~ batch + condition + batch:condition", inter, 6, 224.0);
        check("~ batch * condition", inter, 6, 224.0);
        check(
            "~ condition:batch + batch + condition",
            "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,conditionB.batchb2,conditionC.batchb2",
            6,
            224.0,
        );
        let nested = "Intercept,condition_B_vs_A,condition_C_vs_A,conditionA.batchb2,conditionB.batchb2,conditionC.batchb2";
        check("~ condition + batch:condition", nested, 6, 188.0);
        check("~ condition / batch", nested, 6, 188.0);
        check(
            "~ batch:condition",
            "Intercept,batchb1.conditionA,batchb2.conditionA,batchb1.conditionB,batchb2.conditionB,batchb1.conditionC,batchb2.conditionC",
            6,
            156.0,
        );
        for f in ["~ 0 + condition", "~ condition - 1", "~ condition + 0", "~ -1 + condition"] {
            check(f, "conditionA,conditionB,conditionC", 3, 78.0);
        }
        check("~ 0 + batch + condition", "batchb1,batchb2,conditionB,conditionC", 4, 146.0);
        check(
            "~ 0 + batch:condition",
            "batchb1.conditionA,batchb2.conditionA,batchb1.conditionB,batchb2.conditionB,batchb1.conditionC,batchb2.conditionC",
            6,
            78.0,
        );
        check("~ 0 + x + condition", "x,conditionA,conditionB,conditionC", 4, 111.103);
        check("~ x + condition + x:condition", "Intercept,x,condition_B_vs_A,condition_C_vs_A,x.conditionB,x.conditionC", 6, 208.592);
        check("~ condition + condition:x", "Intercept,condition_B_vs_A,condition_C_vs_A,conditionA.x,conditionB.x,conditionC.x", 6, 179.103);
        check("~ x * z", "Intercept,x,z,x.z", 4, 110.795787);
        check("~ x:z", "Intercept,x.z", 2, 70.417787);
        check(
            "~ batch * condition * genotype",
            "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,genotype_KO_vs_WT,batchb2.conditionB,batchb2.conditionC,\
             batchb2.genotypeKO,conditionB.genotypeKO,conditionC.genotypeKO,batchb2.conditionB.genotypeKO,batchb2.conditionC.genotypeKO",
            8,
            398.0,
        );
        check(
            "~ (batch + condition + genotype)^2",
            "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,genotype_KO_vs_WT,batchb2.conditionB,batchb2.conditionC,\
             batchb2.genotypeKO,conditionB.genotypeKO,conditionC.genotypeKO",
            8,
            368.0,
        );
        check("~ batch * condition - batch", "Intercept,condition_B_vs_A,condition_C_vs_A,batchb2.conditionA,batchb2.conditionB,batchb2.conditionC", 6, 188.0);
        check("~ 1", "Intercept", 1, 78.0);
        check("~ 0 + condition:x", "conditionA.x,conditionB.x,conditionC.x", 3, 33.103);
        check(
            "~ genotype + condition %in% genotype",
            "Intercept,genotype_KO_vs_WT,genotypeWT.conditionB,genotypeKO.conditionB,genotypeWT.conditionC,genotypeKO.conditionC",
            4,
            203.0,
        );
        check("~ x + batch:x", "Intercept,x,x.batchb2", 3, 120.465);
        check(
            "~ 0 + condition + condition:batch",
            "conditionA,conditionB,conditionC,conditionA.batchb2,conditionB.batchb2,conditionC.batchb2",
            6,
            120.0,
        );
    }

    #[test]
    fn test_formula_terms() {
        let f = Formula::parse("~ batch * condition * genotype").unwrap();
        assert_eq!(f.variables, vec!["batch", "condition", "genotype"]);
        assert_eq!(
            f.term_labels(),
            vec!["batch", "condition", "genotype", "batch:condition", "batch:genotype", "condition:genotype", "batch:condition:genotype"]
        );
        assert!(f.intercept && f.has_interactions());
        let f = Formula::parse("condition:batch + batch").unwrap();
        assert_eq!(f.term_labels(), vec!["batch", "condition:batch"]);
        assert!(!Formula::parse("~ 0 + a").unwrap().intercept);
        assert!(Formula::parse("~ a - 1 + 1").unwrap().intercept);
        assert!(Formula::parse("~ log(x)").is_err());
        assert!(Formula::parse("y ~ x").is_err());
        assert!(Formula::parse("~ a +").is_err());
        assert!(Formula::parse("~ (a + b").is_err());
        assert_eq!(Formula::parse("~ `my var` + b").unwrap().variables, vec!["my var", "b"]);
    }

    #[test]
    fn test_make_names() {
        let cases = [
            ("", "X"), ("1a", "X1a"), (".1a", "X.1a"), ("a b", "a.b"), ("if", "if."), ("a:b", "a.b"), ("_x", "X_x"),
            (".x", ".x"), ("TRUE", "TRUE."), ("a-b", "a.b"), ("é", "é"),
        ];
        for (input, expected) in cases {
            assert_eq!(make_names(input), expected, "make.names({:?})", input);
        }
    }

    /// R cleanContrast(): column choice for character contrasts with and without intercept
    #[test]
    fn test_contrast_vectors() {
        use super::super::design::contrast_vector;
        let meta = metadata();
        let refs = HashMap::new();
        let design = |f: &str| formula_design(&Formula::parse(f).unwrap(), &meta, &refs).unwrap().1;
        // ~ 0 + batch + condition: batchb1, batchb2, conditionB, conditionC
        let info = design("~ 0 + batch + condition");
        assert_eq!(contrast_vector(&info, "condition", "C", "A").unwrap(), vec![0.0, 0.0, 0.0, 1.0]);
        assert_eq!(contrast_vector(&info, "condition", "A", "C").unwrap(), vec![0.0, 0.0, 0.0, -1.0]);
        assert_eq!(contrast_vector(&info, "condition", "C", "B").unwrap(), vec![0.0, 0.0, -1.0, 1.0]);
        assert_eq!(contrast_vector(&info, "batch", "b2", "b1").unwrap(), vec![-1.0, 1.0, 0.0, 0.0]);
        // ~ 0 + condition: the first variable uses both columns
        let info = design("~ 0 + condition");
        assert_eq!(contrast_vector(&info, "condition", "B", "A").unwrap(), vec![-1.0, 1.0, 0.0]);
        // with an intercept: the renamed main-effect columns
        let info = design("~ batch * condition");
        assert_eq!(contrast_vector(&info, "condition", "C", "B").unwrap(), vec![0.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
        assert!(contrast_vector(&info, "condition", "D", "A").is_err());
        assert!(contrast_vector(&info, "genotype", "KO", "WT").is_err());
    }

    #[test]
    fn test_full_rank_and_reduced() {
        let meta = metadata();
        let refs = HashMap::new();
        let err = formula_design(&Formula::parse("~ batch:condition").unwrap(), &meta, &refs).unwrap_err();
        assert!(format!("{}", err).contains("not full rank"));
        let (mm, info) = formula_design(&Formula::parse("~ batch * condition").unwrap(), &meta, &refs).unwrap();
        assert_eq!(mm.ncols(), 6);
        assert_eq!(info.factor_columns["batch"], vec![1]);
        assert_eq!(info.factor_columns["condition"], vec![2, 3]);
        assert!(info.interaction_present && info.has_intercept);
        let full_vars = info.design_vars.clone();
        let red = reduced_design(&Formula::parse("~ batch + condition").unwrap(), &full_vars, 6, &meta, &refs).unwrap();
        assert_eq!(red.ncols(), 4);
        assert!(reduced_design(&Formula::parse("~ genotype").unwrap(), &full_vars, 6, &meta, &refs).is_err());
        assert!(reduced_design(&Formula::parse("~ batch * condition").unwrap(), &full_vars, 6, &meta, &refs).is_err());
    }
}
