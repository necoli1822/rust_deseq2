//! The model frame of a design formula: the metadata columns it uses (R: the
//! colData columns of `all.vars(design)`) and the value of each formula variable,
//! evaluated as R's `model.frame()` evaluates `attr(terms, "variables")`
//!
//! Supported functions (with R's argument matching: exact names, unique partial
//! names, then positions): `factor`, `as.factor`, `ordered`, `as.ordered`,
//! `relevel`, `as.numeric`, `as.double`, `as.integer`, `as.character`, `log`
//! (with `base`), `log2`, `log10`, `log1p`, `exp`, `expm1`, `sqrt`, `abs`,
//! `scale`, `I`, `c`, `poly` (orthogonal or `raw = TRUE`), `ns` and `bs` (also as
//! `splines::ns` / `splines::bs`), `offset`, and the operators `+ - * / ^ %% %/%`,
//! comparisons, `!`, `&`, `|` and `%in%`. Other functions, including user-defined R
//! functions, cannot be evaluated here and are an error.

use std::collections::HashMap;

use super::formula::{deparse, Ex, Formula};
use super::rfuns::{self, format_real15, r_pow, SplineArgs};
use crate::data::{r_collate, SampleMetadata};
use crate::error::{DeseqError, Result};

/// A metadata column used by the design (R: the class of `colData(dds)[[v]]`)
#[derive(Debug, Clone)]
pub enum Column {
    /// A factor: the value of each sample and the levels, reference level first
    Factor { values: Vec<String>, levels: Vec<String> },
    /// A numeric column (`--continuous`)
    Numeric(Vec<f64>),
    /// A logical column: every value is TRUE or FALSE (as read.csv() reads them)
    Logical(Vec<bool>),
}

impl Column {
    /// Whether every sample has the same value
    pub fn is_constant(&self) -> bool {
        match self {
            Column::Factor { values, .. } => values.iter().all(|v| *v == values[0]),
            Column::Numeric(v) => v.iter().all(|x| *x == v[0]),
            Column::Logical(v) => v.iter().all(|x| *x == v[0]),
        }
    }
}

/// The value of a formula variable in the model frame, as model.matrix() codes it
#[derive(Debug, Clone)]
pub enum FrameValue {
    /// One numeric column
    Numeric(Vec<f64>),
    /// A numeric matrix (poly(), ns(), bs(), scale()) with its column names
    Matrix { columns: Vec<Vec<f64>>, names: Vec<String> },
    /// A factor (logical and character values become factors, as in model.matrix()):
    /// the 0-based level of each sample
    Factor { codes: Vec<usize>, levels: Vec<String>, ordered: bool },
}

impl FrameValue {
    /// R: nlevels() as modelmatrix() counts it (0 for numeric variables)
    pub fn n_levels(&self) -> usize {
        match self {
            FrameValue::Factor { levels, .. } => levels.len(),
            _ => 0,
        }
    }
}

/// The metadata columns of a formula and the values of its variables
pub struct ModelFrame {
    pub n_samples: usize,
    /// The metadata column of each name in `all.vars(formula)`
    pub columns: HashMap<String, Column>,
    /// The value of each `formula.variables` entry (empty until [`ModelFrame::evaluate`])
    pub values: Vec<FrameValue>,
}

/// The strings read.csv() reads as logical values
fn parse_logical(s: &str) -> Option<bool> {
    match s {
        "TRUE" | "T" | "true" | "True" => Some(true),
        "FALSE" | "F" | "false" | "False" => Some(false),
        _ => None,
    }
}

impl ModelFrame {
    /// The metadata columns of `all.vars(formula)`: numeric if `--continuous`, logical if
    /// every value is TRUE/FALSE, else a factor whose levels are sorted as R's factor()
    /// sorts them with a reference level from `reference_levels` moved to the front
    /// R equivalent: the colData checks of DESeqDataSet() (all variables in the design
    /// formula must be columns in colData) and relevel()
    pub fn new(formula: &Formula, metadata: &SampleMetadata, reference_levels: &HashMap<String, String>) -> Result<Self> {
        let missing: Vec<&str> = formula
            .all_vars
            .iter()
            .filter(|v| !metadata.has_condition(v) && !metadata.has_continuous(v))
            .map(|v| v.as_str())
            .collect();
        if !missing.is_empty() {
            let mut reason =
                format!("all variables in design formula must be columns in colData (not in the metadata: {})", missing.join(", "));
            if missing.contains(&".") {
                reason.push_str(
                    "; '.' (all other columns) is not expanded by DESeqDataSet(), which takes it as a column named '.'",
                );
            }
            if missing.iter().any(|v| ["T", "F", "pi"].contains(v)) {
                reason.push_str("; write TRUE/FALSE instead of T/F (R's all.vars() takes T, F and pi as variables)");
            }
            return Err(DeseqError::InvalidDesignMatrix { reason });
        }
        let n = metadata.n_samples();
        let mut columns = HashMap::new();
        for v in &formula.all_vars {
            let column = if metadata.has_continuous(v) {
                Column::Numeric((0..n).map(|s| metadata.get_continuous_value(v, s)).collect::<Result<Vec<f64>>>()?)
            } else {
                let values = metadata.condition(v).expect("checked above").clone();
                let logical: Option<Vec<bool>> = values.iter().map(|s| parse_logical(s)).collect();
                match logical {
                    Some(l) => Column::Logical(l),
                    None => {
                        let mut lv = metadata.get_levels(v)?;
                        if let Some(reference) = reference_levels.get(v) {
                            let pos = lv.iter().position(|l| l == reference).ok_or_else(|| DeseqError::InvalidInput {
                                reason: format!(
                                    "Reference level '{}' does not exist in factor '{}'. Available levels: {:?}",
                                    reference, v, lv
                                ),
                            })?;
                            let r = lv.remove(pos);
                            lv.insert(0, r);
                        }
                        Column::Factor { values, levels: lv }
                    }
                }
            };
            columns.insert(v.clone(), column);
        }
        Ok(ModelFrame { n_samples: n, columns, values: Vec::new() })
    }

    /// Evaluates every variable of the formula (offsets too, whose values are not used)
    /// R equivalent: model.frame(formula, colData) followed by the conversion of
    /// character and logical variables to factors in model.matrix.default()
    pub fn evaluate(mut self, formula: &Formula) -> Result<Self> {
        let mut values = Vec::with_capacity(formula.exprs.len());
        for (label, e) in formula.variables.iter().zip(&formula.exprs) {
            let wrap = |reason: String| DeseqError::InvalidDesignMatrix {
                reason: format!("in the design formula term '{}': {}", label, reason),
            };
            let v = Eval { frame: &self }.eval(e).map_err(wrap)?;
            values.push(self.frame_value(label, v).map_err(wrap)?);
        }
        self.values = values;
        Ok(self)
    }

    fn frame_value(&self, label: &str, v: Val) -> std::result::Result<FrameValue, String> {
        let n = self.n_samples;
        let check_len = |len: usize| {
            if len == n {
                Ok(())
            } else {
                Err(format!("variable lengths differ (found for '{}'): {} values for {} samples", label, len, n))
            }
        };
        let check_finite = |x: &[f64]| {
            if x.iter().all(|a| a.is_finite()) {
                Ok(())
            } else {
                Err(format!(
                    "the values of '{}' are not all finite (NaN, NA or Inf for some samples); R's model.frame() drops \
                     NA rows and DESeq2 then fails, so this design cannot be fitted",
                    label
                ))
            }
        };
        match v {
            Val::Num(x) => {
                check_len(x.len())?;
                check_finite(&x)?;
                Ok(FrameValue::Numeric(x))
            }
            Val::Mat { cols, names } => {
                for c in &cols {
                    check_len(c.len())?;
                    check_finite(c)?;
                }
                let names = names.unwrap_or_else(|| (1..=cols.len()).map(|i| i.to_string()).collect());
                Ok(FrameValue::Matrix { columns: cols, names })
            }
            Val::Lgl(x) => {
                check_len(x.len())?;
                // R: contrasts<- on a logical variable makes factor(x, levels = c(FALSE, TRUE))
                Ok(FrameValue::Factor {
                    codes: x.iter().map(|&b| b as usize).collect(),
                    levels: vec!["FALSE".to_string(), "TRUE".to_string()],
                    ordered: false,
                })
            }
            Val::Chr(s) => {
                check_len(s.len())?;
                let (codes, levels) = factor_of_strings(&s, None)?;
                self.factor_value(codes, levels, false)
            }
            Val::Fac { codes, levels, ordered } => {
                check_len(codes.len())?;
                self.factor_value(codes, levels, ordered)
            }
            Val::Null => Err("the variable is NULL".to_string()),
        }
    }

    fn factor_value(&self, codes: Vec<usize>, levels: Vec<String>, ordered: bool) -> std::result::Result<FrameValue, String> {
        if levels.len() < 2 {
            return Err("contrasts can be applied only to factors with 2 or more levels".to_string());
        }
        Ok(FrameValue::Factor { codes, levels, ordered })
    }
}

/// An R value during evaluation
#[derive(Debug, Clone)]
enum Val {
    Num(Vec<f64>),
    Lgl(Vec<bool>),
    Chr(Vec<String>),
    /// 0-based level codes
    Fac { codes: Vec<usize>, levels: Vec<String>, ordered: bool },
    /// Columns and optional column names
    Mat { cols: Vec<Vec<f64>>, names: Option<Vec<String>> },
    Null,
}

type EResult<T> = std::result::Result<T, String>;

impl Val {
    fn len(&self) -> usize {
        match self {
            Val::Num(x) => x.len(),
            Val::Lgl(x) => x.len(),
            Val::Chr(x) => x.len(),
            Val::Fac { codes, .. } => codes.len(),
            Val::Mat { cols, .. } => cols.iter().map(|c| c.len()).sum(),
            Val::Null => 0,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Val::Num(_) => "numeric",
            Val::Lgl(_) => "logical",
            Val::Chr(_) => "character",
            Val::Fac { .. } => "factor",
            Val::Mat { .. } => "matrix",
            Val::Null => "NULL",
        }
    }

    /// as.character()
    fn as_strings(&self) -> EResult<Vec<String>> {
        Ok(match self {
            Val::Num(x) => x.iter().map(|v| format_real15(*v)).collect(),
            Val::Lgl(x) => x.iter().map(|b| if *b { "TRUE" } else { "FALSE" }.to_string()).collect(),
            Val::Chr(x) => x.clone(),
            Val::Fac { codes, levels, .. } => codes.iter().map(|&c| levels[c].clone()).collect(),
            Val::Mat { cols, .. } => cols.iter().flatten().map(|v| format_real15(*v)).collect(),
            Val::Null => Vec::new(),
        })
    }

    /// The numeric vector of an argument of a mathematical function (logicals are 0/1)
    fn numeric(&self, what: &str) -> EResult<Vec<f64>> {
        match self {
            Val::Num(x) => Ok(x.clone()),
            Val::Lgl(x) => Ok(x.iter().map(|&b| b as u8 as f64).collect()),
            Val::Mat { cols, .. } if cols.len() == 1 => Ok(cols[0].clone()),
            Val::Fac { .. } => Err(format!("\u{2018}{}\u{2019} not meaningful for factors", what)),
            Val::Chr(_) => Err(format!("non-numeric argument to {}", what)),
            Val::Mat { .. } => Err(format!("{}: a matrix argument with several columns is not supported", what)),
            Val::Null => Err(format!("{}: NULL argument", what)),
        }
    }

    fn scalar(&self, what: &str) -> EResult<f64> {
        let x = self.numeric(what)?;
        if x.len() != 1 {
            return Err(format!("'{}' must be a single number", what));
        }
        Ok(x[0])
    }

    fn logical_scalar(&self, what: &str) -> EResult<bool> {
        match self {
            Val::Lgl(x) if x.len() == 1 => Ok(x[0]),
            Val::Num(x) if x.len() == 1 => Ok(x[0] != 0.0),
            _ => Err(format!("'{}' must be TRUE or FALSE", what)),
        }
    }
}

/// factor(x) of strings: levels sorted as R sorts strings, or the given levels
fn factor_of_strings(x: &[String], levels: Option<Vec<String>>) -> EResult<(Vec<usize>, Vec<String>)> {
    let levels = match levels {
        Some(l) => {
            let mut uniq: Vec<String> = Vec::new();
            for s in l {
                if !uniq.contains(&s) {
                    uniq.push(s);
                }
            }
            uniq
        }
        None => {
            let mut u = x.to_vec();
            u.sort_by(|a, b| r_collate(a, b));
            u.dedup();
            u
        }
    };
    let codes = x
        .iter()
        .map(|s| {
            levels.iter().position(|l| l == s).ok_or_else(|| {
                format!(
                    "the value '{}' is not one of the factor levels, so it would be NA; R's model.frame() drops NA \
                     samples and DESeq2 then fails",
                    s
                )
            })
        })
        .collect::<EResult<Vec<usize>>>()?;
    Ok((codes, levels))
}

/// R's match.arg-free argument matching of a closure call: exact names, then unique
/// partial names for formals before `...`, then positions. Returns the argument of each
/// formal and the arguments that go to `...`.
fn match_args<'a>(
    fun: &str,
    args: &'a [(Option<String>, Ex)],
    formals: &[&str],
) -> EResult<(Vec<Option<&'a Ex>>, Vec<&'a Ex>)> {
    let dots = formals.iter().position(|f| *f == "...");
    let mut matched: Vec<Option<&Ex>> = vec![None; formals.len()];
    let mut used = vec![false; args.len()];
    // exact names
    for (k, (name, e)) in args.iter().enumerate() {
        if let Some(name) = name {
            if let Some(i) = formals.iter().position(|f| f == name && *f != "...") {
                if matched[i].is_some() {
                    return Err(format!("formal argument \"{}\" matched by multiple actual arguments", name));
                }
                matched[i] = Some(e);
                used[k] = true;
            }
        }
    }
    // partial names (formals before ...)
    let partial_end = dots.unwrap_or(formals.len());
    for (k, (name, e)) in args.iter().enumerate() {
        if used[k] {
            continue;
        }
        if let Some(name) = name {
            let cands: Vec<usize> =
                (0..partial_end).filter(|&i| matched[i].is_none() && formals[i].starts_with(name.as_str())).collect();
            match cands.as_slice() {
                [i] => {
                    matched[*i] = Some(e);
                    used[k] = true;
                }
                [] => {}
                _ => return Err(format!("argument {} matches multiple formal arguments", k + 1)),
            }
        }
    }
    // positions
    let mut rest = Vec::new();
    let mut next = 0;
    for (k, (name, e)) in args.iter().enumerate() {
        if used[k] {
            continue;
        }
        if name.is_some() {
            if dots.is_some() {
                rest.push(e);
                continue;
            }
            return Err(format!("unused argument ({} = {})", name.as_ref().unwrap(), deparse(e)));
        }
        while next < partial_end && matched[next].is_some() {
            next += 1;
        }
        if next < partial_end {
            matched[next] = Some(e);
            next += 1;
        } else if dots.is_some() {
            rest.push(e);
        } else {
            return Err(format!("unused argument ({}) in {}()", deparse(e), fun));
        }
    }
    Ok((matched, rest))
}

struct Eval<'a> {
    frame: &'a ModelFrame,
}

/// The functions this module evaluates
const SUPPORTED: &str = "factor, as.factor, ordered, as.ordered, relevel, as.numeric, as.double, as.integer, \
                         as.character, log, log2, log10, log1p, exp, expm1, sqrt, abs, scale, I, c, poly, ns, bs, offset";

impl Eval<'_> {
    fn eval(&self, e: &Ex) -> EResult<Val> {
        match e {
            Ex::Sym(name) => match self.frame.columns.get(name) {
                Some(Column::Numeric(x)) => Ok(Val::Num(x.clone())),
                Some(Column::Logical(x)) => Ok(Val::Lgl(x.clone())),
                Some(Column::Factor { values, levels }) => Ok(Val::Fac {
                    codes: values.iter().map(|v| levels.iter().position(|l| l == v).unwrap()).collect(),
                    levels: levels.clone(),
                    ordered: false,
                }),
                None => Err(format!("object '{}' not found", name)),
            },
            Ex::Num(v, _) => Ok(Val::Num(vec![*v])),
            Ex::Str(s) => Ok(Val::Chr(vec![s.clone()])),
            Ex::Lgl(b) => Ok(Val::Lgl(vec![*b])),
            Ex::Null => Ok(Val::Null),
            Ex::Na => Err("NA values are not supported in the design formula".to_string()),
            Ex::Paren(a) => self.eval(a),
            Ex::Unary(op, a) => {
                let v = self.eval(a)?;
                match *op {
                    "!" => match v {
                        Val::Lgl(x) => Ok(Val::Lgl(x.iter().map(|b| !b).collect())),
                        Val::Num(x) => Ok(Val::Lgl(x.iter().map(|v| *v == 0.0).collect())),
                        _ => Err("invalid argument type to '!'".to_string()),
                    },
                    _ => {
                        let sign = if *op == "-" { -1.0 } else { 1.0 };
                        self.map_numeric(v, op, |x| if sign < 0.0 { -x } else { x })
                    }
                }
            }
            Ex::Binary(op, a, b) => self.binary(op, self.eval(a)?, self.eval(b)?),
            Ex::Call(f, args) => self.call(f, args),
        }
    }

    /// Applies an elementwise numeric function, keeping matrices
    fn map_numeric(&self, v: Val, what: &str, f: impl Fn(f64) -> f64) -> EResult<Val> {
        match v {
            Val::Mat { cols, names } => Ok(Val::Mat { cols: cols.iter().map(|c| c.iter().map(|&x| f(x)).collect()).collect(), names }),
            other => Ok(Val::Num(other.numeric(what)?.into_iter().map(f).collect())),
        }
    }

    fn binary(&self, op: &str, a: Val, b: Val) -> EResult<Val> {
        match op {
            "+" | "-" | "*" | "/" | "^" | "%%" | "%/%" => self.arithmetic(op, a, b),
            "==" | "!=" | "<" | ">" | "<=" | ">=" => self.compare(op, a, b),
            "&" | "&&" | "|" | "||" => {
                let x = logical_vector(&a, op)?;
                let y = logical_vector(&b, op)?;
                let and = op.starts_with('&');
                recycle(x.len(), y.len(), op).map(|n| {
                    Val::Lgl((0..n).map(|i| if and { x[i % x.len()] && y[i % y.len()] } else { x[i % x.len()] || y[i % y.len()] }).collect())
                })
            }
            "%in%" => {
                let x = a.as_strings()?;
                let table = b.as_strings()?;
                Ok(Val::Lgl(x.iter().map(|s| table.contains(s)).collect()))
            }
            other => Err(format!("the operator {} is not supported", other)),
        }
    }

    fn arithmetic(&self, op: &str, a: Val, b: Val) -> EResult<Val> {
        for v in [&a, &b] {
            match v {
                Val::Fac { .. } => return Err(format!("\u{2018}{}\u{2019} not meaningful for factors", op)),
                Val::Chr(_) | Val::Null => return Err("non-numeric argument to binary operator".to_string()),
                _ => {}
            }
        }
        let f = |x: f64, y: f64| -> f64 {
            match op {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "/" => x / y,
                "^" => r_pow(x, y),
                "%%" => r_mod(x, y),
                _ => r_int_div(x, y),
            }
        };
        let cols = |v: &Val| -> EResult<(Vec<Vec<f64>>, Option<Option<Vec<String>>>)> {
            match v {
                Val::Mat { cols, names } => Ok((cols.clone(), Some(names.clone()))),
                other => Ok((vec![other.numeric(op)?], None)),
            }
        };
        let (ca, na) = cols(&a)?;
        let (cb, nb) = cols(&b)?;
        let names = na.or(nb);
        if names.is_none() {
            let (x, y) = (&ca[0], &cb[0]);
            let n = recycle(x.len(), y.len(), op)?;
            return Ok(Val::Num((0..n).map(|i| f(x[i % x.len()], y[i % y.len()])).collect()));
        }
        // a matrix with a scalar, a vector of one column's length, or a matrix of the same shape
        let ncol = ca.len().max(cb.len());
        let nrow = ca.iter().chain(&cb).map(|c| c.len()).max().unwrap_or(0);
        let pick = |c: &Vec<Vec<f64>>, j: usize, i: usize| -> EResult<f64> {
            let col = if c.len() == 1 { &c[0] } else { c.get(j).ok_or("non-conformable arrays")? };
            match col.len() {
                1 => Ok(col[0]),
                l if l == nrow => Ok(col[i]),
                _ => Err("non-conformable arrays".to_string()),
            }
        };
        let mut out = Vec::with_capacity(ncol);
        for j in 0..ncol {
            let mut col = Vec::with_capacity(nrow);
            for i in 0..nrow {
                col.push(f(pick(&ca, j, i)?, pick(&cb, j, i)?));
            }
            out.push(col);
        }
        Ok(Val::Mat { cols: out, names: names.flatten() })
    }

    fn compare(&self, op: &str, a: Val, b: Val) -> EResult<Val> {
        let numeric = |v: &Val| matches!(v, Val::Num(_) | Val::Lgl(_)) || matches!(v, Val::Mat { cols, .. } if cols.len() == 1);
        if numeric(&a) && numeric(&b) {
            let x = a.numeric(op)?;
            let y = b.numeric(op)?;
            let n = recycle(x.len(), y.len(), op)?;
            return Ok(Val::Lgl(
                (0..n)
                    .map(|i| {
                        let (p, q) = (x[i % x.len()], y[i % y.len()]);
                        match op {
                            "==" => p == q,
                            "!=" => p != q,
                            "<" => p < q,
                            ">" => p > q,
                            "<=" => p <= q,
                            _ => p >= q,
                        }
                    })
                    .collect(),
            ));
        }
        if !matches!(op, "==" | "!=") {
            if matches!(a, Val::Fac { .. }) || matches!(b, Val::Fac { .. }) {
                return Err(format!("\u{2018}{}\u{2019} not meaningful for factors", op));
            }
            return Err(format!("the comparison {} of character values is not supported", op));
        }
        let x = a.as_strings()?;
        let y = b.as_strings()?;
        let n = recycle(x.len(), y.len(), op)?;
        Ok(Val::Lgl((0..n).map(|i| (x[i % x.len()] == y[i % y.len()]) == (op == "==")).collect()))
    }

    fn call(&self, f: &str, args: &[(Option<String>, Ex)]) -> EResult<Val> {
        let name = match f.split_once("::") {
            Some(("base" | "stats", g)) if !matches!(g, "ns" | "bs") => g,
            Some(("splines", g @ ("ns" | "bs"))) => g,
            Some(_) => {
                return Err(format!(
                    "the function {}() cannot be evaluated: rust_deseq2 evaluates only these R functions in a design \
                     formula: {}; compute the variable in the metadata instead",
                    f, SUPPORTED
                ));
            }
            None => f,
        };
        let get = |formals: &[&str]| match_args(name, args, formals);
        let arg = |e: Option<&Ex>| -> EResult<Option<Val>> { e.map(|e| self.eval(e)).transpose() };
        match name {
            "I" | "offset" => {
                let (m, _) = get(&["x"])?;
                self.eval(m[0].ok_or(format!("argument missing in {}()", name))?)
            }
            "c" => {
                let (_, dots) = get(&["..."])?;
                let vals: Vec<Val> = dots.iter().map(|e| self.eval(e)).collect::<EResult<_>>()?;
                if vals.iter().any(|v| matches!(v, Val::Chr(_))) {
                    let mut out = Vec::new();
                    for v in &vals {
                        out.extend(v.as_strings()?);
                    }
                    return Ok(Val::Chr(out));
                }
                if vals.iter().all(|v| matches!(v, Val::Lgl(_))) {
                    return Ok(Val::Lgl(vals.into_iter().flat_map(|v| if let Val::Lgl(x) = v { x } else { Vec::new() }).collect()));
                }
                let mut out = Vec::new();
                for v in &vals {
                    match v {
                        Val::Null => {}
                        Val::Fac { .. } => return Err("c() of factors is not supported".to_string()),
                        other => out.extend(other.numeric("c")?),
                    }
                }
                Ok(Val::Num(out))
            }
            "factor" | "ordered" => {
                let (m, _) = get(&["x", "levels", "labels", "exclude", "ordered", "nmax"])?;
                if m[3].is_some() || m[5].is_some() {
                    return Err(format!("the arguments 'exclude' and 'nmax' of {}() are not supported", name));
                }
                let x = arg(m[0])?.ok_or("argument 'x' missing in factor()")?;
                let ordered = match arg(m[4])? {
                    Some(v) => v.logical_scalar("ordered")?,
                    None => name == "ordered" || matches!(x, Val::Fac { ordered: true, .. }),
                };
                make_factor(x, arg(m[1])?, arg(m[2])?, ordered)
            }
            "as.factor" | "as.ordered" => {
                let (m, _) = get(&["x"])?;
                let x = arg(m[0])?.ok_or(format!("argument 'x' missing in {}()", name))?;
                let want_ordered = name == "as.ordered";
                match x {
                    // as.factor() returns a factor unchanged; as.ordered() keeps its levels
                    Val::Fac { codes, levels, ordered } => Ok(Val::Fac { codes, levels, ordered: ordered || want_ordered }),
                    other => make_factor(other, None, None, want_ordered),
                }
            }
            "relevel" => {
                let (m, _) = get(&["x", "ref", "..."])?;
                let x = arg(m[0])?.ok_or("argument 'x' missing in relevel()")?;
                let r = arg(m[1])?.ok_or("argument \"ref\" is missing, with no default")?;
                let Val::Fac { codes, mut levels, ordered: false } = x else {
                    return Err(if matches!(x, Val::Fac { .. }) {
                        "'relevel' only for unordered factors".to_string()
                    } else {
                        "'relevel' only for (unordered) factors".to_string()
                    });
                };
                if r.len() != 1 {
                    return Err("'ref' must be of length one".to_string());
                }
                let idx = match &r {
                    Val::Chr(s) => levels.iter().position(|l| *l == s[0]).ok_or("'ref' must be an existing level")?,
                    Val::Num(v) => {
                        let k = v[0];
                        if !(k >= 1.0 && k <= levels.len() as f64) {
                            return Err(format!("ref = {} must be in 1L:{}", format_real15(k), levels.len()));
                        }
                        k as usize - 1
                    }
                    _ => return Err("'ref' must be an existing level".to_string()),
                };
                let lv = levels.remove(idx);
                levels.insert(0, lv);
                // codes: the reference becomes 0, the levels before it move up by one
                let codes = codes.into_iter().map(|c| if c == idx { 0 } else if c < idx { c + 1 } else { c }).collect();
                Ok(Val::Fac { codes, levels, ordered: false })
            }
            "as.numeric" | "as.double" | "as.integer" => {
                let (m, _) = get(&["x", "..."])?;
                let x = arg(m[0])?.unwrap_or(Val::Num(Vec::new()));
                let v: Vec<f64> = match x {
                    Val::Fac { codes, .. } => codes.iter().map(|&c| (c + 1) as f64).collect(),
                    Val::Chr(s) => s
                        .iter()
                        .map(|t| {
                            crate::io::tximport::r_strtod(t)
                                .ok_or_else(|| format!("{}(\"{}\") is NA (NAs introduced by coercion)", name, t))
                        })
                        .collect::<EResult<_>>()?,
                    Val::Mat { cols, .. } => cols.into_iter().flatten().collect(),
                    other => other.numeric(name)?,
                };
                if name == "as.integer" {
                    if v.iter().any(|x| !x.is_finite() || x.abs() >= 2147483648.0) {
                        return Err("as.integer(): NAs introduced by coercion to integer range".to_string());
                    }
                    return Ok(Val::Num(v.into_iter().map(f64::trunc).collect()));
                }
                Ok(Val::Num(v))
            }
            "as.character" => {
                let (m, _) = get(&["x", "..."])?;
                Ok(Val::Chr(arg(m[0])?.unwrap_or(Val::Chr(Vec::new())).as_strings()?))
            }
            "log" => {
                let (m, _) = get(&["x", "base"])?;
                let x = arg(m[0])?.ok_or("argument \"x\" is missing, with no default")?;
                match arg(m[1])? {
                    None => self.map_numeric(x, "log", r_log),
                    Some(b) => {
                        let base = b.scalar("base")?;
                        self.map_numeric(x, "log", |v| {
                            if base == 10.0 {
                                v.log10()
                            } else if base == 2.0 {
                                v.log2()
                            } else {
                                r_log(v) / r_log(base)
                            }
                        })
                    }
                }
            }
            "log2" | "log10" | "log1p" | "exp" | "expm1" | "sqrt" | "abs" => {
                let (m, _) = get(&["x"])?;
                let x = arg(m[0])?.ok_or(format!("0 arguments passed to '{}' which requires 1", name))?;
                let f: fn(f64) -> f64 = match name {
                    "log2" => f64::log2,
                    "log10" => f64::log10,
                    "log1p" => f64::ln_1p,
                    "exp" => f64::exp,
                    "expm1" => f64::exp_m1,
                    "sqrt" => f64::sqrt,
                    _ => f64::abs,
                };
                self.map_numeric(x, name, f)
            }
            "scale" => {
                let (m, _) = get(&["x", "center", "scale"])?;
                let x = arg(m[0])?.ok_or("argument \"x\" is missing, with no default")?;
                let cols: Vec<Vec<f64>> = match x {
                    Val::Mat { cols, .. } => cols,
                    other => vec![other.numeric("scale")?],
                };
                let spec = |v: Option<Val>, what: &str| -> EResult<Vec<Option<Option<f64>>>> {
                    match v {
                        None => Ok(vec![Some(None); cols.len()]),
                        Some(Val::Lgl(b)) if b.len() == 1 => Ok(vec![if b[0] { Some(None) } else { None }; cols.len()]),
                        Some(other) => {
                            let x = other.numeric(what)?;
                            if x.len() != cols.len() {
                                return Err(format!("length of '{}' must equal the number of columns of 'x'", what));
                            }
                            Ok(x.into_iter().map(|c| Some(Some(c))).collect())
                        }
                    }
                };
                let centers = spec(arg(m[1])?, "center")?;
                let scales = spec(arg(m[2])?, "scale")?;
                let out = cols.iter().enumerate().map(|(j, c)| rfuns::scale_column(c, centers[j], scales[j])).collect();
                Ok(Val::Mat { cols: out, names: None })
            }
            "poly" => {
                let (m, dots) = get(&["x", "...", "degree", "coefs", "raw", "simple"])?;
                let x = arg(m[0])?.ok_or("argument \"x\" is missing, with no default")?;
                let mut degree = match arg(m[2])? {
                    Some(d) => d.scalar("degree")?,
                    None => 1.0,
                };
                match dots.as_slice() {
                    [] => {}
                    [d] => {
                        let v = self.eval(d)?;
                        if v.len() != 1 {
                            return Err("poly() of several variables (polym) is not supported".to_string());
                        }
                        degree = v.scalar("degree")?;
                    }
                    _ => return Err("poly() of several variables (polym) is not supported".to_string()),
                }
                if m[3].is_some_and(|e| *e != Ex::Null) {
                    return Err("poly(coefs = ) is not supported".to_string());
                }
                let raw = match arg(m[4])? {
                    Some(v) => v.logical_scalar("raw")?,
                    None => false,
                };
                // R: if (is.object(x) && mode(x) == "numeric") x <- as.numeric(x)
                let xv: Vec<f64> = match x {
                    Val::Fac { codes, .. } => codes.iter().map(|&c| (c + 1) as f64).collect(),
                    other => other.numeric("poly")?,
                };
                if degree < 1.0 {
                    return Err("'degree' must be at least 1".to_string());
                }
                let d = degree as usize;
                let basis = if raw { rfuns::poly_raw(&xv, d)? } else { rfuns::poly_orthogonal(&xv, d)? };
                Ok(Val::Mat {
                    cols: (0..basis.ncols()).map(|j| basis.column(j).to_vec()).collect(),
                    names: Some((1..=d).map(|k| k.to_string()).collect()),
                })
            }
            "ns" | "bs" => {
                let formals: &[&str] = if name == "ns" {
                    &["x", "df", "knots", "intercept", "Boundary.knots"]
                } else {
                    &["x", "df", "knots", "degree", "intercept", "Boundary.knots", "warn.outside"]
                };
                let (m, _) = get(formals)?;
                let pos = |n: &str| formals.iter().position(|f| *f == n).unwrap();
                let x = arg(m[0])?.ok_or("argument \"x\" is missing, with no default")?;
                let xv = match x {
                    Val::Num(v) => v,
                    Val::Mat { cols, .. } if cols.len() == 1 => cols[0].clone(),
                    other => return Err(format!("{}() needs a numeric variable, not a {}", name, other.type_name())),
                };
                let num_or_null = |e: Option<&Ex>, what: &str| -> EResult<Option<Vec<f64>>> {
                    match arg(e)? {
                        None | Some(Val::Null) => Ok(None),
                        Some(v) => Ok(Some(v.numeric(what)?)),
                    }
                };
                let df = match num_or_null(m[pos("df")], "df")? {
                    Some(v) if v.len() == 1 => Some(v[0]),
                    Some(_) => return Err("'df' must be a single number".to_string()),
                    None => None,
                };
                let boundary = match num_or_null(m[pos("Boundary.knots")], "Boundary.knots")? {
                    Some(v) if v.len() == 2 => Some([v[0], v[1]]),
                    Some(_) => return Err("'Boundary.knots' must have two values".to_string()),
                    None => None,
                };
                let intercept = match arg(m[pos("intercept")])? {
                    Some(v) => v.logical_scalar("intercept")?,
                    None => false,
                };
                let sargs = SplineArgs { df, knots: num_or_null(m[pos("knots")], "knots")?, intercept, boundary_knots: boundary };
                let basis = if name == "ns" {
                    rfuns::ns(&xv, &sargs)?
                } else {
                    let degree = match arg(m[pos("degree")])? {
                        Some(v) => v.scalar("degree")?,
                        None => 3.0,
                    };
                    rfuns::bs(&xv, &sargs, degree)?
                };
                Ok(Val::Mat {
                    cols: (0..basis.ncols()).map(|j| basis.column(j).to_vec()).collect(),
                    names: Some((1..=basis.ncols()).map(|k| k.to_string()).collect()),
                })
            }
            _ => Err(format!(
                "the function {}() cannot be evaluated: rust_deseq2 evaluates only these R functions in a design \
                 formula (it cannot run R code, such as user-defined functions): {}; compute the variable in the \
                 metadata instead",
                f, SUPPORTED
            )),
        }
    }
}

/// factor(x, levels, labels, ordered)
/// R equivalent: base::factor()
fn make_factor(x: Val, levels: Option<Val>, labels: Option<Val>, ordered: bool) -> EResult<Val> {
    let strings = x.as_strings()?;
    let level_strings: Vec<String> = match (&levels, &x) {
        (Some(l), _) => l.as_strings()?,
        // levels of a factor: those present, in level order
        (None, Val::Fac { codes, levels, .. }) => {
            (0..levels.len()).filter(|l| codes.contains(l)).map(|l| levels[l].clone()).collect()
        }
        (None, Val::Num(v)) => {
            let mut u = v.clone();
            u.sort_by(|a, b| a.partial_cmp(b).unwrap());
            u.dedup();
            let mut s: Vec<String> = Vec::new();
            for v in u {
                let t = format_real15(v);
                if !s.contains(&t) {
                    s.push(t);
                }
            }
            s
        }
        (None, Val::Lgl(v)) => ["FALSE", "TRUE"]
            .iter()
            .filter(|l| v.iter().any(|b| (if *b { "TRUE" } else { "FALSE" }) == **l))
            .map(|l| l.to_string())
            .collect(),
        (None, _) => {
            let mut u = strings.clone();
            u.sort_by(|a, b| r_collate(a, b));
            u.dedup();
            u
        }
    };
    let (mut codes, mut lv) = factor_of_strings(&strings, Some(level_strings))?;
    if let Some(lab) = labels {
        let lab = lab.as_strings()?;
        let new: Vec<String> = if lab.len() == lv.len() {
            lab
        } else if lab.len() == 1 {
            (1..=lv.len()).map(|i| format!("{}{}", lab[0], i)).collect()
        } else {
            return Err(format!("invalid 'labels'; length {} should be 1 or {}", lab.len(), lv.len()));
        };
        // duplicated labels merge their levels
        let mut merged: Vec<String> = Vec::new();
        let map: Vec<usize> = new
            .iter()
            .map(|l| match merged.iter().position(|m| m == l) {
                Some(i) => i,
                None => {
                    merged.push(l.clone());
                    merged.len() - 1
                }
            })
            .collect();
        codes = codes.into_iter().map(|c| map[c]).collect();
        lv = merged;
    }
    Ok(Val::Fac { codes, levels: lv, ordered })
}

fn logical_vector(v: &Val, op: &str) -> EResult<Vec<bool>> {
    match v {
        Val::Lgl(x) => Ok(x.clone()),
        Val::Num(x) => Ok(x.iter().map(|a| *a != 0.0).collect()),
        _ => Err(format!("operations are possible only for numeric, logical or complex types ({})", op)),
    }
}

/// The length of a vectorised operation (R recycles; lengths that are not multiples
/// are an error here)
fn recycle(a: usize, b: usize, op: &str) -> EResult<usize> {
    if a == 0 || b == 0 {
        return Ok(0);
    }
    let n = a.max(b);
    if n % a != 0 || n % b != 0 {
        return Err(format!("longer object length is not a multiple of shorter object length in '{}'", op));
    }
    Ok(n)
}

/// R's log(): -Inf at 0, NaN below
fn r_log(x: f64) -> f64 {
    x.ln()
}

/// R's `%%` (arithmetic.c myfmod), with the long double steps in double precision
fn r_mod(x1: f64, x2: f64) -> f64 {
    if x2 == 0.0 {
        return f64::NAN;
    }
    if x2.abs() * f64::EPSILON > 1.0 && x1.is_finite() && x1.abs() <= x2.abs() {
        return if x1.abs() == x2.abs() {
            0.0
        } else if (x1 < 0.0 && x2 > 0.0) || (x2 < 0.0 && x1 > 0.0) {
            x1 + x2
        } else {
            x1
        };
    }
    let q = x1 / x2;
    let tmp = (-q.floor()).mul_add(x2, x1);
    tmp - (tmp / x2).floor() * x2
}

/// R's `%/%` (arithmetic.c myfloor)
fn r_int_div(x1: f64, x2: f64) -> f64 {
    let q = x1 / x2;
    if x2 == 0.0 || q.abs() * f64::EPSILON > 1.0 || !q.is_finite() {
        return q;
    }
    if q.abs() < 1.0 {
        return if q < 0.0 {
            -1.0
        } else if (x1 < 0.0 && x2 > 0.0) || (x1 > 0.0 && x2 < 0.0) {
            -1.0
        } else {
            0.0
        };
    }
    let tmp = (-q.floor()).mul_add(x2, x1);
    q.floor() + (tmp / x2).floor()
}
