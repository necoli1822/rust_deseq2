//! Port of R's `optim(method = "L-BFGS-B")`.
//!
//! * [`optim_lbfgsb`] reproduces the `.External2(C_optim, ...)` path of
//!   R/optim.R and src/library/stats/src/optim.c for `method = "L-BFGS-B"`:
//!   `par / parscale` scaling, `fnscale` (negative to maximise), the
//!   finite-difference gradient `fmingr()` with its handling of the bounds,
//!   the `nbd` bound codes for infinite bounds, and the driver `lbfgsb()`
//!   (src/appl/optim.c) with its iteration count, convergence codes and
//!   message.
//! * [`lbfgsb`] is the port of src/appl/lbfgsb.c and [`blas`] of the
//!   BLAS/LINPACK routines it calls.
//! * [`nmath`] ports the R density functions (`dnbinom`, `dnorm`) and `sum()`
//!   that DESeq2's objective function evaluates.

mod blas;
pub(crate) mod lbfgsb;
pub mod nmath;

use std::fmt;

/// Control parameters of `optim(method = "L-BFGS-B")`, with R's defaults
/// (R/optim.R: `con <- list(trace = 0, fnscale = 1, parscale = rep.int(1,
/// npar), ndeps = rep.int(1e-3, npar), maxit = 100L, REPORT = 10, lmm = 5,
/// factr = 1e7, pgtol = 0)`).
#[derive(Debug, Clone)]
pub struct LbfgsbControl {
    /// Overall scaling of `fn`; a negative value maximises. R: `fnscale = 1`.
    pub fnscale: f64,
    /// Per-parameter scaling; `None` means `rep(1, npar)`. R: `parscale`.
    pub parscale: Option<Vec<f64>>,
    /// Finite-difference steps (on the `par/parscale` scale); `None` means
    /// `rep(1e-3, npar)`. R: `ndeps`.
    pub ndeps: Option<Vec<f64>>,
    /// Maximum number of iterations. R: `maxit = 100`.
    pub maxit: i32,
    /// Relative reduction tolerance in units of machine epsilon. R: `factr = 1e7`.
    pub factr: f64,
    /// Projected-gradient tolerance. R: `pgtol = 0`.
    pub pgtol: f64,
    /// Number of BFGS updates kept. R: `lmm = 5`.
    pub lmm: usize,
    /// Tracing level. Accepted for completeness; nothing is printed. R: `trace = 0`.
    pub trace: i32,
    /// Reporting interval for tracing; must be > 0 as in R. R: `REPORT = 10`.
    pub report: i32,
}

impl Default for LbfgsbControl {
    fn default() -> Self {
        LbfgsbControl {
            fnscale: 1.0,
            parscale: None,
            ndeps: None,
            maxit: 100,
            factr: 1e7,
            pgtol: 0.0,
            lmm: 5,
            trace: 0,
            report: 10,
        }
    }
}

/// Counts of `fn` and `gr` evaluations, as `optim()$counts`. For L-BFGS-B
/// R reports the number of (f, g) evaluations requested by the algorithm for
/// both; the extra `fn` calls of the finite-difference gradient are not
/// counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimCounts {
    pub function: i32,
    pub gradient: i32,
}

/// The value of `optim()` for `method = "L-BFGS-B"`.
#[derive(Debug, Clone)]
pub struct OptimResult {
    /// Best parameters found (`$par`).
    pub par: Vec<f64>,
    /// `fn(par)` (`$value`, on the original scale, i.e. times `fnscale`).
    pub value: f64,
    /// `$counts`.
    pub counts: OptimCounts,
    /// `$convergence`: 0 = converged, 1 = `maxit` reached, 51 = warning,
    /// 52 = error (e.g. "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH").
    pub convergence: i32,
    /// `$message`, the final L-BFGS-B task string.
    pub message: String,
}

/// Conditions for which R's `optim()` signals an error (`stop()`) instead
/// of returning a result.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimError {
    /// "L-BFGS-B needs finite values of 'fn'"
    NonFiniteFn,
    /// "non-finite value supplied by optim"
    NonFinitePar,
    /// "non-finite finite-difference value \[i\]" (1-based index)
    NonFiniteFiniteDifference(usize),
    /// Length mismatch of `parscale`, `ndeps`, `lower` or `upper`.
    WrongLength(&'static str),
    /// "REPORT must be > 0 (method = \"L-BFGS-B\")"
    BadReport,
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimError::NonFiniteFn => write!(f, "L-BFGS-B needs finite values of 'fn'"),
            OptimError::NonFinitePar => write!(f, "non-finite value supplied by optim"),
            OptimError::NonFiniteFiniteDifference(i) => write!(f, "non-finite finite-difference value [{i}]"),
            OptimError::WrongLength(what) => write!(f, "'{what}' is of the wrong length"),
            OptimError::BadReport => write!(f, "REPORT must be > 0 (method = \"L-BFGS-B\")"),
        }
    }
}

impl std::error::Error for OptimError {}

/// Analytic gradient callback: `gr(par, grad_out)`.
pub type GradFn<'a> = &'a dyn Fn(&[f64], &mut [f64]);

/// The objective/gradient wrapper of optim.c (`OptStruct`, `fminfn`, `fmingr`).
struct OptStruct<'a, F: Fn(&[f64]) -> f64> {
    fcall: F,
    gcall: Option<GradFn<'a>>,
    ndeps: Vec<f64>,
    fnscale: f64,
    parscale: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    x: Vec<f64>,
}

// The loops keep optim.c's indexing.
#[allow(clippy::needless_range_loop)]
impl<F: Fn(&[f64]) -> f64> OptStruct<'_, F> {
    /// R: `fminfn()`.
    fn fminfn(&mut self, p: &[f64]) -> Result<f64, OptimError> {
        for i in 0..p.len() {
            if !p[i].is_finite() {
                return Err(OptimError::NonFinitePar);
            }
            self.x[i] = p[i] * self.parscale[i];
        }
        Ok((self.fcall)(&self.x) / self.fnscale)
    }

    /// R: `fmingr()` with `usebounds = 1` (always the case for L-BFGS-B).
    fn fmingr(&mut self, p: &[f64], df: &mut [f64]) -> Result<(), OptimError> {
        let n = p.len();
        if let Some(gr) = self.gcall {
            // analytical derivatives
            for i in 0..n {
                if !p[i].is_finite() {
                    return Err(OptimError::NonFinitePar);
                }
                self.x[i] = p[i] * self.parscale[i];
            }
            gr(&self.x, df);
            for i in 0..n {
                df[i] = df[i] * self.parscale[i] / self.fnscale;
            }
        } else {
            // numerical derivatives
            for i in 0..n {
                self.x[i] = p[i] * self.parscale[i];
            }
            for i in 0..n {
                let mut eps = self.ndeps[i];
                let mut epsused = eps;
                let mut tmp = p[i] + eps;
                if tmp > self.upper[i] {
                    tmp = self.upper[i];
                    epsused = tmp - p[i];
                }
                self.x[i] = tmp * self.parscale[i];
                let val1 = (self.fcall)(&self.x) / self.fnscale;
                tmp = p[i] - eps;
                if tmp < self.lower[i] {
                    tmp = self.lower[i];
                    eps = p[i] - tmp;
                }
                self.x[i] = tmp * self.parscale[i];
                let val2 = (self.fcall)(&self.x) / self.fnscale;
                df[i] = (val1 - val2) / (epsused + eps);
                if !df[i].is_finite() {
                    return Err(OptimError::NonFiniteFiniteDifference(i + 1));
                }
                self.x[i] = p[i] * self.parscale[i];
            }
        }
        Ok(())
    }
}

/// R: `optim(par, fn, gr, method = "L-BFGS-B", lower, upper, control)`.
///
/// `lower`/`upper` are recycled to `length(par)` as `rep_len()` does (pass a
/// single value for a common bound; use `f64::NEG_INFINITY`/`f64::INFINITY`
/// for no bound). Without `gr` the gradient is approximated by the
/// finite differences of optim.c's `fmingr()`, which are one-sided next to an
/// active bound. Returns `Err` where R stops with an error.
pub fn optim_lbfgsb<F: Fn(&[f64]) -> f64>(
    par: &[f64],
    fn_: F,
    gr: Option<GradFn<'_>>,
    lower: &[f64],
    upper: &[f64],
    control: &LbfgsbControl,
) -> Result<OptimResult, OptimError> {
    let npar = par.len();
    let parscale = control.parscale.clone().unwrap_or_else(|| vec![1.0; npar]);
    if parscale.len() != npar {
        return Err(OptimError::WrongLength("parscale"));
    }
    let ndeps = control.ndeps.clone().unwrap_or_else(|| vec![1e-3; npar]);
    if gr.is_none() && ndeps.len() != npar {
        return Err(OptimError::WrongLength("ndeps"));
    }
    if lower.is_empty() && npar > 0 {
        return Err(OptimError::WrongLength("lower"));
    }
    if upper.is_empty() && npar > 0 {
        return Err(OptimError::WrongLength("upper"));
    }
    // R/optim.R: lower <- as.double(rep_len(lower, npar)) (same for upper)
    let slower: Vec<f64> = (0..npar).map(|i| lower[i % lower.len()]).collect();
    let supper: Vec<f64> = (0..npar).map(|i| upper[i % upper.len()]).collect();

    // optim.c: dpar[i] = REAL(par)[i] / (OS->parscale[i])
    let mut dpar: Vec<f64> = (0..npar).map(|i| par[i] / parscale[i]).collect();
    let mut l = vec![0.0; npar];
    let mut u = vec![0.0; npar];
    let mut nbd = vec![0i32; npar];
    for i in 0..npar {
        l[i] = slower[i] / parscale[i];
        u[i] = supper[i] / parscale[i];
        nbd[i] = if !l[i].is_finite() {
            if !u[i].is_finite() {
                0
            } else {
                3
            }
        } else if !u[i].is_finite() {
            1
        } else {
            2
        };
    }
    let mut os = OptStruct {
        fcall: fn_,
        gcall: gr,
        ndeps,
        fnscale: control.fnscale,
        parscale: parscale.clone(),
        lower: l.clone(),
        upper: u.clone(),
        x: vec![0.0; npar],
    };

    let (val, fail, count, msg) = lbfgsb_driver(
        npar,
        control.lmm,
        &mut dpar,
        &l,
        &u,
        &nbd,
        &mut os,
        control.factr,
        control.pgtol,
        control.maxit,
        control.report,
    )?;

    Ok(OptimResult {
        par: (0..npar).map(|i| dpar[i] * parscale[i]).collect(),
        value: val * control.fnscale,
        counts: OptimCounts { function: count, gradient: count },
        convergence: fail,
        message: msg,
    })
}

/// R: `lbfgsb()` in src/appl/optim.c -- the driver loop around `setulb()`.
/// Returns `(Fmin, fail, fncount = grcount, msg)`.
#[allow(clippy::too_many_arguments)]
fn lbfgsb_driver<F: Fn(&[f64]) -> f64>(
    n: usize,
    m: usize,
    x: &mut [f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    os: &mut OptStruct<'_, F>,
    factr: f64,
    pgtol: f64,
    maxit: i32,
    n_report: i32,
) -> Result<(f64, i32, i32, String), OptimError> {
    if n == 0 {
        // not handled in setulb
        let f = os.fminfn(u)?;
        return Ok((f, 0, 1, "NOTHING TO DO".to_string()));
    }
    if n_report <= 0 {
        return Err(OptimError::BadReport);
    }
    let mut fail = 0;
    let mut f = 0.0f64;
    let mut g = vec![0.0; n];
    let mut state = lbfgsb::Lbfgsb::new(n, m);
    let mut task = "START".to_string();
    let mut iter = 0;
    loop {
        state.setulb(x, l, u, nbd, &mut f, &mut g, factr, pgtol, &mut task);
        if task.starts_with("FG") {
            f = os.fminfn(x)?;
            if !f.is_finite() {
                return Err(OptimError::NonFiniteFn);
            }
            os.fmingr(x, &mut g)?;
        } else if task.starts_with("NEW_X") {
            iter += 1;
            if iter > maxit {
                fail = 1;
                break;
            }
        } else if task.starts_with("WARN") {
            fail = 51;
            break;
        } else if task.starts_with("CONV") {
            break;
        } else if task.starts_with("ERROR") {
            fail = 52;
            break;
        } else {
            // some other condition that is not supposed to happen
            fail = 52;
            break;
        }
    }
    Ok((f, fail, state.nfgv, task))
}

#[cfg(test)]
mod tests;
