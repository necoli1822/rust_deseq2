//! Agreement of [`optim_lbfgsb`] with R 4.5.3's `optim(method = "L-BFGS-B")`.
//!
//! Reference values were produced by running the same problems through
//! `optim(..., method = "L-BFGS-B")` in R 4.5.3 (OpenBLAS 0.3.34) and printing
//! `par`, `value`, `counts`, `convergence` and `message` with `%.17g`. The
//! objective functions below are written so that they evaluate bit for bit
//! like their R counterparts.

// Reference values are R's %.17g output, kept verbatim.
#![allow(clippy::excessive_precision)]

use super::nmath::{dnbinom_mu_log, dnorm_log, r_sum};
use super::*;

/// Relative tolerance for agreement with R. The port reproduces R's
/// iterates exactly where the arithmetic is reproducible (see the module
/// docs of `blas`), so this only absorbs ulp-level effects.
const TOL: f64 = 1e-12;

struct Expected<'a> {
    par: &'a [f64],
    value: f64,
    counts: i32,
    convergence: i32,
    message: &'static str,
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b {
        0.0
    } else {
        (a - b).abs() / b.abs().max(1e-300)
    }
}

fn check(name: &str, o: &OptimResult, e: &Expected<'_>, tol: f64) {
    assert_eq!(o.par.len(), e.par.len(), "{name}: npar");
    let max_rel = o
        .par
        .iter()
        .zip(e.par)
        .map(|(&a, &b)| rel(a, b))
        .fold(rel(o.value, e.value), f64::max);
    eprintln!(
        "{name}: max rel diff vs R = {max_rel:e}, counts {} (R {}), conv {} (R {})",
        o.counts.function, e.counts, o.convergence, e.convergence
    );
    assert!(max_rel <= tol, "{name}: par {:?} value {:e}; R par {:?} value {:e}", o.par, o.value, e.par, e.value);
    assert_eq!(o.counts.function, e.counts, "{name}: counts");
    assert_eq!(o.counts.gradient, e.counts, "{name}: counts");
    assert_eq!(o.convergence, e.convergence, "{name}: convergence");
    assert_eq!(o.message, e.message, "{name}: message");
}

fn fr(x: &[f64]) -> f64 {
    100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
}

fn grr(x: &[f64], g: &mut [f64]) {
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    g[1] = 200.0 * (x[1] - x[0] * x[0]);
}

/// R's `^2` is `x*x` (arithmetic.c R_POW special case), as is `powi(2)`.
fn flb(x: &[f64]) -> f64 {
    let p = x.len();
    r_sum((0..p).map(|i| {
        let c = if i == 0 { 1.0 } else { 4.0 };
        let prev = if i == 0 { 1.0 } else { x[i - 1] };
        let t = x[i] - prev * prev;
        c * (t * t)
    }))
}

const CONV_F: &str = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH";

#[test]
fn rosenbrock_numerical_gradient() {
    let o = optim_lbfgsb(&[-1.2, 1.0], fr, None, &[f64::NEG_INFINITY], &[f64::INFINITY], &LbfgsbControl::default())
        .unwrap();
    let e = Expected {
        par: &[0.99980003783386839, 0.99960011579799257],
        value: 3.9984867886157939e-08,
        counts: 49,
        convergence: 0,
        message: CONV_F,
    };
    check("rosenbrock_numgrad", &o, &e, TOL);
}

#[test]
fn rosenbrock_analytic_gradient() {
    let o = optim_lbfgsb(
        &[-1.2, 1.0],
        fr,
        Some(&grr),
        &[f64::NEG_INFINITY],
        &[f64::INFINITY],
        &LbfgsbControl::default(),
    )
    .unwrap();
    let e = Expected {
        par: &[0.99999971674923405, 0.99999947177993476],
        value: 2.2677745083912764e-13,
        counts: 47,
        convergence: 0,
        message: CONV_F,
    };
    check("rosenbrock_gradient", &o, &e, TOL);
}

#[test]
fn active_bounds() {
    let o = optim_lbfgsb(&[3.0; 8], flb, None, &[2.0], &[4.0], &LbfgsbControl::default()).unwrap();
    let e = Expected {
        par: &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1090931369710688, 4.0],
        value: 96.105912874337548,
        counts: 7,
        convergence: 0,
        message: CONV_F,
    };
    check("flb8_bounds", &o, &e, TOL);
}

#[test]
fn active_bounds_25_parameters() {
    // 25 parameters: OpenBLAS's vectorised ddot kernel (n >= 16) is not
    // reproduced bit for bit, so allow a looser tolerance here.
    let o = optim_lbfgsb(&[3.0; 25], flb, None, &[2.0], &[4.0], &LbfgsbControl::default()).unwrap();
    let mut par = [2.0; 25];
    par[23] = 2.1090930615370751;
    par[24] = 4.0;
    let e = Expected { par: &par, value: 368.10591287434056, counts: 6, convergence: 0, message: CONV_F };
    check("flb25_bounds", &o, &e, 1e-9);
}

#[test]
fn one_sided_bounds() {
    let o = optim_lbfgsb(
        &[-1.2, 1.0],
        fr,
        Some(&grr),
        &[-0.5, f64::NEG_INFINITY],
        &[f64::INFINITY, 0.2],
        &LbfgsbControl::default(),
    )
    .unwrap();
    let e = Expected {
        par: &[0.45388973477663491, 0.20000000000000001],
        value: 0.30185551663851373,
        counts: 8,
        convergence: 0,
        message: CONV_F,
    };
    check("rosenbrock_onesided", &o, &e, TOL);
}

#[test]
fn maximise_with_fnscale_and_parscale() {
    let fmax = |x: &[f64]| -(x[0] - 1.0).powi(2) - 2.0 * (x[1] + 0.5).powi(2) + 0.3 * x[0] * x[1] + x[0].sin();
    let control = LbfgsbControl { fnscale: -1.0, parscale: Some(vec![1.0, 10.0]), ..Default::default() };
    let o = optim_lbfgsb(&[0.3, 0.2], fmax, None, &[f64::NEG_INFINITY], &[f64::INFINITY], &control).unwrap();
    let e = Expected {
        par: &[1.1445874966181084, -0.41415593775359999],
        value: 0.73268437900995576,
        counts: 12,
        convergence: 0,
        message: CONV_F,
    };
    check("maximise_fnscale", &o, &e, TOL);
}

#[test]
fn maxit_reached() {
    let control = LbfgsbControl { maxit: 5, ..Default::default() };
    let o = optim_lbfgsb(&[-1.2, 1.0], fr, None, &[f64::NEG_INFINITY], &[f64::INFINITY], &control).unwrap();
    let e = Expected {
        par: &[-1.0020272469819695, 0.98860243338961062],
        value: 4.0320024177068516,
        counts: 8,
        convergence: 1,
        message: "NEW_X",
    };
    check("rosenbrock_maxit5", &o, &e, TOL);
}

#[test]
fn non_finite_fn_is_an_error() {
    let f = |x: &[f64]| if x[0] > 0.5 { f64::NAN } else { (x[0] - 1.0).powi(2) };
    let r = optim_lbfgsb(&[0.0], f, None, &[f64::NEG_INFINITY], &[f64::INFINITY], &LbfgsbControl::default());
    assert!(matches!(r, Err(OptimError::NonFiniteFn) | Err(OptimError::NonFiniteFiniteDifference(_))));
}

/// DESeq2's `objectiveFn` in `fitNbinomGLMsOptim` for design ~condition (3 vs 3).
fn deseq2_objective(k: &'static [f64], nf: &'static [f64], alpha: f64) -> impl Fn(&[f64]) -> f64 {
    move |p: &[f64]| {
        let size = 1.0 / alpha;
        let ll = r_sum((0..6).map(|i| {
            let x1 = if i >= 3 { 1.0 } else { 0.0 };
            let eta = x1 * p[1] + p[0];
            let mu = nf[i] * 2f64.powf(eta);
            dnbinom_mu_log(k[i], size, mu)
        }));
        let sd = (1.0 / 1e-6f64).sqrt();
        let lp = r_sum(p.iter().map(|&b| dnorm_log(b, 0.0, sd)));
        // R: -1 * (logLike + logPrior)
        let v = -(ll + lp);
        if v.is_finite() {
            v
        } else {
            1e300
        }
    }
}

const NF: [f64; 6] = [0.95, 1.02, 1.1, 0.9, 1.05, 0.98];

#[test]
fn deseq2_degenerate_gene() {
    // counts 0 0 0 0 0 535, alpha 10: IRLS diverges, DESeq2 falls back to optim
    let f = deseq2_objective(&[0.0, 0.0, 0.0, 0.0, 0.0, 535.0], &NF, 10.0);
    let start = [-2.3025850929940463, 2.8684125693342355];
    let o = optim_lbfgsb(&start, f, None, &[-30.0], &[30.0], &LbfgsbControl::default()).unwrap();
    let e = Expected {
        par: &[-14.945714046039754, 22.4537950628057],
        value: 26.105026251759849,
        counts: 30,
        convergence: 0,
        message: CONV_F,
    };
    check("nb_degenerate", &o, &e, TOL);
}

#[test]
fn deseq2_regular_gene() {
    let f = deseq2_objective(&[3.0, 0.0, 1.0, 12.0, 25.0, 9.0], &NF, 0.3);
    let start = [-0.37081801898333955, 3.0375758063390386];
    let o = optim_lbfgsb(&start, f, None, &[-30.0], &[30.0], &LbfgsbControl::default()).unwrap();
    let e = Expected {
        par: &[0.39792998395163026, 3.5550732558317546],
        value: 30.291087097020281,
        counts: 10,
        convergence: 0,
        message: CONV_F,
    };
    check("nb_regular", &o, &e, TOL);
}

#[test]
fn deseq2_zero_group_gene() {
    let f = deseq2_objective(&[0.0, 0.0, 0.0, 4.0, 7.0, 2.0], &NF, 0.05);
    let start = [-2.3025850929940468, 3.6982855264553116];
    let o = optim_lbfgsb(&start, f, None, &[-30.0], &[30.0], &LbfgsbControl::default()).unwrap();
    let e = Expected {
        par: &[-15.932304128469019, 18.078580673331835],
        value: 21.867245348459971,
        counts: 28,
        convergence: 0,
        message: CONV_F,
    };
    check("nb_zero_group", &o, &e, TOL);
}

#[test]
fn dnbinom_and_dnorm_match_r() {
    let cases: [(f64, f64, f64, f64); 12] = [
        (0.0, 0.10000000000000001, 3.0, -0.34339872044851466),
        (535.0, 0.10000000000000001, 89.0, -9.1868574381690635),
        (535.0, 0.10000000000000001, 0.5, -105.62804568873679),
        (1.0, 0.10000000000000001, 0.5, -2.6640825967108062),
        (7.0, 2.5, 7.2999999999999998, -2.5972962079022586),
        (12.0, 1000000.0, 10.0, -2.3561973797207654),
        (3.0, 0.75, 1.0, -2.822545601162604),
        (100.0, 20.0, 80.0, -4.5574768843698319),
        (2.0, 1000000000000.0, 3.0, -1.4959226032287258),
        (5.0, 0.29999999999999999, 0.001, -30.780203329305621),
        (40.0, 3.7000000000000002, 41.200000000000003, -4.0219985330173831),
        (1.0, 1.0, 1.0, -1.3862943611198908),
    ];
    for (x, size, mu, r) in cases {
        let v = dnbinom_mu_log(x, size, mu);
        assert_eq!(v, r, "dnbinom({x}, size={size}, mu={mu}) = {v:e}, R {r:e}");
    }
    let sd = (1.0 / 1e-6f64).sqrt();
    for (x, r) in [(0.0, -7.8266938121868099), (1.5, -7.8266949371868098), (-23.100000000000001, -7.8269606171868098), (30.0, -7.8271438121868098)] {
        assert_eq!(dnorm_log(x, 0.0, sd), r, "dnorm({x})");
    }
}
