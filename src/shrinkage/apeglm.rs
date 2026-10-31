//! apeglm (Approximate Posterior Estimation for GLM) shrinkage
//!
//! Implements the apeglm method (Zhu, Ibrahim, Love 2018) for log fold change
//! shrinkage using Laplace approximation with a Cauchy (t, df=1) prior.
//!
//! This is a 1:1 port of:
//! - R package `apeglm` (azhu513/apeglm) - `apeglm()`, `apeglm.single()`, `priorVar()`
//! - C++ `nbinomGLM.cpp` - L-BFGS MAP estimation
//! - DESeq2 `lfcShrink()` - data preparation and post-processing
//!
//! Algorithm flow (matching R's `method="nbinomCR"`):
//! 1. Estimate prior variance via `priorVar()` (method of moments)
//! 2. Compute normalization constants (`cnst`) per gene
//! 3. L-BFGS MAP estimation for all nonzero genes (two runs for convergence check)
//! 4. Numerical Hessian at MAP → Laplace posterior SD
//! 5. FSR (false sign rate) and s-values

use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::VecDeque;

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};
use crate::io::DESeqResults;

// ============================================================================
// Public API
// ============================================================================

/// Parameters for apeglm shrinkage
/// R equivalent: apeglm() control parameters in apeglm package (azhu513/apeglm)
#[derive(Debug, Clone)]
pub struct ApeglmParams {
    /// Multiplier for prior scale (default: 1.0, R's `multiplier`)
    pub multiplier: f64,
    /// Maximum prior scale (default: 1.0, R caps at 1.0)
    pub max_prior_scale: f64,
    /// Prior scale for non-shrunk coefficients (default: 15.0, R's `prior.no.shrink.scale`)
    pub no_shrink_scale: f64,
    /// L-BFGS max iterations (default: 300, R's optim_lbfgs maxit)
    pub lbfgs_max_iter: usize,
    /// L-BFGS convergence tolerance (default: 1e-8)
    pub lbfgs_epsilon: f64,
    /// L-BFGS memory size (default: 6)
    pub lbfgs_m: usize,
    /// Hessian finite difference step size (default: 1e-3, R's ndeps)
    pub hessian_step: f64,
    /// Convergence check threshold for two-init comparison (default: 0.01)
    pub convergence_threshold: f64,
}

impl Default for ApeglmParams {
    fn default() -> Self {
        Self {
            multiplier: 1.0,
            max_prior_scale: 1.0,
            no_shrink_scale: 15.0,
            lbfgs_max_iter: 300,
            lbfgs_epsilon: 1e-8,
            lbfgs_m: 6,
            hessian_step: 1e-3,
            convergence_threshold: 0.01,
        }
    }
}

/// Prior control parameters (matches R's `prior.control` list)
#[derive(Debug, Clone)]
struct PriorControl {
    /// Indices of coefficients NOT shrunk (0-indexed; typically intercept = 0)
    no_shrink: Vec<usize>,
    /// Indices of coefficients to shrink (0-indexed)
    shrink: Vec<usize>,
    /// Prior variance for shrink coefficients (S^2 in C++)
    prior_scale_sq: f64,
    /// Prior SD for non-shrink coefficients (sigma in R, sigma^2 = 225)
    no_shrink_scale_sq: f64,
}

/// Result of apeglm shrinkage for a single gene
#[derive(Debug, Clone)]
struct GeneApeglmResult {
    /// MAP estimates for all coefficients (natural log scale)
    map: Vec<f64>,
    /// Posterior SDs for all coefficients
    sd: Vec<f64>,
    /// False sign rate for the target coefficient
    fsr: f64,
    /// Whether optimization converged
    _converged: bool,
}

/// Apply apeglm shrinkage to log fold changes
/// R equivalent: lfcShrink(type="apeglm") in lfcShrink.R
///
/// Matches R's `lfcShrink(dds, coef=coefNum, type="apeglm")` with
/// `apeMethod="nbinomCR"` (the default).
pub fn shrink_lfc_apeglm(
    dds: &DESeqDataSet,
    results: &mut DESeqResults,
    coef_idx: usize,
    params: &ApeglmParams,
) -> Result<()> {
    shrink_lfc_apeglm_with_counts(dds, None, results, coef_idx, params)
}

/// [`shrink_lfc_apeglm`] with `Y = fit_counts` instead of the dataset's counts.
/// After outlier replacement, R's counts(dds) are the original counts again
/// (refitWithoutOutliers restores them), and lfcShrink() passes those to apeglm
/// with the refitted dispersions and MLE.
/// R equivalent: lfcShrink(type="apeglm") after DESeq() replaced outliers
pub fn shrink_lfc_apeglm_with_counts(
    dds: &DESeqDataSet,
    fit_counts: Option<&ndarray::Array2<f64>>,
    results: &mut DESeqResults,
    coef_idx: usize,
    params: &ApeglmParams,
) -> Result<()> {
    let counts = fit_counts.map(|c| c.view()).unwrap_or_else(|| dds.counts().counts());
    if !dds.has_size_or_norm_factors() {
        return Err(DeseqError::InvalidContrast {
            reason: "Size factors required for apeglm".to_string(),
        });
    }
    let dispersions = dds.dispersions().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Dispersions required for apeglm".to_string(),
    })?;
    let design = dds.design_matrix().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Design matrix required for apeglm".to_string(),
    })?;
    let _coefficients = dds.coefficients().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Coefficients required for apeglm".to_string(),
    })?;

    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();
    let n_coefs = design.ncols();

    // Offset matrix [genes x samples], one row per gene below
    // R: offset <- matrix(log(sizeFactors(dds)), nrow=nrow(dds), ncol=ncol(dds), byrow=TRUE)
    //    or, with normalization factors, offset <- log(normalizationFactors(dds))
    let gene_offsets = |i: usize| -> Vec<f64> { dds.gene_norm_factors(i).unwrap().iter().map(|f| f.ln()).collect() };

    // Identify nonzero genes: R filters with nonzero <- rowSums(Y) > 0
    // In R, zero-count genes get NA LFC/SE, so priorVar automatically excludes them
    let nonzero: Vec<bool> = (0..n_genes)
        .map(|i| {
            let row_sum: f64 = (0..n_samples).map(|j| counts[[i, j]]).sum();
            row_sum > 0.0
        })
        .collect();

    // Prepare MLE in natural log scale: mle = log(2) * cbind(res$log2FoldChange, res$lfcSE)
    let ln2 = 2.0_f64.ln();
    let mle_lfc: Vec<f64> = results.log2_fold_changes.iter().map(|lfc| ln2 * lfc).collect();
    let mle_se: Vec<f64> = results.lfc_se.iter().map(|se| ln2 * se).collect();

    // Step 1: Estimate prior variance via priorVar (R's apeglm:::priorVar)
    // Only use nonzero genes (matching R where zero-count genes have NA MLE)
    let nonzero_mle_lfc: Vec<f64> = mle_lfc
        .iter()
        .zip(nonzero.iter())
        .filter(|(_, &nz)| nz)
        .map(|(&lfc, _)| lfc)
        .collect();
    let nonzero_mle_se: Vec<f64> = mle_se
        .iter()
        .zip(nonzero.iter())
        .filter(|(_, &nz)| nz)
        .map(|(&se, _)| se)
        .collect();
    let prior_var = prior_var(&nonzero_mle_lfc, &nonzero_mle_se);
    let prior_scale = (params.multiplier * prior_var.sqrt()).min(params.max_prior_scale);
    log::info!(
        "[apeglm] prior_var={:.6}, prior_scale={:.7}, n_nonzero={}",
        prior_var,
        prior_scale,
        nonzero.iter().filter(|&&nz| nz).count()
    );

    // Step 2: Set up prior control
    // R: no.shrink = setdiff(seq_len(ncol(x)), coef) — everything except target coef
    // coef_idx is 0-based in Rust, 1-based in R
    let no_shrink: Vec<usize> = (0..n_coefs).filter(|&k| k != coef_idx).collect();
    let shrink: Vec<usize> = vec![coef_idx];

    let prior = PriorControl {
        no_shrink,
        shrink,
        prior_scale_sq: prior_scale * prior_scale,
        no_shrink_scale_sq: params.no_shrink_scale * params.no_shrink_scale,
    };

    // Convert design matrix to row-major Vec<Vec<f64>> for efficient per-gene access
    let design_rows: Vec<Vec<f64>> = (0..n_samples)
        .map(|j| (0..n_coefs).map(|k| design[[j, k]]).collect())
        .collect();

    // Step 3: L-BFGS MAP estimation for all genes (matching C++ nbinomGLM)
    // Process each gene in parallel
    let gene_results: Vec<GeneApeglmResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            // Extract per-gene data
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[i, j]]).collect();
            let row_sum: f64 = gene_counts.iter().sum();

            // R: if (log.link & all(y == 0)) { return NA }
            if row_sum == 0.0 {
                return GeneApeglmResult {
                    map: vec![f64::NAN; n_coefs],
                    sd: vec![f64::NAN; n_coefs],
                    fsr: f64::NAN,
                    _converged: true,
                };
            }

            let size = 1.0 / dispersions[i]; // R: size = 1/param
            let offsets = gene_offsets(i);

            // Gene data struct for objective function
            let gd = GeneData {
                y: &gene_counts,
                x: &design_rows,
                size,
                offsets: &offsets,
                prior: &prior,
            };

            // Compute normalization constant (cnst)
            // R: cnst <- nbinomFn(init=0, ..., cnst=0); cnst <- ifelse(cnst > 1, cnst, 1)
            let init_zero = vec![0.0; n_coefs];
            let cnst_raw = nb_neg_log_posterior(&init_zero, &gd);
            let cnst = if cnst_raw > 1.0 { cnst_raw } else { 1.0 };

            // L-BFGS run 1: init = [0.1, -0.1, 0.1, -0.1, ...]
            let mut init1: Vec<f64> = (0..n_coefs)
                .map(|k| if k % 2 == 0 { 0.1 } else { -0.1 })
                .collect();
            let result1 = lbfgs_minimize(
                |x, g| nb_neg_log_posterior_and_grad(x, g, &gd, cnst),
                &mut init1,
                params.lbfgs_max_iter,
                params.lbfgs_epsilon,
                params.lbfgs_m,
            );

            // L-BFGS run 2: init = [-0.1, 0.1, -0.1, 0.1, ...]
            let mut init2: Vec<f64> = (0..n_coefs)
                .map(|k| if k % 2 == 0 { -0.1 } else { 0.1 })
                .collect();
            let _result2 = lbfgs_minimize(
                |x, g| nb_neg_log_posterior_and_grad(x, g, &gd, cnst),
                &mut init2,
                params.lbfgs_max_iter,
                params.lbfgs_epsilon,
                params.lbfgs_m,
            );

            // R (nbinomCppRoutine): the MAP is run 1; the row counts as not converged
            // when the C++ optimiser failed (status -1) or the two runs differ by more
            // than 0.01 in any coefficient.
            let max_delta = init1
                .iter()
                .zip(init2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let prefit_converged = !result1.failed && max_delta <= params.convergence_threshold;

            // R (optimNbinomHess): for a converged row, use optimHess at the C++ MAP;
            // if the row did not converge or that gives a non-positive variance,
            // re-optimise with optim(method = "BFGS") from the C++ MAP (optimNbinom),
            // with the Hessian from optim(hessian = TRUE) at its solution.
            let converged_fit = if prefit_converged {
                let hessian = optim_hessian(&init1, &gd, params.hessian_step);
                let (sd, valid) = laplace_sd_from_hessian(&hessian, n_coefs);
                valid.then_some((init1.clone(), sd))
            } else {
                None
            };
            let (map, final_sd) = converged_fit.unwrap_or_else(|| {
                let mut beta = init1.clone();
                // optimNbinom: cnst <- -nbinomFn(init, ..., cnst=0) - 1
                let shift = -nb_neg_log_posterior(&beta, &gd) - 1.0;
                r_vmmin(
                    |b| nb_neg_log_posterior(b, &gd) + shift,
                    |b, g| nb_gradient(b, g, &gd),
                    &mut beta,
                    100,
                    f64::NEG_INFINITY,
                    f64::EPSILON.sqrt(),
                );
                let hessian = optim_hessian(&beta, &gd, params.hessian_step);
                // R: if any(diag(cov.mat) <= 0), sd is NA but the map is kept
                let (sd, _) = laplace_sd_from_hessian(&hessian, n_coefs);
                (beta, sd)
            });

            // Step 5: Compute FSR (false sign rate)
            // R: fsr = pnorm(-|map[coef]|, 0, sd[coef])
            let fsr = if final_sd[coef_idx].is_finite() && final_sd[coef_idx] > 0.0 {
                let normal = Normal::new(0.0, final_sd[coef_idx]).unwrap();
                normal.cdf(-map[coef_idx].abs())
            } else {
                f64::NAN
            };

            GeneApeglmResult {
                map,
                sd: final_sd,
                fsr,
                _converged: prefit_converged,
            }
        })
        .collect();

    // Step 6: Compute s-values from FSR
    let fsr_values: Vec<f64> = gene_results.iter().map(|r| r.fsr).collect();
    let s_values = svalue(&fsr_values);

    // Step 7: Update results
    // R: res$log2FoldChange = log2(exp(1)) * fit$map[, coefNum]
    // R: res$lfcSE = log2(exp(1)) * fit$sd[, coefNum]
    let log2_e = 1.0_f64 / ln2; // log2(e) = 1/ln(2)

    for (i, gr) in gene_results.iter().enumerate() {
        results.log2_fold_changes[i] = log2_e * gr.map[coef_idx];
        results.lfc_se[i] = log2_e * gr.sd[coef_idx];
    }

    // Note: p-values and padj remain from the Wald test (R does the same for svalue=FALSE)
    // When svalue=TRUE, R replaces padj with s-values, but by default (lfcThreshold=0),
    // it keeps pvalue and padj from the original results
    let _ = s_values; // s-values computed but not used unless svalue mode

    Ok(())
}

// ============================================================================
// Prior Variance Estimation (matches R's apeglm:::priorVar)
// ============================================================================

/// Estimate prior variance using method of moments
///
/// Matches R's `priorVar(mle)` where `mle = log(2) * cbind(LFC, SE)`.
///
/// Solves the fixed-point equation:
///   A = sum((S_i - D_i) * I_i(A)) / sum(I_i(A))
/// where:
///   S_i = X_i^2 (squared MLE LFC)
///   D_i = SE_i^2 (variance of MLE)
///   I_i(A) = 1 / (2 * (A + D_i)^2) (information weight)
fn prior_var(mle_lfc: &[f64], mle_se: &[f64]) -> f64 {
    let min_var = 0.001 * 0.001; // 1e-6
    let max_var = 20.0 * 20.0; // 400.0

    // Filter valid (non-NA) entries
    // R: keep <- !is.na(mle[, 1])
    let valid: Vec<(f64, f64)> = mle_lfc
        .iter()
        .zip(mle_se.iter())
        .filter(|(lfc, se)| lfc.is_finite() && se.is_finite())
        .map(|(&lfc, &se)| (lfc, se))
        .collect();

    if valid.is_empty() {
        return min_var;
    }

    // X = mle[,1], D = mle[,2]^2, S = X^2
    let x_vals: Vec<f64> = valid.iter().map(|(lfc, _)| *lfc).collect();
    let d_vals: Vec<f64> = valid.iter().map(|(_, se)| se * se).collect();
    let s_vals: Vec<f64> = x_vals.iter().map(|x| x * x).collect();

    // Ahat(A) = sum((S - D) * I(A)) / sum(I(A))
    // I(A) = 1 / (2 * (A + D)^2)
    let ahat = |a: f64| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..s_vals.len() {
            let inv_info = 2.0 * (a + d_vals[j]).powi(2);
            let info = 1.0 / inv_info;
            num += (s_vals[j] - d_vals[j]) * info;
            den += info;
        }
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    };

    // objective(A) = Ahat(A) - A
    let objective = |a: f64| -> f64 { ahat(a) - a };

    // R: if (objective(min.var) < 0) { zero <- min.var }
    if objective(min_var) < 0.0 {
        return min_var;
    }

    // R: uniroot(objective, interval = c(min.var, max.var))$root, with uniroot's
    // default tol = .Machine$double.eps^0.25. The root is only that precise, and
    // the prior scale it sets changes every MAP estimate, so reproduce R's solver
    // (and its stopping point) rather than solving more accurately.
    let f_lo = objective(min_var);
    let f_hi = objective(max_var);
    if f_lo * f_hi > 0.0 {
        // uniroot stops with "f() values at end points not of opposite sign"
        return if f_lo.abs() < f_hi.abs() { min_var } else { max_var };
    }
    r_zeroin(objective, min_var, max_var, f_lo, f_hi, f64::EPSILON.powf(0.25), 1000)
}

/// Brent's root finder exactly as R's R_zeroin2 (src/library/stats/src/zeroin.c),
/// which uniroot() calls. `fa`, `fb` are f(ax), f(bx), which must differ in sign.
fn r_zeroin<F: Fn(f64) -> f64>(f: F, ax: f64, bx: f64, fa: f64, fb: f64, tol: f64, maxit: usize) -> f64 {
    let (mut a, mut b) = (ax, bx);
    let (mut fa, mut fb) = (fa, fb);
    let (mut c, mut fc) = (a, fa);

    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }

    for _ in 0..=maxit {
        let prev_step = b - a;
        if fc.abs() < fb.abs() {
            // swap so that b is the best approximation
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;

        if new_step.abs() <= tol_act || fb == 0.0 {
            return b;
        }

        // Try interpolation if the previous step was large enough and in the right direction
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q);
            if a == c {
                // linear interpolation
                let t1 = fb / fa;
                p = cb * t1;
                q = 1.0 - t1;
            } else {
                // inverse quadratic interpolation
                let qq = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                p = t2 * (cb * qq * (qq - t1) - (b - a) * (t1 - 1.0));
                q = (qq - 1.0) * (t1 - 1.0) * (t2 - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if p < (0.75 * cb * q - (tol_act * q).abs() / 2.0) && p < (prev_step * q / 2.0).abs() {
                new_step = p / q;
            }
        }

        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }
        a = b;
        fa = fb;
        b += new_step;
        fb = f(b);
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    b
}

// ============================================================================
// Negative Binomial Log-Posterior (matches C++ optimFun::f_grad)
// ============================================================================

/// Per-gene data needed for objective function evaluation
struct GeneData<'a> {
    /// Count vector for this gene [n_samples]
    y: &'a [f64],
    /// Design matrix rows [n_samples][n_coefs]
    x: &'a [Vec<f64>],
    /// NB size parameter (1/dispersion)
    size: f64,
    /// Log(size_factors) [n_samples]
    offsets: &'a [f64],
    /// Prior control parameters
    prior: &'a PriorControl,
}

/// Compute negative log-posterior WITHOUT normalization
///
/// Matches R's `nbinomFn(beta, ..., cnst=0)`:
///   f = -sum(w * (y * (xbeta + offset) - (y + size) * log(size + exp(xbeta + offset)))) + neg_prior
fn nb_neg_log_posterior(beta: &[f64], gd: &GeneData) -> f64 {
    let n = gd.y.len();
    let mut neg_log_lik = 0.0;

    for j in 0..n {
        // xbeta = X[j,:] * beta
        let xbeta: f64 = gd.x[j].iter().zip(beta.iter()).map(|(x, b)| x * b).sum();
        let xbeta_off = xbeta + gd.offsets[j];
        let exp_xbeta_off = xbeta_off.exp();

        // d = y * xbeta - (y + size) * log(exp(xbeta + offset) + size)
        // (y * xbeta, not y * (xbeta + offset), exactly as apeglm's nbinomFn and C++
        // f_grad: the constant changes cnst and the LBFGS relative stopping test)
        let a = gd.y[j] + gd.size;
        let b = exp_xbeta_off + gd.size;
        let d = gd.y[j] * xbeta - a * b.ln();

        // weights = 1.0 (all weights are 1 in default DESeq2 usage)
        neg_log_lik -= d;
    }

    // Prior terms
    let mut neg_prior = 0.0;

    // Normal prior for non-shrink coefficients: beta^2 / (2 * sigma^2)
    for &k in &gd.prior.no_shrink {
        neg_prior += beta[k] * beta[k] / (2.0 * gd.prior.no_shrink_scale_sq);
    }

    // Cauchy prior for shrink coefficients: log(1 + beta^2 / S^2)
    for &k in &gd.prior.shrink {
        neg_prior += (1.0 + beta[k] * beta[k] / gd.prior.prior_scale_sq).ln();
    }

    neg_log_lik + neg_prior
}

/// Compute gradient of negative log-posterior (no cnst normalization)
///
/// Matches R's `nbinomGr(beta, ...)`:
///   grad = -X^T * (w * (y - (y+size)*exp(xb+off)/(size+exp(xb+off)))) + prior_grad
fn nb_gradient(beta: &[f64], grad: &mut [f64], gd: &GeneData) {
    let n = gd.y.len();
    let n_coefs = beta.len();

    for g in grad.iter_mut() {
        *g = 0.0;
    }

    for j in 0..n {
        let xbeta: f64 = gd.x[j].iter().zip(beta.iter()).map(|(x, b)| x * b).sum();
        let xbeta_off = xbeta + gd.offsets[j];
        let exp_xbeta_off = xbeta_off.exp();

        let a = gd.y[j] + gd.size;
        let b = exp_xbeta_off + gd.size;
        let c = gd.y[j] - a * exp_xbeta_off / b;

        // weights = 1.0 in default DESeq2 usage
        for k in 0..n_coefs {
            grad[k] -= gd.x[j][k] * c;
        }
    }

    // Prior gradient
    for &k in &gd.prior.no_shrink {
        grad[k] += beta[k] / gd.prior.no_shrink_scale_sq;
    }
    for &k in &gd.prior.shrink {
        let b2 = beta[k] * beta[k];
        let s2 = gd.prior.prior_scale_sq;
        grad[k] += 2.0 * beta[k] / (s2 + b2);
    }
}

/// Compute normalized negative log-posterior AND gradient
///
/// Matches C++ `optimFun::f_grad()`:
///   f = neg_log_post / cnst + 10.0
///   grad = d_neg_log_post / cnst
fn nb_neg_log_posterior_and_grad(
    beta: &[f64],
    grad: &mut [f64],
    gd: &GeneData,
    cnst: f64,
) -> f64 {
    let n = gd.y.len();
    let n_coefs = beta.len();

    // Zero gradient
    for g in grad.iter_mut() {
        *g = 0.0;
    }

    let mut neg_log_lik = 0.0;

    for j in 0..n {
        // xbeta = X[j,:] * beta
        let xbeta: f64 = gd.x[j].iter().zip(beta.iter()).map(|(x, b)| x * b).sum();
        let xbeta_off = xbeta + gd.offsets[j];
        let exp_xbeta_off = xbeta_off.exp();

        let a = gd.y[j] + gd.size; // y + size
        let b = exp_xbeta_off + gd.size; // exp(xbeta + offset) + size

        // Negative log-likelihood contribution
        // R: y * log(mu) where mu = exp(xbeta + offset), so log(mu) = xbeta_off
        let d = gd.y[j] * xbeta - a * b.ln();
        neg_log_lik -= d; // weights = 1.0

        // Gradient of negative log-likelihood
        // c = y - (y + size) * exp(xbeta + offset) / (exp(xbeta + offset) + size)
        let c = gd.y[j] - a * exp_xbeta_off / b;
        // cw = c * weight = c * 1.0
        // d_neg_lik[k] -= x[j,k] * c  (accumulated as positive for neg log-lik)
        for k in 0..n_coefs {
            grad[k] -= gd.x[j][k] * c;
        }
    }

    // Prior gradient
    let mut neg_prior = 0.0;

    for &k in &gd.prior.no_shrink {
        neg_prior += beta[k] * beta[k] / (2.0 * gd.prior.no_shrink_scale_sq);
        grad[k] += beta[k] / gd.prior.no_shrink_scale_sq;
    }

    for &k in &gd.prior.shrink {
        let b2 = beta[k] * beta[k];
        let s2 = gd.prior.prior_scale_sq;
        neg_prior += (1.0 + b2 / s2).ln();
        grad[k] += 2.0 * beta[k] / (s2 + b2);
    }

    // Normalize by cnst (matching C++ exactly)
    let f = (neg_log_lik + neg_prior) / cnst + 10.0;
    for g in grad.iter_mut() {
        *g /= cnst;
    }

    f
}

// ============================================================================
// L-BFGS Optimizer (exact match of LBFGSpp via RcppNumerical 0.7-0)
// ============================================================================

/// Result of L-BFGS optimization
struct LbfgsResult {
    _value: f64,
    _iterations: usize,
    /// A convergence test was met (diagnostic; R only distinguishes `failed`)
    #[allow(dead_code)]
    converged: bool,
    /// The line search failed, where LBFGSpp throws and RcppNumerical's optim_lbfgs
    /// reports status -1. Reaching the iteration limit is still status 0.
    failed: bool,
}

/// L-BFGS minimization — exact match of LBFGSpp::LBFGSSolver::minimize()
///
/// Called from RcppNumerical as:
///   optim_lbfgs(nll, beta, fopt, 300, 1e-8, 1e-8)
///
/// LBFGSpp parameters (from wrapper.h):
///   epsilon = epsilon_rel = 1e-8, past = 1, delta = 1e-8
///   max_iterations = 300, max_linesearch = 100
///   linesearch = BACKTRACKING_STRONG_WOLFE
///   m = 6, ftol = 1e-4, wolfe = 0.9
///
/// `linesearch` only names the condition: the solver's line search algorithm is
/// the LBFGSSolver template default, which in the LBFGSpp bundled with
/// RcppNumerical 0.7-0 is LineSearchNocedalWright, not LineSearchBacktracking.
fn lbfgs_minimize<F>(
    f_grad: F,
    x: &mut [f64],
    max_iter: usize,
    epsilon: f64,
    m: usize,
) -> LbfgsResult
where
    F: Fn(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let fpast = 1_usize; // LBFGSpp wrapper sets param.past = 1
    let delta = epsilon; // wrapper sets param.delta = eps_f = same as eps_g

    // Line search parameters (LBFGSpp defaults)
    let ftol = 1e-4;
    let wolfe_param = 0.9;
    let max_linesearch = 100;
    let sy_eps = f64::EPSILON; // ~2.2e-16, LBFGSpp: std::numeric_limits<double>::epsilon()

    // L-BFGS history (BFGS matrix approximation)
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(m);

    // Past function values for delta-based convergence (circular buffer)
    let mut f_past_buf: Vec<f64> = vec![0.0; fpast];

    // ---- Initial evaluation ----
    let mut grad = vec![0.0; n];
    let mut fx = f_grad(x, &mut grad);
    let mut gnorm = grad.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
    f_past_buf[0] = fx;

    // Early exit if initial x is already a minimizer
    // LBFGSpp: if (m_gnorm <= param.epsilon || m_gnorm <= param.epsilon_rel * x.norm())
    {
        let xnorm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if gnorm <= epsilon || gnorm <= epsilon * xnorm {
            return LbfgsResult {
                _value: fx,
                _iterations: 1,
                converged: true,
                failed: false,
            };
        }
    }

    // Initial direction: d = -g
    let mut drt: Vec<f64> = grad.iter().map(|gi| -gi).collect();
    // Initial step size: 1/||d|| (LBFGSpp: step = 1/drt.norm())
    let mut step = 1.0 / drt.iter().map(|di| di * di).sum::<f64>().sqrt();

    // Saved previous state
    let mut xp = vec![0.0; n];
    let mut gradp = vec![0.0; n];

    let mut k = 1_usize;
    loop {
        // Save current x and gradient
        xp.copy_from_slice(x);
        gradp.copy_from_slice(&grad);
        // ---- Line search (LBFGSpp LineSearchNocedalWright) ----
        // On failure LBFGSpp throws; RcppNumerical catches it, keeps x at the last
        // trial point and reports status -1.
        if line_search_nocedal_wright(
            &f_grad, &xp, &drt, &mut step, &mut fx, &mut grad, x, ftol, wolfe_param, max_linesearch,
        )
        .is_err()
        {
            return LbfgsResult {
                _value: fx,
                _iterations: k,
                converged: false,
                failed: true,
            };
        }

        // New gradient norm
        gnorm = grad.iter().map(|gi| gi * gi).sum::<f64>().sqrt();

        // ---- Convergence test: gradient ----
        // LBFGSpp: gnorm <= epsilon || gnorm <= epsilon_rel * x.norm()
        {
            let xnorm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
            if gnorm <= epsilon || gnorm <= epsilon * xnorm {
                return LbfgsResult {
                    _value: fx,
                    _iterations: k,
                    converged: true,
                    failed: false,
                };
            }
        }

        // ---- Convergence test: objective function value (past-based) ----
        // LBFGSpp: |f_past - fx| <= delta * max(max(|fx|, |f_past|), 1)
        if k >= fpast {
            let fxd = f_past_buf[k % fpast];
            let denom = fx.abs().max(fxd.abs()).max(1.0);
            if (fxd - fx).abs() <= delta * denom {
                return LbfgsResult {
                    _value: fx,
                    _iterations: k,
                    converged: true,
                    failed: false,
                };
            }
        }
        f_past_buf[k % fpast] = fx;

        // ---- Maximum iterations check ----
        if k >= max_iter {
            return LbfgsResult {
                _value: fx,
                _iterations: k,
                converged: false,
                failed: false,
            };
        }

        // ---- Update s and y vectors ----
        let s: Vec<f64> = (0..n).map(|i| x[i] - xp[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| grad[i] - gradp[i]).collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
        let yy: f64 = y.iter().map(|yi| yi * yi).sum();

        // LBFGSpp: if (s.dot(y) > eps * y.squaredNorm())
        if sy > sy_eps * yy {
            if s_hist.len() >= m {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s);
            y_hist.push_back(y);
        }

        // ---- Compute search direction: d = -H * g ----
        drt = grad.iter().map(|gi| -gi).collect();
        two_loop_recursion(&mut drt, &s_hist, &y_hist, &rho_hist);

        // Reset step = 1.0 for next iteration (LBFGSpp does this)
        step = 1.0;
        k += 1;
    }
}

/// Strong-Wolfe line search of Nocedal & Wright (2006), Algorithms 3.5 and 3.6:
/// a bracketing phase that doubles the step, then a zoom phase with a quadratic
/// interpolation (bisection when the interpolant is unusable).
/// Port of LBFGSpp's LineSearchNocedalWright as bundled with RcppNumerical 0.7-0.
/// `x`, `fx` and `grad` end at the accepted point; `Err` is where LBFGSpp throws.
#[allow(clippy::too_many_arguments)]
fn line_search_nocedal_wright<F>(
    f_grad: &F,
    xp: &[f64],
    drt: &[f64],
    step: &mut f64,
    fx: &mut f64,
    grad: &mut [f64],
    x: &mut [f64],
    ftol: f64,
    wolfe: f64,
    max_linesearch: usize,
) -> std::result::Result<(), ()>
where
    F: Fn(&[f64], &mut [f64]) -> f64,
{
    if *step <= 0.0 {
        return Err(());
    }
    let expansion = 2.0;
    let fx_init = *fx;
    let dg_init = dot(grad, drt);
    if dg_init > 0.0 {
        return Err(());
    }
    let test_decr = ftol * dg_init;
    let test_curv = -wolfe * dg_init;

    let evaluate = |step: f64, x: &mut [f64], grad: &mut [f64]| -> (f64, f64) {
        for i in 0..x.len() {
            x[i] = xp[i] + step * drt[i];
        }
        let f = f_grad(x, grad);
        (f, dot(grad, drt))
    };

    let (mut step_hi, mut fx_hi);
    let (mut step_lo, mut fx_lo, mut dg_lo) = (0.0, fx_init, dg_init);
    let mut x_lo = xp.to_vec();
    let mut grad_lo = grad.to_vec();

    // Bracketing phase
    let mut iter = 0;
    loop {
        let (f, dg) = evaluate(*step, x, grad);
        *fx = f;
        if f - fx_init > *step * test_decr || (0.0 < step_lo && f >= fx_lo) {
            step_hi = *step;
            fx_hi = f;
            break;
        }
        if dg.abs() <= test_curv {
            return Ok(());
        }
        step_hi = step_lo;
        fx_hi = fx_lo;
        step_lo = *step;
        fx_lo = f;
        dg_lo = dg;
        x.swap_with_slice(&mut x_lo);
        grad.swap_with_slice(&mut grad_lo);
        if dg >= 0.0 {
            break;
        }
        iter += 1;
        if iter >= max_linesearch {
            // Every step decreased f: return the latest (= the _lo state)
            x.swap_with_slice(&mut x_lo);
            grad.swap_with_slice(&mut grad_lo);
            return Ok(());
        }
        *step *= expansion;
    }

    // Zoom phase
    loop {
        *step = quad_interp(step_lo, step_hi, fx_lo, fx_hi, dg_lo);
        let (f, dg) = evaluate(*step, x, grad);
        *fx = f;
        if f - fx_init > *step * test_decr || f >= fx_lo {
            if *step == step_hi {
                return Err(());
            }
            step_hi = *step;
            fx_hi = f;
        } else {
            if dg.abs() <= test_curv {
                return Ok(());
            }
            if dg * (step_hi - step_lo) >= 0.0 {
                step_hi = step_lo;
                fx_hi = fx_lo;
            }
            if *step == step_lo {
                return Err(());
            }
            step_lo = *step;
            fx_lo = f;
            dg_lo = dg;
            x.swap_with_slice(&mut x_lo);
            grad.swap_with_slice(&mut grad_lo);
        }
        iter += 1;
        if iter >= max_linesearch {
            if step_lo <= 0.0 {
                return Err(());
            }
            *step = step_lo;
            *fx = fx_lo;
            x.swap_with_slice(&mut x_lo);
            grad.swap_with_slice(&mut grad_lo);
            return Ok(());
        }
    }
}

/// Minimiser of the quadratic through (step_lo, fx_lo) with slope dg_lo and
/// (step_hi, fx_hi); bisection when that is non-finite, outside the interval or
/// within 1% of its width from an end. LBFGSpp LineSearchNocedalWright::quad_interp.
fn quad_interp(step_lo: f64, step_hi: f64, fx_lo: f64, fx_hi: f64, dg_lo: f64) -> f64 {
    let fdiff = fx_hi - fx_lo;
    let sdiff = step_hi - step_lo;
    let smid = (step_hi + step_lo) / 2.0;
    let candid = (fdiff * step_lo - smid * sdiff * dg_lo) / (fdiff - sdiff * dg_lo);
    let end_dist = (candid - step_lo).abs().min((candid - step_hi).abs());
    let bisect = !candid.is_finite()
        || candid <= step_lo.min(step_hi)
        || candid >= step_lo.max(step_hi)
        || end_dist < 0.01 * sdiff.abs();
    if bisect {
        smid
    } else {
        candid
    }
}

/// Two-loop recursion for L-BFGS search direction
///
/// Computes d = -H * g where H is the L-BFGS Hessian approximation.
/// Input: d = -g (negative gradient)
/// Output: d = -H * g (search direction)
fn two_loop_recursion(
    d: &mut [f64],
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
    rho_hist: &VecDeque<f64>,
) {
    let m = s_hist.len();
    if m == 0 {
        return; // Use identity Hessian: d = -g (already set)
    }

    let mut alpha_vec = vec![0.0; m];

    // Forward loop (most recent to oldest)
    for i in (0..m).rev() {
        alpha_vec[i] = rho_hist[i] * dot(&s_hist[i], d);
        for j in 0..d.len() {
            d[j] -= alpha_vec[i] * y_hist[i][j];
        }
    }

    // Scale by initial Hessian approximation: gamma * I
    // gamma = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
    let last = m - 1;
    let ys: f64 = y_hist[last]
        .iter()
        .zip(s_hist[last].iter())
        .map(|(y, s)| y * s)
        .sum();
    let yy: f64 = y_hist[last].iter().map(|y| y * y).sum();
    let gamma = if yy > 0.0 { ys / yy } else { 1.0 };
    for dj in d.iter_mut() {
        *dj *= gamma;
    }

    // Backward loop (oldest to most recent)
    for i in 0..m {
        let beta = rho_hist[i] * dot(&y_hist[i], d);
        for j in 0..d.len() {
            d[j] += s_hist[i][j] * (alpha_vec[i] - beta);
        }
    }
}

/// Dot product
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ============================================================================
// Hessian Computation (matches R's optimHess with gr=nbinomGr)
// ============================================================================

/// Hessian of the negative log-posterior from central differences of the gradient,
/// exactly as R's optimHess (C optimhess): par + eps, then - 2 eps, then + eps back,
/// with eps = ndeps = 1e-3, followed by symmetrisation (H + H')/2.
fn optim_hessian(x: &[f64], gd: &GeneData, step: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];
    let mut dpar = x.to_vec();
    let mut df1 = vec![0.0; n];
    let mut df2 = vec![0.0; n];

    for i in 0..n {
        dpar[i] += step;
        nb_gradient(&dpar, &mut df1, gd);
        dpar[i] -= 2.0 * step;
        nb_gradient(&dpar, &mut df2, gd);
        for j in 0..n {
            hess[i][j] = (df1[j] - df2[j]) / (2.0 * step);
        }
        dpar[i] += step;
    }
    for i in 0..n {
        for j in 0..i {
            let tmp = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = tmp;
            hess[j][i] = tmp;
        }
    }
    hess
}

/// Variable-metric (BFGS) minimiser exactly as R's vmmin (src/appl/optim.c),
/// which optim(method = "BFGS") uses; `fn`/`gr` evaluations follow R's order.
/// Returns true when it stopped before `maxit` (R: fail = 0).
fn r_vmmin<F, G>(f: F, gr: G, b: &mut [f64], maxit: usize, abstol: f64, reltol: f64) -> bool
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64], &mut [f64]),
{
    const STEPREDN: f64 = 0.2;
    const ACCTOL: f64 = 0.0001;
    const RELTEST: f64 = 10.0;

    let n = b.len();
    let mut g = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut xv = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut bm = vec![vec![0.0; n]; n];

    let mut fval = f(b);
    if !fval.is_finite() {
        return false;
    }
    let mut fmin = fval;
    gr(b, &mut g);
    let mut gradcount = 1usize;
    let mut iter = 1usize;
    let mut ilast = gradcount;
    let mut count;

    loop {
        if ilast == gradcount {
            for i in 0..n {
                for j in 0..i {
                    bm[i][j] = 0.0;
                }
                bm[i][i] = 1.0;
            }
        }
        for i in 0..n {
            xv[i] = b[i];
            c[i] = g[i];
        }
        let mut gradproj = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..=i {
                s -= bm[i][j] * g[j];
            }
            for j in (i + 1)..n {
                s -= bm[j][i] * g[j];
            }
            t[i] = s;
            gradproj += s * g[i];
        }

        if gradproj < 0.0 {
            // search direction is downhill
            let mut steplength = 1.0;
            let mut accpoint = false;
            loop {
                count = 0;
                for i in 0..n {
                    b[i] = xv[i] + steplength * t[i];
                    if RELTEST + xv[i] == RELTEST + b[i] {
                        count += 1;
                    }
                }
                if count < n {
                    fval = f(b);
                    accpoint = fval.is_finite() && fval <= fmin + gradproj * steplength * ACCTOL;
                    if !accpoint {
                        steplength *= STEPREDN;
                    }
                }
                if count == n || accpoint {
                    break;
                }
            }
            let enough = fval > abstol && (fval - fmin).abs() > reltol * (fmin.abs() + reltol);
            // stop if the value is small or the relative change is low
            if !enough {
                count = n;
                fmin = fval;
            }
            if count < n {
                // making progress
                fmin = fval;
                gr(b, &mut g);
                gradcount += 1;
                iter += 1;
                let mut d1 = 0.0;
                for i in 0..n {
                    t[i] *= steplength;
                    c[i] = g[i] - c[i];
                    d1 += t[i] * c[i];
                }
                if d1 > 0.0 {
                    let mut d2 = 0.0;
                    for i in 0..n {
                        let mut s = 0.0;
                        for j in 0..=i {
                            s += bm[i][j] * c[j];
                        }
                        for j in (i + 1)..n {
                            s += bm[j][i] * c[j];
                        }
                        xv[i] = s;
                        d2 += s * c[i];
                    }
                    d2 = 1.0 + d2 / d1;
                    for i in 0..n {
                        for j in 0..=i {
                            bm[i][j] += (d2 * t[i] * t[j] - xv[i] * t[j] - t[i] * xv[j]) / d1;
                        }
                    }
                } else {
                    ilast = gradcount;
                }
            } else if ilast < gradcount {
                // no progress
                count = 0;
                ilast = gradcount;
            }
        } else {
            // uphill search: reset unless it has just been reset
            count = 0;
            if ilast == gradcount {
                count = n;
            } else {
                ilast = gradcount;
            }
        }
        if iter >= maxit {
            break;
        }
        if gradcount - ilast > 2 * n {
            ilast = gradcount; // periodic restart
        }
        if count == n && ilast == gradcount {
            break;
        }
    }
    iter < maxit
}

/// Compute Laplace posterior SDs from Hessian of negative log-posterior
///
/// The posterior covariance is inv(H) where H is the Hessian of neg_log_post.
/// At the MAP, H should be positive definite (since neg_log_post is convex there).
///
/// Returns (sd_vector, is_valid)
fn laplace_sd_from_hessian(hess: &[Vec<f64>], n: usize) -> (Vec<f64>, bool) {
    // For small n (typically 2-3), use direct inversion
    if n == 1 {
        if hess[0][0] > 0.0 {
            return (vec![(1.0 / hess[0][0]).sqrt()], true);
        } else {
            return (vec![f64::NAN], false);
        }
    }

    if n == 2 {
        // 2x2 matrix inversion
        let a = hess[0][0];
        let b = hess[0][1];
        let c = hess[1][0];
        let d = hess[1][1];
        let det = a * d - b * c;

        if det <= 0.0 {
            return (vec![f64::NAN; n], false);
        }

        let inv_00 = d / det;
        let inv_11 = a / det;

        if inv_00 <= 0.0 || inv_11 <= 0.0 {
            return (vec![f64::NAN; n], false);
        }

        return (vec![inv_00.sqrt(), inv_11.sqrt()], true);
    }

    // General case: Cholesky or direct inversion for small matrices
    // For n <= 5 (typical DESeq2), use explicit Gaussian elimination
    let inv = invert_symmetric_pd(hess, n);
    match inv {
        Some(inv_mat) => {
            let mut sd = vec![0.0; n];
            let mut valid = true;
            for i in 0..n {
                if inv_mat[i][i] > 0.0 {
                    sd[i] = inv_mat[i][i].sqrt();
                } else {
                    sd[i] = f64::NAN;
                    valid = false;
                }
            }
            (sd, valid)
        }
        None => (vec![f64::NAN; n], false),
    }
}

/// Invert a symmetric positive-definite matrix using Gaussian elimination
/// Returns None if matrix is not positive definite
fn invert_symmetric_pd(mat: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    // Create augmented matrix [A | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below
        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..2 * n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    for col in (0..n).rev() {
        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return None;
        }

        // Scale row
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }

        // Eliminate above
        for row in 0..col {
            let factor = aug[row][col];
            for j in 0..2 * n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse
    let inv: Vec<Vec<f64>> = (0..n).map(|i| aug[i][n..2 * n].to_vec()).collect();
    Some(inv)
}

// ============================================================================
// S-value computation (matches R's apeglm:::svalue)
// ============================================================================

/// Compute s-values from false sign rates
///
/// s-value = running average of sorted FSR values, mapped back to original order.
/// Matches R: `(cumsum(lfsr.sorted) / seq_along(lfsr))[rank(lfsr, ties="first", na.last=TRUE)]`
fn svalue(fsr: &[f64]) -> Vec<f64> {
    let n = fsr.len();
    if n == 0 {
        return vec![];
    }

    // Create indices sorted by FSR (NA/NaN last)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let fa = fsr[a];
        let fb = fsr[b];
        match (fa.is_nan(), fb.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    // Compute cumulative mean of sorted FSR
    let mut cumsum = 0.0;
    let mut sval_sorted = vec![f64::NAN; n];
    for (rank, &idx) in indices.iter().enumerate() {
        if fsr[idx].is_nan() {
            sval_sorted[rank] = f64::NAN;
        } else {
            cumsum += fsr[idx];
            sval_sorted[rank] = cumsum / (rank + 1) as f64;
        }
    }

    // Map back to original order
    let mut result = vec![f64::NAN; n];
    for (rank, &idx) in indices.iter().enumerate() {
        result[idx] = sval_sorted[rank];
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prior_var_basic() {
        // Simple test: all LFCs = 1.0, all SEs = 0.5
        let lfc = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.3, -0.3, 1.5, -1.5];
        let se = vec![0.5; 10];
        let pv = prior_var(&lfc, &se);

        // Prior variance should be positive and reasonable
        assert!(pv > 0.0, "prior_var should be positive, got {}", pv);
        assert!(pv < 400.0, "prior_var should be < max_var");
    }

    #[test]
    fn test_prior_var_small_se() {
        // When SE is small relative to LFC, prior var should be larger
        let lfc = vec![3.0, -3.0, 2.0, -2.0, 4.0, -4.0, 1.0, -1.0, 5.0, -5.0];
        let se = vec![0.1; 10];
        let pv = prior_var(&lfc, &se);
        assert!(pv > 0.5, "prior_var should be substantial for large LFCs");
    }

    #[test]
    fn test_prior_var_large_se() {
        // When SE is large, prior var should be small
        let lfc = vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15, 0.08, -0.08];
        let se = vec![2.0; 10];
        let pv = prior_var(&lfc, &se);
        // Should hit min_var since all S < D
        assert!(pv <= 0.001 * 0.001 + 1e-10, "prior_var should be min_var");
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        // Test L-BFGS on Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        let mut x = vec![-1.0, -1.0];
        let result = lbfgs_minimize(
            |x, g| {
                let t1 = 1.0 - x[0];
                let t2 = x[1] - x[0] * x[0];
                g[0] = -2.0 * t1 - 400.0 * x[0] * t2;
                g[1] = 200.0 * t2;
                t1 * t1 + 100.0 * t2 * t2
            },
            &mut x,
            1000,
            1e-10,
            6,
        );

        assert!(result.converged, "L-BFGS should converge on Rosenbrock");
        assert!(
            (x[0] - 1.0).abs() < 1e-4,
            "x[0] should be near 1.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 1.0).abs() < 1e-4,
            "x[1] should be near 1.0, got {}",
            x[1]
        );
    }

    #[test]
    fn test_lbfgs_quadratic() {
        // Simple quadratic: f(x) = x[0]^2 + 2*x[1]^2
        let mut x = vec![5.0, 3.0];
        let result = lbfgs_minimize(
            |x, g| {
                g[0] = 2.0 * x[0];
                g[1] = 4.0 * x[1];
                x[0] * x[0] + 2.0 * x[1] * x[1]
            },
            &mut x,
            100,
            1e-8,
            6,
        );

        assert!(result.converged);
        assert!(x[0].abs() < 1e-6, "x[0] should be ~0, got {}", x[0]);
        assert!(x[1].abs() < 1e-6, "x[1] should be ~0, got {}", x[1]);
    }

    #[test]
    fn test_svalue() {
        let fsr = vec![0.1, 0.05, 0.2, f64::NAN, 0.01];
        let sv = svalue(&fsr);

        // Sorted FSR: 0.01 (idx4), 0.05 (idx1), 0.1 (idx0), 0.2 (idx2), NaN (idx3)
        // Cumulative means: 0.01, 0.03, 0.053, 0.09, NaN

        assert!(sv[3].is_nan(), "NaN input should give NaN s-value");
        assert!(sv[4] < sv[1], "lower FSR should have lower s-value");
        assert!(sv[1] < sv[0], "lower FSR should have lower s-value");
    }

    #[test]
    fn test_finite_difference_hessian_quadratic() {
        // f(x) = 3*x[0]^2 + 2*x[0]*x[1] + 5*x[1]^2
        // H = [[6, 2], [2, 10]]
        let _prior = PriorControl {
            no_shrink: vec![],
            shrink: vec![],
            prior_scale_sq: 1.0,
            no_shrink_scale_sq: 225.0,
        };

        // We can't easily test with GeneData without real data,
        // but we can verify the 2x2 inversion
        let hess = vec![vec![6.0, 2.0], vec![2.0, 10.0]];
        let (sd, valid) = laplace_sd_from_hessian(&hess, 2);

        assert!(valid);
        // inv([[6,2],[2,10]]) = [[10,-2],[-2,6]] / (60-4) = [[10,-2],[-2,6]] / 56
        let expected_var_0: f64 = 10.0 / 56.0;
        let expected_var_1: f64 = 6.0 / 56.0;
        assert!(
            (sd[0] - expected_var_0.sqrt()).abs() < 1e-10,
            "sd[0] should be {}, got {}",
            expected_var_0.sqrt(),
            sd[0]
        );
        assert!(
            (sd[1] - expected_var_1.sqrt()).abs() < 1e-10,
            "sd[1] should be {}, got {}",
            expected_var_1.sqrt(),
            sd[1]
        );
    }
}
