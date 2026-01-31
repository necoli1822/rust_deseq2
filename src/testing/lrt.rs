//! Likelihood Ratio Test for differential expression
//!
//! LRT compares nested models (full vs reduced) to test for differential expression.
//! It's particularly useful for testing multiple coefficients at once, or when
//! testing factors with more than two levels.
//!
//! Algorithm (matches R DESeq2 nbinomLRT):
//! 1. Fit full and reduced NB GLMs via IRLS for every gene
//! 2. Compute log-likelihood for each fit
//! 3. LRT_stat = 2 * (logLike_full - logLike_reduced)
//! 4. pvalue = pchisq(LRT_stat, df = ncol(full) - ncol(reduced), lower.tail = FALSE)
//! 5. BH correction for padj

use ndarray::Array2;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;

use super::fdr::benjamini_hochberg;
use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};
use crate::io::{Contrast, DESeqResults};

/// Minimum mu value during GLM fitting (matches DESeq2's minmu = 0.5)
const MIN_MU: f64 = 0.5;

/// Maximum absolute value for non-intercept beta coefficients
const MAX_LFC_BETA: f64 = 30.0;

/// Maximum eta to prevent exp() overflow
const MAX_ETA: f64 = 700.0;

/// Ridge penalty scale: lambda = 1e-6 / ln(2)^2 (matches DESeq2 C++ fitBeta)
fn ridge_lambda() -> f64 {
    let ln2 = std::f64::consts::LN_2;
    1e-6 / (ln2 * ln2)
}

/// Negative binomial log-likelihood for a single observation.
///
/// Matches R's dnbinom(y, size=1/alpha, mu=mu, log=TRUE):
///   lgamma(y + size) - lgamma(size) - lgamma(y + 1)
///     + size * ln(size / (size + mu))
///     + y * ln(mu / (size + mu))
///
/// Edge cases:
///   - y == 0 && mu == 0  =>  0.0  (both log terms vanish)
///   - mu <= 0 || alpha <= 0  =>  NEG_INFINITY
fn nb_log_likelihood(y: f64, mu: f64, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // When both y and mu are 0, the probability mass is 1 => log(1) = 0
    if y == 0.0 && mu == 0.0 {
        return 0.0;
    }
    if mu <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let size = 1.0 / alpha;
    ln_gamma(y + size) - ln_gamma(size) - ln_gamma(y + 1.0)
        + size * (size / (size + mu)).ln()
        + y * (mu / (size + mu)).ln()
}

/// Result from fitting a NB GLM for a single gene.
struct GlmGeneResult {
    /// Fitted coefficients (on natural-log scale)
    betas: Vec<f64>,
    /// Fitted mean vector (length = n_samples)
    mu: Vec<f64>,
    /// Total log-likelihood = sum_j dnbinom(y_j, mu=mu_j, size=1/alpha, log=TRUE)
    log_likelihood: f64,
    /// Whether fitting converged (IRLS or optimizer fallback)
    #[allow(dead_code)]
    converged: bool,
}

/// Fit a negative binomial GLM for a single gene using IRLS.
///
/// This is a self-contained fitter that returns log-likelihood,
/// matching the algorithm in `fit_single_gene` from fitting.rs:
///   - OLS initialization on log(normalized_counts + 0.1)
///   - IRLS with ridge penalty lambda = 1e-6/ln(2)^2 for all coefficients
///   - Convergence: |dev - dev_old| / (|dev| + 0.1) < 1e-8
///   - Max iterations: 100
///   - Min mu: 0.5
///   - Max beta: 30
fn fit_nb_glm_gene(
    counts: &[f64],
    size_factors: &[f64],
    alpha: f64,
    design: &Array2<f64>,
    initial_beta: Option<&[f64]>,
) -> GlmGeneResult {
    let n_samples = counts.len();
    let n_coefs = design.ncols();
    let max_iter = 100;
    let tol = 1e-8;
    let lambda = ridge_lambda();

    // Initialize beta
    let mut beta = if let Some(init) = initial_beta {
        init.to_vec()
    } else {
        // OLS initialization on log(normalized_counts)
        let log_counts: Vec<f64> = counts
            .iter()
            .zip(size_factors.iter())
            .map(|(&c, &s)| {
                let norm_ct = if s > 0.0 { c / s } else { 0.0 };
                // R C++ fitBeta: y_hat(j) = std::log(k(i,j) / nf(i,j) + 0.1);
                (norm_ct + 0.1).ln()
            })
            .collect();

        // OLS: beta = (X'X)^-1 X'y
        let mut xtx = vec![0.0; n_coefs * n_coefs];
        let mut xty = vec![0.0; n_coefs];
        for i in 0..n_samples {
            for j in 0..n_coefs {
                for k in 0..n_coefs {
                    xtx[j * n_coefs + k] += design[[i, j]] * design[[i, k]];
                }
                xty[j] += design[[i, j]] * log_counts[i];
            }
        }
        let ols_beta = solve_symmetric_system(&xtx, &xty, n_coefs);

        // Fall back if OLS gave non-finite values
        if ols_beta.iter().any(|&b| !b.is_finite()) {
            let mean_count: f64 = counts
                .iter()
                .zip(size_factors.iter())
                .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
                .sum::<f64>()
                / n_samples as f64;
            let mut fallback = vec![0.0; n_coefs];
            fallback[0] = mean_count.max(0.1).ln();
            fallback
        } else {
            ols_beta
        }
    };

    // Compute initial deviance before loop
    let mut prev_deviance: f64 = {
        let mut init_dev = 0.0f64;
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
            let mu = (size_factors[i] * eta_clamped.exp()).max(MIN_MU);
            init_dev += -2.0 * nb_log_likelihood(counts[i], mu, alpha);
        }
        init_dev
    };

    let mut converged = false;

    // Allocate buffers once, outside the loop (fully overwritten each iteration)
    let mut mus = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];
    let mut working_response = vec![0.0; n_samples];

    for _iter in 0..max_iter {
        // Compute mu, weights, and working response
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
            let mu_raw = size_factors[i] * eta_clamped.exp();
            let mu_clamped = mu_raw.max(MIN_MU);
            mus[i] = mu_clamped;
            weights[i] = mu_clamped / (1.0 + alpha * mu_clamped);
            working_response[i] =
                (mu_clamped / size_factors[i]).ln() + (counts[i] - mu_clamped) / mu_clamped;
        }

        // Weighted least squares with ridge
        let new_beta =
            weighted_least_squares_ridge(design, &weights, &working_response, lambda);

        // R's C++ fitBeta behavior: if any |beta| > large, mark as non-converged and break
        // This routes the gene to the optim fallback path rather than clamping in-place
        if new_beta.iter().skip(1).any(|&b| b.abs() > MAX_LFC_BETA) {
            converged = false;
            break; // exit IRLS, route to optim fallback
        }

        // Compute deviance for the full Newton step
        let mut new_mus = vec![0.0; n_samples];
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * new_beta[j]).sum();
            let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
            new_mus[i] = (size_factors[i] * eta_clamped.exp()).max(MIN_MU);
        }
        let new_deviance: f64 = (0..n_samples)
            .map(|i| -2.0 * nb_log_likelihood(counts[i], new_mus[i], alpha))
            .sum();

        // R's C++ fitBeta takes the full Newton step without backtracking.
        // Convergence is handled by the iteration limit and the optim fallback.

        let conv_test = (new_deviance - prev_deviance).abs() / (new_deviance.abs() + 0.1);
        prev_deviance = new_deviance;
        beta = new_beta;

        // R's C++ fitBeta has a (t > 0) guard: don't declare convergence on iteration 0
        if _iter > 0 && conv_test < tol {
            converged = true;
            break;
        }
    }

    // L-BFGS-B fallback when IRLS did not converge (matches R DESeq2 optim fallback)
    // R only runs optim() for genes where fitBeta's IRLS didn't converge.
    if !converged {
        // R unconditionally uses optim result when IRLS doesn't converge (fitNbinomGLMs.R:382)
        let optim_result = optim_nb_fallback(counts, size_factors, alpha, design, &beta);
        beta = optim_result.betas;
        converged = true;
    }

    // Compute final mu and log-likelihood using UNCLAMPED mu
    // R DESeq2 computes log-likelihood from raw mu = sf * exp(X * beta)
    // WITHOUT applying minmu clamping. The minmu clamp is only used during
    // IRLS iterations for numerical stability, not for the final LL.
    let mut final_mu = vec![0.0; n_samples];
    let mut ll = 0.0;
    for i in 0..n_samples {
        let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
        let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
        let mu = size_factors[i] * eta_clamped.exp();
        // Store clamped mu for downstream use (SE, hat diagonals)
        final_mu[i] = mu.max(MIN_MU);
        // Use unclamped mu for log-likelihood (matches R)
        ll += nb_log_likelihood(counts[i], mu, alpha);
    }

    GlmGeneResult {
        betas: beta,
        mu: final_mu,
        log_likelihood: ll,
        converged,
    }
}

/// Result from L-BFGS-B optimization fallback.
struct OptimResult {
    betas: Vec<f64>,
    #[allow(dead_code)]
    log_likelihood: f64,
}

/// L-BFGS-B optimizer fallback for NB GLM fitting.
///
/// When IRLS fails to converge, this optimizer minimizes the negative
/// log-posterior of the negative binomial model, matching R DESeq2's
/// `fitNbinomGLMsOptim()` which calls `optim(..., method="L-BFGS-B")`.
///
/// Works in **log2 scale** exactly like R DESeq2:
///   mu = nf * 2^(X %*% p), lambda = 1e-6, bounds = [-30, 30]
///   negLogPost = -(logLike + logPrior)
///
/// `initial_beta` is in **natural log scale** and gets converted to log2 internally.
/// Returns betas converted back to **natural log scale**.
///
/// L-BFGS with m=6 history vectors, Strong Wolfe line search, max 100 iterations.
fn optim_nb_fallback(
    counts: &[f64],
    size_factors: &[f64],
    alpha: f64,
    design: &Array2<f64>,
    initial_beta: &[f64],
) -> OptimResult {
    let n_samples = counts.len();
    let n_coefs = design.ncols();
    let m = 6; // L-BFGS memory size
    let max_iter = 100;
    let grad_tol = 1e-8;
    let f_rel_tol = 1e-12;
    let ln2 = std::f64::consts::LN_2;

    // R: lambda <- rep(1e-6, ncol(modelMatrix))  (log2 scale)
    let lambda: Vec<f64> = vec![1e-6; n_coefs];

    // R: lower=-large, upper=large where large=30 (log2 scale)
    let lower: Vec<f64> = vec![-30.0; n_coefs];
    let upper: Vec<f64> = vec![30.0; n_coefs];

    // Negative log-posterior: -(logLike + logPrior)
    // R: mu_row <- as.numeric(nf * 2^(x %*% p))
    // p is in log2 scale: mu = sf * 2^(X*p) = sf * exp(X*p * ln2)
    let nll = |p: &[f64]| -> f64 {
        let mut val = 0.0;
        for i in 0..n_samples {
            let eta_log2: f64 = (0..n_coefs).map(|j| design[[i, j]] * p[j]).sum();
            let eta_nat = eta_log2 * ln2;
            let eta_nat_clamped = eta_nat.clamp(-MAX_ETA, MAX_ETA);
            let mu = size_factors[i] * eta_nat_clamped.exp();
            val -= nb_log_likelihood(counts[i], mu, alpha);
        }
        // Add prior penalty: 0.5 * sum(lambda_j * p_j^2) (log2 scale)
        for j in 0..n_coefs {
            val += 0.5 * lambda[j] * p[j] * p[j];
        }
        val
    };

    // Numerical gradient using central finite differences (R default ndeps=1e-3)
    let nll_grad = |p: &[f64], grad: &mut [f64]| {
        let h = 1e-3;
        let mut p_work = p.to_vec();
        for j in 0..p.len() {
            let orig = p_work[j];
            p_work[j] = orig + h;
            let f_plus = nll(&p_work);
            p_work[j] = orig - h;
            let f_minus = nll(&p_work);
            p_work[j] = orig;
            grad[j] = (f_plus - f_minus) / (2.0 * h);
        }
    };

    // Project beta onto bounds
    let project = |beta: &mut [f64]| {
        for j in 0..n_coefs {
            if beta[j] < lower[j] {
                beta[j] = lower[j];
            }
            if beta[j] > upper[j] {
                beta[j] = upper[j];
            }
        }
    };

    // Initialize: convert initial_beta from natural log to log2 scale
    let mut beta: Vec<f64> = initial_beta.iter().map(|&b| b / ln2).collect();
    project(&mut beta);

    let mut f_val = nll(&beta);
    let mut grad = vec![0.0; n_coefs];
    nll_grad(&beta, &mut grad);

    // L-BFGS history storage
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    for _iter in 0..max_iter {
        // Check gradient convergence (infinity norm of projected gradient)
        let grad_inf = projected_gradient_inf(&beta, &grad, &lower, &upper);
        if grad_inf < grad_tol {
            break;
        }

        // Compute search direction using L-BFGS two-loop recursion
        let dir = lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist);

        // Negate for descent direction
        let mut d: Vec<f64> = dir.iter().map(|&v| -v).collect();

        // Project the search direction: for bounded variables at their limit,
        // zero out components that would push them further out of bounds
        for j in 0..n_coefs {
            if (beta[j] <= lower[j] && d[j] < 0.0) || (beta[j] >= upper[j] && d[j] > 0.0) {
                d[j] = 0.0;
            }
        }

        // Directional derivative
        let dg: f64 = grad.iter().zip(d.iter()).map(|(&g, &di)| g * di).sum();
        if dg >= 0.0 {
            // Not a descent direction; use steepest descent
            for j in 0..n_coefs {
                d[j] = -grad[j];
            }
            for j in 0..n_coefs {
                if (beta[j] <= lower[j] && d[j] < 0.0) || (beta[j] >= upper[j] && d[j] > 0.0) {
                    d[j] = 0.0;
                }
            }
        }

        // Strong Wolfe line search
        let c1 = 1e-4;
        let c2 = 0.9;
        let dg_init: f64 = grad.iter().zip(d.iter()).map(|(&g, &di)| g * di).sum();

        if dg_init >= 0.0 {
            // No descent possible from this point
            break;
        }

        let mut step = 1.0;
        let mut beta_new = vec![0.0; n_coefs];
        let mut grad_new = vec![0.0; n_coefs];
        let mut f_new;

        // Compute maximum step that stays within bounds
        let mut max_step = f64::INFINITY;
        for j in 0..n_coefs {
            if d[j] > 0.0 && upper[j].is_finite() {
                let s = (upper[j] - beta[j]) / d[j];
                if s < max_step {
                    max_step = s;
                }
            } else if d[j] < 0.0 && lower[j].is_finite() {
                let s = (lower[j] - beta[j]) / d[j];
                if s < max_step {
                    max_step = s;
                }
            }
        }
        if step > max_step {
            step = max_step * 0.99;
        }

        // Backtracking with interpolation (limited iterations)
        let mut ls_ok = false;
        for _ls in 0..40 {
            for j in 0..n_coefs {
                beta_new[j] = beta[j] + step * d[j];
            }
            project(&mut beta_new);

            f_new = nll(&beta_new);
            nll_grad(&beta_new, &mut grad_new);

            let dg_new: f64 = grad_new.iter().zip(d.iter()).map(|(&g, &di)| g * di).sum();

            // Sufficient decrease (Armijo)
            if f_new <= f_val + c1 * step * dg_init {
                // Curvature condition (strong Wolfe)
                if dg_new.abs() <= c2 * dg_init.abs() {
                    ls_ok = true;
                    break;
                }
                // Curvature not satisfied but function decreased enough;
                // if dg_new > 0, we overshot; try shorter step
                if dg_new > 0.0 {
                    step *= 0.5;
                    continue;
                }
                // dg_new < 0 and Armijo satisfied: accept
                ls_ok = true;
                break;
            }
            step *= 0.5;
            if step < 1e-20 {
                break;
            }
        }

        if !ls_ok {
            // Line search failed; try with current best
            for j in 0..n_coefs {
                beta_new[j] = beta[j] + step * d[j];
            }
            project(&mut beta_new);
            f_new = nll(&beta_new);
            nll_grad(&beta_new, &mut grad_new);
            if f_new >= f_val {
                break; // Cannot make progress
            }
        } else {
            f_new = nll(&beta_new);
            nll_grad(&beta_new, &mut grad_new);
        }

        // Check function value convergence
        let f_rel = (f_val - f_new).abs() / (f_new.abs() + 1.0);

        // Update L-BFGS history
        let s_vec: Vec<f64> = beta_new.iter().zip(beta.iter()).map(|(&bn, &bo)| bn - bo).collect();
        let y_vec: Vec<f64> = grad_new.iter().zip(grad.iter()).map(|(&gn, &go)| gn - go).collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(&s, &y)| s * y).sum();

        if sy > 1e-20 {
            // Only update if curvature condition is satisfied
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        beta = beta_new;
        f_val = f_new;
        grad = grad_new;

        if f_rel < f_rel_tol {
            break;
        }
    }

    // Compute pure log-likelihood (without prior) for comparison
    // beta is in log2 scale; mu = sf * 2^(X*p) = sf * exp(X*p * ln2)
    let ll: f64 = (0..n_samples)
        .map(|i| {
            let eta_log2: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let eta_nat = eta_log2 * ln2;
            let eta_nat_clamped = eta_nat.clamp(-MAX_ETA, MAX_ETA);
            let mu = size_factors[i] * eta_nat_clamped.exp();
            nb_log_likelihood(counts[i], mu, alpha)
        })
        .sum();

    // Convert optimized betas from log2 scale back to natural log scale
    let betas_natlog: Vec<f64> = beta.iter().map(|&p| p * ln2).collect();

    OptimResult {
        betas: betas_natlog,
        log_likelihood: ll,
    }
}

/// Compute the infinity norm of the projected gradient.
/// For bounded variables at their limits, the gradient component is zeroed
/// if it would push the variable out of bounds.
fn projected_gradient_inf(
    beta: &[f64],
    grad: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let mut max_val = 0.0f64;
    for j in 0..beta.len() {
        let g = if beta[j] <= lower[j] && grad[j] > 0.0 {
            0.0 // at lower bound, gradient pushes further down: projected to 0
        } else if beta[j] >= upper[j] && grad[j] < 0.0 {
            0.0 // at upper bound, gradient pushes further up: projected to 0
        } else {
            grad[j]
        };
        if g.abs() > max_val {
            max_val = g.abs();
        }
    }
    max_val
}

/// L-BFGS two-loop recursion to compute H * grad (approximate inverse Hessian times gradient).
fn lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let n = grad.len();
    let k = s_hist.len();

    let mut q = grad.to_vec();

    if k == 0 {
        // No history: return gradient (steepest descent)
        return q;
    }

    let mut alpha_vals = vec![0.0; k];

    // First loop (backward)
    for i in (0..k).rev() {
        let a: f64 = rho_hist[i]
            * s_hist[i].iter().zip(q.iter()).map(|(&s, &qi)| s * qi).sum::<f64>();
        alpha_vals[i] = a;
        for j in 0..n {
            q[j] -= a * y_hist[i][j];
        }
    }

    // Initial Hessian approximation: H0 = (s'y / y'y) * I
    let sy: f64 = s_hist[k - 1]
        .iter()
        .zip(y_hist[k - 1].iter())
        .map(|(&s, &y)| s * y)
        .sum();
    let yy: f64 = y_hist[k - 1].iter().map(|&y| y * y).sum();
    let gamma = if yy > 0.0 { sy / yy } else { 1.0 };

    let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

    // Second loop (forward)
    for i in 0..k {
        let b: f64 = rho_hist[i]
            * y_hist[i].iter().zip(r.iter()).map(|(&y, &ri)| y * ri).sum::<f64>();
        for j in 0..n {
            r[j] += s_hist[i][j] * (alpha_vals[i] - b);
        }
    }

    r
}

/// Fit an intercept-only reduced model using the closed-form solution.
///
/// For the intercept-only model (~1), R DESeq2 has an optimization that
/// bypasses IRLS:
///   beta_intercept = ln(mean(normalized_counts))
///   mu_j = size_factor_j * exp(beta_intercept)
///
/// This is exact for the intercept-only NB GLM when the link is log.
/// Compute standard errors from the sandwich estimator.
///
/// SE = sqrt(diag( (X'WX + Lambda)^{-1} X'WX (X'WX + Lambda)^{-1} ))
fn compute_standard_errors(
    design: &Array2<f64>,
    mu: &[f64],
    alpha: f64,
) -> Vec<f64> {
    let n_samples = design.nrows();
    let n_coefs = design.ncols();
    let lambda = ridge_lambda();

    // Build X'WX
    let mut xtwx = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_samples {
        let w = mu[i] / (1.0 + alpha * mu[i]);
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                xtwx[j * n_coefs + k] += w * design[[i, j]] * design[[i, k]];
            }
        }
    }

    // X'WX + Lambda
    let mut xtwx_ridge = xtwx.clone();
    for j in 0..n_coefs {
        xtwx_ridge[j * n_coefs + j] += lambda;
    }

    // Invert (X'WX + Lambda)
    let xtwx_ridge_inv = invert_symmetric_matrix(&xtwx_ridge, n_coefs);

    // Sandwich: (X'WX + Lambda)^{-1} * X'WX * (X'WX + Lambda)^{-1}
    let mut temp = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_coefs {
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                temp[i * n_coefs + j] += xtwx_ridge_inv[i * n_coefs + k] * xtwx[k * n_coefs + j];
            }
        }
    }
    let mut sigma = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_coefs {
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                sigma[i * n_coefs + j] += temp[i * n_coefs + k] * xtwx_ridge_inv[k * n_coefs + j];
            }
        }
    }

    (0..n_coefs)
        .map(|i| if sigma[i * n_coefs + i] > 0.0 { sigma[i * n_coefs + i].sqrt() } else { f64::NAN })
        .collect()
}

/// Perform Likelihood Ratio Test comparing full vs reduced models.
/// R equivalent: nbinomLRT() in core.R
///
/// # Arguments
/// * `dds` - DESeqDataSet with size factors and dispersions estimated
/// * `design_full` - Full model design matrix (n_samples x p_full)
/// * `design_reduced` - Reduced model design matrix (n_samples x p_reduced)
///
/// # Returns
/// `DESeqResults` with LRT statistics, p-values, log2 fold changes from full model
pub fn likelihood_ratio_test(
    dds: &DESeqDataSet,
    design_full: &Array2<f64>,
    design_reduced: &Array2<f64>,
) -> Result<DESeqResults> {
    let counts = dds.counts().counts();
    let size_factors = dds.size_factors().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Size factors required for LRT".to_string(),
    })?;
    let dispersions = dds.dispersions().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Dispersions required for LRT".to_string(),
    })?;

    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();

    // Validate dimensions
    if design_full.nrows() != n_samples || design_reduced.nrows() != n_samples {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!(
                "Design matrix rows ({}/{}) must match number of samples ({})",
                design_full.nrows(),
                design_reduced.nrows(),
                n_samples
            ),
        });
    }

    // Degrees of freedom
    let df_full = design_full.ncols();
    let df_reduced = design_reduced.ncols();
    if df_full <= df_reduced {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: format!(
                "Full model ({} params) must have more parameters than reduced model ({} params)",
                df_full, df_reduced
            ),
        });
    }
    let df = (df_full - df_reduced) as f64;

    // Determine if the reduced model is intercept-only (single column of 1s)
    let is_intercept_only = design_reduced.ncols() == 1 && {
        let col = design_reduced.column(0);
        col.iter().all(|&v| (v - 1.0).abs() < 1e-12)
    };

    // Prepare size factors as a slice
    let sf: Vec<f64> = size_factors.iter().copied().collect();

    // R's nbinomLRT refits both full and reduced models from scratch
    // using fitNbinomGLMs (which includes IRLS + optim fallback).

    // Fit both models for all genes in parallel
    let lrt_results: Vec<LrtGeneResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[i, j]]).collect();
            let alpha = dispersions[i];

            // Check for allZero genes
            let row_sum: f64 = gene_counts.iter().sum();
            if row_sum == 0.0 {
                return LrtGeneResult {
                    base_mean: 0.0,
                    base_var: 0.0,
                    lrt_stat: f64::NAN,
                    log2_fold_change: f64::NAN,
                    lfc_se: f64::NAN,
                    betas_full: vec![f64::NAN; df_full],
                    mu_full: vec![f64::NAN; n_samples],
                };
            }

            // R's nbinomLRT always starts from OLS (cold-start), never warm-starts
            // from previous Wald coefficients. Warm-start causes premature IRLS convergence
            // for genes with extreme dispersion (=10.0 cap), preventing the optim fallback
            // from finding the correct optimum.
            let full_result = fit_nb_glm_gene(
                &gene_counts, &sf, alpha, design_full,
                None,
            );
            let (full_ll, full_betas, full_mu) = (full_result.log_likelihood, full_result.betas, full_result.mu);

            // Fit reduced model
            // For intercept-only models, R DESeq2 uses a closed-form shortcut
            // that bypasses IRLS entirely (fitNbinomGLMs.R lines 104-137):
            //   beta0 = log2(mean(counts / size_factors))
            //   mu_j  = size_factor_j * mean(counts / size_factors)
            //   logLike = sum(dnbinom(k_j, mu=mu_j, size=1/alpha, log=TRUE))
            // No IRLS, no optim fallback. The intercept-only NB MLE for the
            // log-link is the arithmetic mean of normalized counts.
            let reduced_result = if is_intercept_only {
                // mean_norm = mean(k / sf) -- arithmetic mean of normalized counts
                let mean_norm: f64 = gene_counts
                    .iter()
                    .zip(sf.iter())
                    .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
                    .sum::<f64>()
                    / n_samples as f64;

                // beta0 in natural-log scale (R stores log2 internally, equivalent)
                let beta0 = if mean_norm > 0.0 { mean_norm.ln() } else { f64::NEG_INFINITY };

                // mu_j = sf_j * mean_norm (UNCLAMPED, matching R)
                // Log-likelihood uses unclamped mu; stored mu is clamped for downstream use
                let mut ll = 0.0;
                let mut mu_stored = vec![0.0; n_samples];
                for j in 0..n_samples {
                    let mu_j = sf[j] * mean_norm; // unclamped
                    ll += nb_log_likelihood(gene_counts[j], mu_j, alpha);
                    mu_stored[j] = mu_j.max(MIN_MU); // clamped for SE / hat diagonals
                }

                GlmGeneResult {
                    betas: vec![beta0],
                    mu: mu_stored,
                    log_likelihood: ll,
                    converged: true,
                }
            } else {
                fit_nb_glm_gene(&gene_counts, &sf, alpha, design_reduced, None)
            };

            // LRT statistic
            let lrt_stat = 2.0 * (full_ll - reduced_result.log_likelihood);
            // Note: R does not clamp LRT stat to >= 0. Negative values shouldn't
            // happen theoretically but can arise from numerical issues. We clamp
            // to 0 for safety; this is a minor deviation from R behavior.
            let lrt_stat = lrt_stat.max(0.0);

            // Base mean and base var of normalized counts
            let norm_counts: Vec<f64> = gene_counts
                .iter()
                .zip(sf.iter())
                .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
                .collect();
            let base_mean: f64 = norm_counts.iter().sum::<f64>() / n_samples as f64;
            let base_var: f64 = norm_counts.iter()
                .map(|&x| (x - base_mean).powi(2))
                .sum::<f64>() / (n_samples as f64 - 1.0);

            // Log2 fold change: use full model's last non-intercept coefficient
            // Convert from natural log to log2 scale: beta_natlog * log2(e)
            let log2_e = std::f64::consts::LOG2_E;
            let log2_fold_change = if full_betas.len() > 1 {
                // Use the last non-intercept coefficient (the main effect)
                full_betas[full_betas.len() - 1] * log2_e
            } else {
                0.0
            };

            // SE from full model sandwich estimator
            let se_vec = compute_standard_errors(design_full, &full_mu, alpha);
            let lfc_se = if se_vec.len() > 1 {
                se_vec[se_vec.len() - 1] * log2_e
            } else {
                f64::NAN
            };

            LrtGeneResult {
                base_mean,
                base_var,
                lrt_stat,
                log2_fold_change,
                lfc_se,
                betas_full: full_betas,
                mu_full: full_mu,
            }
        })
        .collect();

    // Calculate p-values from chi-squared distribution
    let chi2 = ChiSquared::new(df).map_err(|e| DeseqError::InvalidInput {
        reason: format!("Invalid degrees of freedom {}: {}", df, e),
    })?;

    let pvalues: Vec<f64> = lrt_results
        .iter()
        .map(|r| {
            if r.lrt_stat.is_finite() && r.lrt_stat >= 0.0 {
                chi2.sf(r.lrt_stat)
            } else {
                f64::NAN
            }
        })
        .collect();

    // Apply BH correction
    let padj = benjamini_hochberg(&pvalues);

    // Get dispersion info for output
    let gene_wise_dispersions: Vec<f64> = if let Some(disp) = dds.gene_dispersions() {
        disp.to_vec()
    } else {
        vec![f64::NAN; n_genes]
    };

    let trended_dispersions: Vec<f64> = if let Some(disp) = dds.trended_dispersions() {
        disp.to_vec()
    } else {
        vec![f64::NAN; n_genes]
    };

    let contrast = Contrast {
        variable: dds.design_variable().to_string(),
        numerator: "LRT".to_string(),
        denominator: "reduced".to_string(),
    };

    Ok(DESeqResults {
        gene_ids: dds.counts().gene_ids().to_vec(),
        base_means: lrt_results.iter().map(|r| r.base_mean).collect(),
        base_vars: lrt_results.iter().map(|r| r.base_var).collect(),
        log2_fold_changes: lrt_results.iter().map(|r| r.log2_fold_change).collect(),
        lfc_se: lrt_results.iter().map(|r| r.lfc_se).collect(),
        stat: lrt_results.iter().map(|r| r.lrt_stat).collect(),
        pvalues,
        padj,
        dispersions: dispersions.to_vec(),
        gene_wise_dispersions,
        trended_dispersions,
        contrast,
    })
}

/// Per-gene LRT result (intermediate)
struct LrtGeneResult {
    base_mean: f64,
    base_var: f64,
    lrt_stat: f64,
    log2_fold_change: f64,
    lfc_se: f64,
    #[allow(dead_code)]
    betas_full: Vec<f64>,
    #[allow(dead_code)]
    mu_full: Vec<f64>,
}

// ============================================================================
// Linear algebra helpers (self-contained, no dependency on fitting.rs)
// ============================================================================

/// Weighted least squares with uniform ridge penalty on all coefficients.
fn weighted_least_squares_ridge(
    design: &Array2<f64>,
    weights: &[f64],
    response: &[f64],
    lambda: f64,
) -> Vec<f64> {
    let n_coefs = design.ncols();

    // X'WX
    let mut xtwx = vec![0.0; n_coefs * n_coefs];
    for i in 0..design.nrows() {
        let w = weights[i];
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                xtwx[j * n_coefs + k] += w * design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Ridge penalty
    for j in 0..n_coefs {
        xtwx[j * n_coefs + j] += lambda;
    }

    // X'Wz
    let mut xtwz = vec![0.0; n_coefs];
    for i in 0..design.nrows() {
        let w = weights[i];
        for j in 0..n_coefs {
            xtwz[j] += w * design[[i, j]] * response[i];
        }
    }

    solve_symmetric_system(&xtwx, &xtwz, n_coefs)
}

/// Solve a symmetric positive definite system Ax = b via Cholesky decomposition.
fn solve_symmetric_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    sum = 1e-12; // numerical stabilization
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Forward substitution: Ly = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }

    // Back substitution: L'x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// Invert a symmetric positive definite matrix using Cholesky + column solves.
fn invert_symmetric_matrix(a: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        let col = solve_symmetric_system(a, &e, n);
        for j in 0..n {
            result[j * n + i] = col[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, SampleMetadata};
    use crate::dispersion::{estimate_dispersions, DispersionParams, TrendFitMethod};
    use crate::glm::{fit_glm, GlmFitParams};
    use crate::normalization::{estimate_size_factors, SizeFactorMethod};
    use ndarray::array;

    #[test]
    fn test_nb_log_likelihood_basic() {
        // y=0, mu=0 should return 0
        assert_eq!(nb_log_likelihood(0.0, 0.0, 0.1), 0.0);

        // Finite for normal values
        let ll = nb_log_likelihood(5.0, 5.0, 0.1);
        assert!(ll.is_finite());
        assert!(ll < 0.0);

        // mu=0, y>0 should return -inf
        assert_eq!(nb_log_likelihood(5.0, 0.0, 0.1), f64::NEG_INFINITY);
    }

    #[test]
    fn test_lrt_with_design_matrices() {
        let counts = CountMatrix::new(
            array![
                [100.0, 110.0, 90.0, 400.0, 420.0, 380.0],
                [500.0, 520.0, 480.0, 500.0, 510.0, 490.0],
                [200.0, 210.0, 190.0, 50.0, 55.0, 45.0]
            ],
            vec![
                "gene_up".to_string(),
                "gene_nc".to_string(),
                "gene_down".to_string(),
            ],
            vec![
                "s1".to_string(),
                "s2".to_string(),
                "s3".to_string(),
                "s4".to_string(),
                "s5".to_string(),
                "s6".to_string(),
            ],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
            "s5".to_string(),
            "s6".to_string(),
        ]);
        metadata
            .add_condition(
                "treatment",
                vec![
                    "control".to_string(),
                    "control".to_string(),
                    "control".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                ],
            )
            .unwrap();

        let mut dds = DESeqDataSet::new(counts, metadata, "treatment").unwrap();
        estimate_size_factors(&mut dds, SizeFactorMethod::Ratio).unwrap();
        estimate_dispersions(&mut dds, TrendFitMethod::Mean, &DispersionParams::default()).unwrap();
        fit_glm(&mut dds, &GlmFitParams::default()).unwrap();

        // Full design: intercept + treatment effect
        let design_full = dds.design_matrix().unwrap().clone();

        // Reduced design: intercept only
        let n_samples = dds.n_samples();
        let design_reduced = Array2::from_elem((n_samples, 1), 1.0);

        let results =
            likelihood_ratio_test(&dds, &design_full, &design_reduced).unwrap();

        // All LRT statistics should be non-negative (for non-allZero genes)
        for stat in &results.stat {
            if stat.is_finite() {
                assert!(*stat >= 0.0, "LRT stat should be non-negative, got {}", stat);
            }
        }

        // P-values should be in [0, 1]
        for p in &results.pvalues {
            if p.is_finite() {
                assert!(*p >= 0.0 && *p <= 1.0);
            }
        }

        // gene_up should have positive LFC
        assert!(
            results.log2_fold_changes[0] > 0.5,
            "gene_up LFC should be positive"
        );

        // gene_down should have negative LFC
        assert!(
            results.log2_fold_changes[2] < -0.5,
            "gene_down LFC should be negative"
        );
    }

    #[test]
    fn test_allzero_gene() {
        let counts = CountMatrix::new(
            array![
                [0.0, 0.0, 0.0, 0.0],
                [100.0, 110.0, 200.0, 210.0],
            ],
            vec!["zero_gene".to_string(), "normal_gene".to_string()],
            vec![
                "s1".to_string(),
                "s2".to_string(),
                "s3".to_string(),
                "s4".to_string(),
            ],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ]);
        metadata
            .add_condition(
                "group",
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                ],
            )
            .unwrap();

        let mut dds = DESeqDataSet::new(counts, metadata, "group").unwrap();
        estimate_size_factors(&mut dds, SizeFactorMethod::Ratio).unwrap();
        estimate_dispersions(&mut dds, TrendFitMethod::Mean, &DispersionParams::default()).unwrap();
        fit_glm(&mut dds, &GlmFitParams::default()).unwrap();

        let design_full = dds.design_matrix().unwrap().clone();
        let design_reduced = Array2::from_elem((4, 1), 1.0);

        let results =
            likelihood_ratio_test(&dds, &design_full, &design_reduced).unwrap();

        // allZero gene should have NaN stat and pvalue
        assert!(results.stat[0].is_nan());
        assert!(results.pvalues[0].is_nan());

        // Normal gene should have finite values
        assert!(results.stat[1].is_finite());
        assert!(results.pvalues[1].is_finite());
    }
}
