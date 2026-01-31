//! GLM fitting using Iteratively Reweighted Least Squares (IRLS)

use ndarray::{Array1, Array2, Array3, ArrayView1};
use rayon::prelude::*;

use super::design::{create_design_matrix, DesignInfo};
use super::negative_binomial::{nb_mean, nb_weight, MAX_ETA, MAX_LFC_BETA, MIN_MU};
use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};
use statrs::function::gamma::ln_gamma;

/// Configurable parameters for GLM fitting.
/// R equivalent: parameters of nbinomWaldTest() in core.R
#[derive(Debug, Clone)]
pub struct GlmFitParams {
    /// Maximum IRLS iterations. R: nbinomWaldTest(maxit=100)
    pub maxit: usize,
    /// Beta convergence tolerance. R: nbinomWaldTest(betaTol=1e-8)
    pub beta_tol: f64,
}

impl Default for GlmFitParams {
    fn default() -> Self {
        Self {
            maxit: 100,
            beta_tol: 1e-8,
        }
    }
}

/// Calculate NB log-likelihood matching R's dnbinom_mu(x, size, mu, log=TRUE)
/// size = 1/alpha (dispersion), mu = mean
fn nb_log_likelihood_for_deviance(y: f64, mu: f64, size: f64) -> f64 {
    if mu <= 0.0 || size <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // R's dnbinom_mu uses: prob = size / (size + mu)
    let prob = size / (size + mu);

    // log P(Y = y | size, prob) = lgamma(y + size) - lgamma(size) - lgamma(y + 1)
    //                            + size * log(prob) + y * log(1 - prob)
    ln_gamma(y + size) - ln_gamma(size) - ln_gamma(y + 1.0)
        + size * prob.ln()
        + y * (1.0 - prob).ln()
}

/// Fit GLM to all genes in the dataset
/// Supports both simple two-group designs and complex multi-factor designs
///
/// NOTE: R DESeq2 supports `useQR=TRUE` for numerically stable GLM fitting via
/// QR decomposition of the design matrix. This is not implemented here; the current
/// direct solve approach (Cholesky on X'WX) works for well-conditioned design matrices
/// but may be less stable for rank-deficient or near-collinear designs.
pub fn fit_glm(dds: &mut DESeqDataSet, params: &GlmFitParams) -> Result<DesignInfo> {
    // Check prerequisites
    if !dds.has_size_factors() {
        return Err(DeseqError::GLMConvergenceFailed {
            gene_id: "N/A".to_string(),
            reason: "Size factors must be estimated first".to_string(),
        });
    }

    if !dds.has_dispersions() {
        return Err(DeseqError::GLMConvergenceFailed {
            gene_id: "N/A".to_string(),
            reason: "Dispersions must be estimated first".to_string(),
        });
    }

    // Create design matrix - use extended design if multi-factor model or custom reference levels
    let has_custom_ref = !dds.reference_levels().is_empty();
    let (design, info) = if dds.has_batch_effect() || dds.has_interactions() || dds.has_continuous() || has_custom_ref {
        // Use extended design matrix for complex models or when reference levels are specified
        super::design::create_extended_design_matrix(dds)?
    } else {
        // Use simple design matrix for two-group comparison (uses alphabetical reference)
        create_design_matrix(dds.sample_metadata(), dds.design_variable())?
    };

    // Validate design matrix rank (DESeq2: stopifnot(all(colSums(abs(modelMatrix)) > 0)))
    for j in 0..design.ncols() {
        let col_sum: f64 = (0..design.nrows()).map(|i| design[[i, j]].abs()).sum();
        if col_sum == 0.0 {
            return Err(DeseqError::InvalidDesignMatrix {
                reason: format!("Design matrix column {} is all zeros", j),
            });
        }
    }

    let n_coefs = design.ncols();

    // Get data
    let counts = dds.counts().counts();
    let size_factors = dds.size_factors().unwrap();
    let dispersions = dds.dispersions().unwrap();
    let n_genes = dds.n_genes();

    // Fit GLM for each gene in parallel
    let results: Vec<GlmFitResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts = counts.row(i);
            let alpha = dispersions[i];
            fit_single_gene(gene_counts, &design, size_factors.view(), alpha, params)
        })
        .collect();

    // Extract coefficients, standard errors, hat diagonals, mu, and log_likelihood
    let n_samples = dds.n_samples();
    let mut coefficients = Array2::zeros((n_genes, n_coefs));
    let mut standard_errors = Array2::zeros((n_genes, n_coefs));
    let mut covariances = Array3::zeros((n_genes, n_coefs, n_coefs));
    let mut hat_diagonals = Array2::zeros((n_genes, n_samples));
    let mut mu = Array2::zeros((n_genes, n_samples));
    let mut converged = Vec::with_capacity(n_genes);
    let mut deviance = Array1::zeros(n_genes);

    for (i, result) in results.into_iter().enumerate() {
        for j in 0..n_coefs {
            coefficients[[i, j]] = result.coefficients[j];
            standard_errors[[i, j]] = result.standard_errors[j];
            for k in 0..n_coefs {
                covariances[[i, j, k]] = result.covariance[j * n_coefs + k];
            }
        }
        for j in 0..n_samples {
            hat_diagonals[[i, j]] = result.hat_diagonals[j];
            mu[[i, j]] = result.mu[j];
        }
        converged.push(result.converged);
        // Compute deviance from log_likelihood: deviance = -2 * logLike
        deviance[i] = -2.0 * result.log_likelihood;
    }

    dds.set_design_matrix(design)?;
    dds.set_coefficients(coefficients)?;
    dds.set_standard_errors(standard_errors)?;
    dds.set_covariances(covariances)?;
    dds.set_hat_diagonals(hat_diagonals)?;
    dds.set_mu(mu)?;
    dds.set_converged(converged);
    dds.set_deviance(deviance)?;

    dds.set_design_column_names(info.coef_names.clone());

    Ok(info)
}

pub struct GlmFitResult {
    pub coefficients: Vec<f64>,
    pub standard_errors: Vec<f64>,
    /// Covariance matrix stored as flat row-major array: element (i,j) = covariance[i * n_coefs + j]
    pub covariance: Vec<f64>,
    /// Number of coefficients (dimension of covariance matrix)
    pub n_coefs: usize,
    pub converged: bool,
    pub hat_diagonals: Vec<f64>,
    pub mu: Vec<f64>,
    pub log_likelihood: f64,
}

pub fn fit_single_gene(
    counts: ArrayView1<f64>,
    design: &Array2<f64>,
    size_factors: ArrayView1<f64>,
    alpha: f64,
    params: &GlmFitParams,
) -> GlmFitResult {
    let n_samples = counts.len();
    let n_coefs = design.ncols();
    let max_iter = params.maxit;
    let tol = params.beta_tol;

    // Initialize coefficients using OLS on log counts
    // R C++ fitBeta: y_hat(j) = std::log(k(i,j) / nf(i,j) + 0.1);
    // R fitNbinomGLMs.R: y <- t(log(counts(object,normalized=TRUE) + .1))
    let log_counts: Vec<f64> = counts
        .iter()
        .zip(size_factors.iter())
        .map(|(&c, &s)| {
            let norm_ct = if s > 0.0 { c / s } else { 0.0 };
            (norm_ct + 0.1).ln()
        })
        .collect();

    // Solve OLS: beta = (X'X)^-1 * X'y
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
    let mut beta = solve_symmetric_system(&xtx, &xty, n_coefs);

    if beta.iter().any(|&b| !b.is_finite()) {
        let mean_count: f64 = counts.iter().zip(size_factors.iter())
            .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
            .sum::<f64>() / n_samples as f64;
        beta = vec![0.0; n_coefs];
        beta[0] = (mean_count.max(0.1)).ln();
    }

    // Save OLS initial betas for optim fallback
    // R: beta_mat[row,] is used as optim initial when IRLS betas are extreme
    let ols_initial_beta = beta.clone();

    let mut converged = false;

    // R C++: dev = 0.0; dev_old = 0.0;
    let mut dev_old = 0.0f64;

    // Pre-allocate IRLS buffers outside loop to avoid repeated allocation
    let mut mus = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];
    let mut working_response = vec![0.0; n_samples];

    for _iter in 0..max_iter {
        // Compute mus, weights, working response from current beta
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let mu_raw = nb_mean(eta, size_factors[i]);
            let mu_clamped = mu_raw.max(MIN_MU);
            mus[i] = mu_clamped;
            weights[i] = nb_weight(mu_clamped, alpha);
            working_response[i] = (mu_clamped / size_factors[i]).ln() + (counts[i] - mu_clamped) / mu_clamped;
        }

        // WLS: beta_hat = (X'WX + ridge)^-1 X'Wz
        // R C++: solve(beta_hat, x.t() * (x.each_col() % w_vec) + ridge, x.t() * (z % w_vec))
        beta = weighted_least_squares_ridge(design, &weights, &working_response);

        // R C++: if (sum(abs(beta_hat) > large) > 0) { iter(i) = maxit; break; }
        if beta.iter().any(|&b| b.abs() > MAX_LFC_BETA) {
            break;
        }

        // Compute mu from updated beta
        // R C++: mu_hat = nfrow % exp(x * beta_hat); mu_hat(j) = fmax(mu_hat(j), minmu)
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            mus[i] = nb_mean(eta, size_factors[i]).max(MIN_MU);
        }

        // Compute deviance
        // R C++: dev = sum(-2.0 * Rf_dnbinom_mu(yrow(j), 1.0/alpha_hat(i), mu_hat(j), 1))
        let dev: f64 = mus.iter().zip(counts.iter())
            .map(|(&mu, &y)| {
                let r = 1.0 / alpha;
                -2.0 * nb_log_likelihood_for_deviance(y, mu, r)
            })
            .sum();

        // Convergence test
        // R C++: conv_test = fabs(dev - dev_old)/(fabs(dev) + 0.1)
        let conv_test = (dev - dev_old).abs() / (dev.abs() + 0.1);

        // R C++: if (std::isnan(conv_test)) { iter(i) = maxit; break; }
        if conv_test.is_nan() {
            break;
        }

        // R C++: if ((t > 0) & (conv_test < tol)) { break; }
        if _iter > 0 && conv_test < tol {
            converged = true;
            break;
        }

        // R C++: dev_old = dev;
        dev_old = dev;
    }

    // L-BFGS-B fallback when IRLS did not converge (matches R DESeq2 optim fallback)
    // R: rowsForOptim <- which(!betaConv | !rowStable | !rowVarPositive)
    // R always stores the optim result: betaMatrix[row,] <- o$par
    // R sets betaConv[row] <- TRUE if o$convergence == 0

    // R: rowStable = all betas are finite
    let row_stable = beta.iter().all(|b| b.is_finite());

    // R: rowVarPositive = all diagonal elements of W are positive
    // W_ii = mu_i / (1 + alpha * mu_i)
    let row_var_positive = {
        let mut all_pos = true;
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let mu = nb_mean(eta, size_factors[i]).max(MIN_MU);
            let w = nb_weight(mu, alpha);
            if w <= 0.0 || !w.is_finite() {
                all_pos = false;
                break;
            }
        }
        all_pos
    };

    let needs_optim = !converged || !row_stable || !row_var_positive;

    let log_likelihood = if needs_optim {
        let ln2 = std::f64::consts::LN_2;
        // R: betaRow <- if (rowStable[row] & all(abs(betaMatrix[row,]) < large)) {
        //        betaMatrix[row,]     # IRLS result in LOG2 scale
        //    } else {
        //        beta_mat[row,]       # OLS initial in NATURAL LOG, treated as log2!
        //    }
        // R checks betaMatrix (log2 scale) against large=30
        let beta_log2: Vec<f64> = beta.iter().map(|&b| b / ln2).collect();
        let betas_within_bounds = beta_log2.iter().all(|&b| b.abs() < 30.0);

        let initial_for_optim = if row_stable && betas_within_bounds {
            beta_log2  // IRLS result converted to log2 (matches R: betaMatrix[row,])
        } else {
            ols_initial_beta.clone()  // OLS in NATURAL LOG, treated as log2 (matches R: beta_mat[row,])
        };

        let counts_slice: Vec<f64> = counts.to_vec();
        let sf_slice: Vec<f64> = size_factors.to_vec();
        // R: lambda <- rep(1e-6, ncol(modelMatrix))  (log2 scale)
        let lambda_log2: Vec<f64> = vec![1e-6; design.ncols()];
        let optim_result =
            optim_nb_fallback_fitting(&counts_slice, &sf_slice, alpha, design, &initial_for_optim, &lambda_log2);
        // Result is already in natural log (converted inside optim function)
        beta = optim_result.betas;
        converged = optim_result.optim_converged;
        let lbfgsb_ll = optim_result.log_likelihood;

        // If L-BFGS-B also didn't converge, try coordinate-wise Newton-Raphson
        // as a secondary fallback. Use whichever result has better log-likelihood.
        if !converged {
            let coord_result = fit_single_gene_optim(
                counts, design, size_factors, alpha, Some(&beta),
            );
            if coord_result.converged && coord_result.log_likelihood > lbfgsb_ll {
                return coord_result;
            }
        }

        lbfgsb_ll
    } else {
        // IRLS converged, compute log_likelihood from final mu values
        let size = 1.0 / alpha;
        (0..n_samples)
            .map(|i| {
                let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
                let mu = nb_mean(eta, size_factors[i]).max(MIN_MU);
                nb_log_likelihood_for_deviance(counts[i], mu, size)
            })
            .sum()
    };

    let mut mus = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];
    for i in 0..n_samples {
        let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
        let mu = nb_mean(eta, size_factors[i]).max(MIN_MU);
        mus[i] = mu;
        weights[i] = nb_weight(mu, alpha);
    }

    let (standard_errors, hat_diagonals, covariance) = calculate_standard_errors_and_hat(design, &weights);

    GlmFitResult {
        coefficients: beta,
        standard_errors,
        covariance,
        n_coefs,
        converged,
        hat_diagonals,
        mu: mus,
        log_likelihood,
    }
}

/// R equivalent: fitNbinomGLMsOptim() — coordinate-wise Newton-Raphson fallback.
///
/// When both IRLS and the L-BFGS-B optimizer fail to converge, this function
/// provides a robust coordinate-wise optimization of the NB log-likelihood.
/// Each coefficient is optimized individually while holding others fixed,
/// using a Newton step with Armijo line search to guarantee monotone improvement.
///
/// NB log-likelihood:
///   ll = sum_j [ y_j * ln(mu_j) - (y_j + 1/alpha) * ln(mu_j + 1/alpha) ] + const
///   where mu_j = size_factors[j] * exp(X[j,:] . beta)
///
/// Gradient w.r.t. beta_k:
///   d_ll/d_beta_k = sum_j [ (y_j - mu_j) / (1 + alpha * mu_j) * mu_j * X[j,k] ]
///
/// Hessian diagonal:
///   d2_ll/d_beta_k^2 = -sum_j [ w_j * mu_j^2 * X[j,k]^2 ]
///   where w_j = 1 / (mu_j + alpha * mu_j^2)
fn fit_single_gene_optim(
    counts: ArrayView1<f64>,
    design: &Array2<f64>,
    size_factors: ArrayView1<f64>,
    alpha: f64,
    initial_beta: Option<&[f64]>,
) -> GlmFitResult {
    let n_samples = counts.len();
    let n_coefs = design.ncols();
    let max_iter = 5000; // R default for optim fallback
    let tol = 1e-8;

    // Initialize beta from provided initial values or zeros
    let mut beta: Vec<f64> = match initial_beta {
        Some(b) => b.to_vec(),
        None => {
            let mut b = vec![0.0; n_coefs];
            let mean_count: f64 = counts
                .iter()
                .zip(size_factors.iter())
                .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
                .sum::<f64>()
                / n_samples as f64;
            b[0] = (mean_count.max(0.1)).ln();
            b
        }
    };

    // Compute NB log-likelihood for current beta
    let compute_ll = |beta: &[f64]| -> f64 {
        let size = 1.0 / alpha;
        let mut ll = 0.0;
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
            let mu = (size_factors[i] * eta_clamped.exp()).max(MIN_MU);
            ll += nb_log_likelihood_for_deviance(counts[i], mu, size);
        }
        ll
    };

    let mut ll_old = compute_ll(&beta);
    let mut converged = false;

    for _outer in 0..max_iter {
        let ll_start = ll_old;

        // Coordinate-wise optimization: optimize each beta[j] individually
        for j in 0..n_coefs {
            // Compute gradient and Hessian diagonal for coefficient j
            let mut grad_j = 0.0;
            let mut hess_jj = 0.0;

            for i in 0..n_samples {
                let eta: f64 = (0..n_coefs).map(|k| design[[i, k]] * beta[k]).sum();
                let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
                let mu = (size_factors[i] * eta_clamped.exp()).max(MIN_MU);

                // Gradient: (y - mu) / (1 + alpha * mu) * mu * X[i,j]
                // which simplifies to (y - mu) * w * mu * X[i,j] where w = 1/(mu + alpha*mu^2)
                // but more directly: (y - mu) / (1 + alpha * mu) * X[i,j]
                // since mu / (1 + alpha * mu) = mu * w_irls
                let resid_scaled = (counts[i] - mu) / (1.0 + alpha * mu);
                grad_j += resid_scaled * design[[i, j]];

                // Hessian diagonal: -sum_i [ mu_i / (1 + alpha * mu_i) * X[i,j]^2 ]
                let w = mu / (1.0 + alpha * mu);
                hess_jj -= w * design[[i, j]] * design[[i, j]];
            }

            // Newton step: delta = -grad / hess
            // Since hess_jj is negative (concave), -grad/hess = grad/|hess|
            if hess_jj.abs() < 1e-20 {
                continue; // Skip if Hessian is essentially zero
            }

            let delta = -grad_j / hess_jj;

            // Skip tiny updates
            if delta.abs() < 1e-14 {
                continue;
            }

            // Armijo line search: find step size that improves log-likelihood
            let armijo_c = 1e-4;
            let mut step = 1.0;
            let beta_j_old = beta[j];
            let mut improved = false;

            for _ls in 0..30 {
                beta[j] = beta_j_old + step * delta;

                // Clamp beta to prevent extreme values
                beta[j] = beta[j].clamp(-MAX_LFC_BETA, MAX_LFC_BETA);

                let ll_new = compute_ll(&beta);

                // Armijo condition: f(x + step*d) >= f(x) + c * step * grad * d
                // (we are maximizing, so check improvement)
                if ll_new >= ll_old + armijo_c * step * grad_j * delta {
                    ll_old = ll_new;
                    improved = true;
                    break;
                }

                step *= 0.5;
                if step < 1e-20 {
                    break;
                }
            }

            if !improved {
                beta[j] = beta_j_old; // Revert if no improvement found
            }
        }

        // Check convergence: relative change in log-likelihood
        let ll_change = (ll_old - ll_start).abs() / (ll_old.abs() + 0.1);
        if ll_change < tol {
            converged = true;
            break;
        }
    }

    // Compute final mu and weights for SE/hat calculation
    let mut mus = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];
    for i in 0..n_samples {
        let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
        let mu = nb_mean(eta, size_factors[i]).max(MIN_MU);
        mus[i] = mu;
        weights[i] = nb_weight(mu, alpha);
    }

    let (standard_errors, hat_diagonals, covariance) =
        calculate_standard_errors_and_hat(design, &weights);

    let log_likelihood = compute_ll(&beta);

    GlmFitResult {
        coefficients: beta,
        standard_errors,
        covariance,
        n_coefs,
        converged,
        hat_diagonals,
        mu: mus,
        log_likelihood,
    }
}

fn weighted_least_squares_ridge(
    design: &Array2<f64>,
    weights: &[f64],
    response: &[f64],
) -> Vec<f64> {
    let n_coefs = design.ncols();
    let mut xtwx = vec![0.0; n_coefs * n_coefs];
    for i in 0..design.nrows() {
        let w = weights[i];
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                xtwx[j * n_coefs + k] += w * design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Use R's lambda scale
    let ln2 = std::f64::consts::LN_2;
    let lambda = 1e-6 / (ln2 * ln2);
    for j in 0..n_coefs {
        xtwx[j * n_coefs + j] += lambda;
    }

    let mut xtwz = vec![0.0; n_coefs];
    for i in 0..design.nrows() {
        let w = weights[i];
        for j in 0..n_coefs {
            xtwz[j] += w * design[[i, j]] * response[i];
        }
    }

    solve_symmetric_system(&xtwx, &xtwz, n_coefs)
}

fn solve_symmetric_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                // If the matrix is not positive definite, add a small epsilon to the diagonal
                // to maintain numerical stability, matching DESeq2's robust approach.
                if sum <= 0.0 {
                    sum = 1e-12;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i { sum -= l[i * n + j] * y[j]; }
        y[i] = sum / l[i * n + i];
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n { sum -= l[j * n + i] * x[j]; }
        x[i] = sum / l[i * n + i];
    }
    x
}

fn calculate_standard_errors_and_hat(design: &Array2<f64>, weights: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_samples = design.nrows();
    let n_coefs = design.ncols();

    let mut xtwx = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_samples {
        let w = weights[i];
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                xtwx[j * n_coefs + k] += w * design[[i, j]] * design[[i, k]];
            }
        }
    }

    let ln2 = std::f64::consts::LN_2;
    let lambda = 1e-6 / (ln2 * ln2);
    let mut xtwx_ridge = xtwx.clone();
    for j in 0..n_coefs {
        xtwx_ridge[j * n_coefs + j] += lambda;
    }

    let xtwx_ridge_inv = invert_symmetric_matrix(&xtwx_ridge, n_coefs);

    let mut hat_diagonals = vec![0.0; n_samples];
    for i in 0..n_samples {
        let w = weights[i];
        let mut h = 0.0;
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                h += w * design[[i, j]] * xtwx_ridge_inv[j * n_coefs + k] * design[[i, k]];
            }
        }
        hat_diagonals[i] = h;
    }

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

    let standard_errors = (0..n_coefs)
        .map(|i| if sigma[i * n_coefs + i] > 0.0 { sigma[i * n_coefs + i].sqrt() } else { f64::NAN })
        .collect();

    (standard_errors, hat_diagonals, sigma)
}

fn invert_symmetric_matrix(a: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        let col = solve_symmetric_system(a, &e, n);
        for j in 0..n { result[j * n + i] = col[j]; }
    }
    result
}

// ==========================================================================
// L-BFGS-B optimizer fallback for IRLS non-convergence
// ==========================================================================

/// Result from L-BFGS-B optimization fallback.
struct OptimFallbackResult {
    betas: Vec<f64>,
    log_likelihood: f64,
    /// R: if (o$convergence == 0) { betaConv[row] <- TRUE }
    optim_converged: bool,
}

/// L-BFGS-B optimizer fallback for NB GLM fitting.
///
/// Matches R DESeq2's `fitNbinomGLMsOptim()` which minimizes:
///   negLogPost = -(logLike + logPrior)
/// where logPrior = sum(dnorm(p, 0, sqrt(1/lambda), log=TRUE))
///
/// Works in **LOG2 scale** exactly like R DESeq2:
///   mu = nf * 2^(X %*% p), lambda = 1e-6, bounds = [-30, 30]
///
/// `initial_beta` is expected in **LOG2 scale**.
/// Returns betas converted back to **natural log scale**.
fn optim_nb_fallback_fitting(
    counts: &[f64],
    size_factors: &[f64],
    alpha: f64,
    design: &Array2<f64>,
    initial_beta: &[f64],
    lambda_log2: &[f64],
) -> OptimFallbackResult {
    let n_samples = counts.len();
    let n_coefs = design.ncols();
    let m = 6; // L-BFGS memory size
    let max_iter = 100;
    let grad_tol = 1e-8;
    let f_rel_tol = 1e-12;

    // R: lambda passed from caller (log2 scale)
    let ln2 = std::f64::consts::LN_2;
    let lambda: Vec<f64> = lambda_log2.to_vec();

    // R: lower=-large, upper=large where large=30 (log2 scale)
    let lower: Vec<f64> = vec![-30.0; n_coefs];
    let upper: Vec<f64> = vec![30.0; n_coefs];

    // Negative log-posterior: -(logLike + logPrior)
    // R: objectiveFn <- function(p) {
    //      mu_row <- as.numeric(nf * 2^(x %*% p))
    //      logLike <- sum(dnbinom(k, mu=mu_row, size=1/alpha, log=TRUE))
    //      logPrior <- sum(dnorm(p, 0, sqrt(1/lambda), log=TRUE))
    //      negLogPost <- -1 * (logLike + logPrior)
    //    }
    // p is in log2 scale: mu = sf * 2^(X*p) = sf * exp(X*p * ln2)
    let nll = |p: &[f64]| -> f64 {
        let mut val = 0.0;
        for i in 0..n_samples {
            let eta_log2: f64 = (0..n_coefs).map(|j| design[[i, j]] * p[j]).sum();
            let eta_nat = eta_log2 * ln2;
            let eta_nat_clamped = eta_nat.clamp(-MAX_ETA, MAX_ETA);
            let mu = size_factors[i] * eta_nat_clamped.exp();
            let size = 1.0 / alpha;
            val -= nb_log_likelihood_for_deviance(counts[i], mu, size);
        }
        // Add prior penalty: 0.5 * sum(lambda_j * p_j^2) (log2 scale)
        for j in 0..n_coefs {
            val += 0.5 * lambda[j] * p[j] * p[j];
        }
        if val.is_finite() { val } else { 1e300 }
    };

    // Numerical gradient of negative log-posterior
    // R's optim() with method="L-BFGS-B" uses central finite differences (ndeps=1e-3)
    // when no gradient function is provided. We match this behavior exactly.
    let nll_grad = |p: &[f64], grad: &mut [f64]| {
        let h = 1e-3; // R default ndeps
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

    let mut beta = initial_beta.to_vec();
    project(&mut beta);

    let mut f_val = nll(&beta);
    let mut grad = vec![0.0; n_coefs];
    nll_grad(&beta, &mut grad);

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    // Track convergence: R checks o$convergence == 0
    let mut optim_converged = false;

    for _iter in 0..max_iter {
        let grad_inf = fitting_projected_gradient_inf(&beta, &grad, &lower, &upper);
        if grad_inf < grad_tol {
            optim_converged = true;
            break;
        }

        let dir = fitting_lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist);
        let mut d: Vec<f64> = dir.iter().map(|&v| -v).collect();

        for j in 0..n_coefs {
            if (beta[j] <= lower[j] && d[j] < 0.0) || (beta[j] >= upper[j] && d[j] > 0.0) {
                d[j] = 0.0;
            }
        }

        let dg: f64 = grad.iter().zip(d.iter()).map(|(&g, &di)| g * di).sum();
        if dg >= 0.0 {
            for j in 0..n_coefs {
                d[j] = -grad[j];
            }
            for j in 0..n_coefs {
                if (beta[j] <= lower[j] && d[j] < 0.0) || (beta[j] >= upper[j] && d[j] > 0.0) {
                    d[j] = 0.0;
                }
            }
        }

        let c1 = 1e-4;
        let c2 = 0.9;
        let dg_init: f64 = grad.iter().zip(d.iter()).map(|(&g, &di)| g * di).sum();

        if dg_init >= 0.0 {
            break;
        }

        let mut step = 1.0;
        let mut beta_new = vec![0.0; n_coefs];
        let mut grad_new = vec![0.0; n_coefs];
        let mut f_new;

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

        let mut ls_ok = false;
        for _ls in 0..40 {
            for j in 0..n_coefs {
                beta_new[j] = beta[j] + step * d[j];
            }
            project(&mut beta_new);

            f_new = nll(&beta_new);
            nll_grad(&beta_new, &mut grad_new);

            let dg_new: f64 = grad_new.iter().zip(d.iter()).map(|(&g, &di)| g * di).sum();

            if f_new <= f_val + c1 * step * dg_init {
                if dg_new.abs() <= c2 * dg_init.abs() {
                    ls_ok = true;
                    break;
                }
                if dg_new > 0.0 {
                    step *= 0.5;
                    continue;
                }
                ls_ok = true;
                break;
            }
            step *= 0.5;
            if step < 1e-20 {
                break;
            }
        }

        if !ls_ok {
            for j in 0..n_coefs {
                beta_new[j] = beta[j] + step * d[j];
            }
            project(&mut beta_new);
            f_new = nll(&beta_new);
            nll_grad(&beta_new, &mut grad_new);
            if f_new >= f_val {
                break;
            }
        } else {
            f_new = nll(&beta_new);
            nll_grad(&beta_new, &mut grad_new);
        }

        let f_rel = (f_val - f_new).abs() / (f_new.abs() + 1.0);

        let s_vec: Vec<f64> = beta_new
            .iter()
            .zip(beta.iter())
            .map(|(&bn, &bo)| bn - bo)
            .collect();
        let y_vec: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(&gn, &go)| gn - go)
            .collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(&s, &y)| s * y).sum();

        if sy > 1e-20 {
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
            optim_converged = true;
            break;
        }
    }

    // Compute pure log-likelihood (without prior) for comparison
    // R: logLike[row] <- sum(dnbinom(k, mu=mu_row, size=1/alpha, log=TRUE))
    // beta (p) is still in log2 scale here; mu = sf * 2^(X*p) = sf * exp(X*p * ln2)
    let ll: f64 = (0..n_samples)
        .map(|i| {
            let eta_log2: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let eta_nat = eta_log2 * ln2;
            let eta_nat_clamped = eta_nat.clamp(-MAX_ETA, MAX_ETA);
            let mu = size_factors[i] * eta_nat_clamped.exp();
            let size = 1.0 / alpha;
            nb_log_likelihood_for_deviance(counts[i], mu, size)
        })
        .sum();

    // Convert optimized betas from log2 scale back to natural log scale
    // R: betaMatrix[row,] = log2(exp(1)) * betaRes$beta_mat  means natlog = log2_val * ln2
    let betas_natlog: Vec<f64> = beta.iter().map(|&p| p * ln2).collect();

    OptimFallbackResult {
        betas: betas_natlog,
        log_likelihood: ll,
        optim_converged,
    }
}

fn fitting_projected_gradient_inf(
    beta: &[f64],
    grad: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let mut max_val = 0.0f64;
    for j in 0..beta.len() {
        let g = if beta[j] <= lower[j] && grad[j] > 0.0 {
            0.0
        } else if beta[j] >= upper[j] && grad[j] < 0.0 {
            0.0
        } else {
            grad[j]
        };
        if g.abs() > max_val {
            max_val = g.abs();
        }
    }
    max_val
}

fn fitting_lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let n = grad.len();
    let k = s_hist.len();

    let mut q = grad.to_vec();

    if k == 0 {
        return q;
    }

    let mut alpha_vals = vec![0.0; k];

    for i in (0..k).rev() {
        let a: f64 = rho_hist[i]
            * s_hist[i]
                .iter()
                .zip(q.iter())
                .map(|(&s, &qi)| s * qi)
                .sum::<f64>();
        alpha_vals[i] = a;
        for j in 0..n {
            q[j] -= a * y_hist[i][j];
        }
    }

    let sy: f64 = s_hist[k - 1]
        .iter()
        .zip(y_hist[k - 1].iter())
        .map(|(&s, &y)| s * y)
        .sum();
    let yy: f64 = y_hist[k - 1].iter().map(|&y| y * y).sum();
    let gamma = if yy > 0.0 { sy / yy } else { 1.0 };

    let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

    for i in 0..k {
        let b: f64 = rho_hist[i]
            * y_hist[i]
                .iter()
                .zip(r.iter())
                .map(|(&y, &ri)| y * ri)
                .sum::<f64>();
        for j in 0..n {
            r[j] += s_hist[i][j] * (alpha_vals[i] - b);
        }
    }

    r
}

// ==========================================================================
// GLM re-fit with custom per-coefficient lambda (for LFC shrinkage)
// ==========================================================================

/// Re-fit GLM for all genes with a custom per-coefficient lambda vector.
/// Used for LFC shrinkage (betaPrior=TRUE in R DESeq2).
///
/// Returns (coefficients, standard_errors) both on log2 scale.
///
/// # Arguments
/// * `dds` - DESeqDataSet with size factors, dispersions, and initial GLM fit
/// * `design` - Design matrix (n_samples x n_coefs)
/// * `lambda` - Per-coefficient ridge penalty on natural log scale
pub fn refit_glm_with_prior(
    dds: &DESeqDataSet,
    design: &Array2<f64>,
    lambda: &[f64],
) -> Result<(Array2<f64>, Array2<f64>)> {
    let n_genes = dds.n_genes();
    let n_coefs = design.ncols();

    let counts = dds.counts().counts();
    let size_factors = dds.size_factors().ok_or_else(|| DeseqError::InvalidInput {
        reason: "Size factors required for GLM refit".to_string(),
    })?;
    let dispersions = dds.dispersions().ok_or_else(|| DeseqError::InvalidInput {
        reason: "Dispersions required for GLM refit".to_string(),
    })?;

    // Fit each gene in parallel with the custom lambda
    let results: Vec<GlmRefitResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts = counts.row(i);
            let alpha = dispersions[i];
            fit_single_gene_with_lambda(gene_counts, design, size_factors.view(), alpha, lambda)
        })
        .collect();

    // Extract coefficients and SE, converting to log2 scale
    let mut coefficients = Array2::zeros((n_genes, n_coefs));
    let mut standard_errors = Array2::zeros((n_genes, n_coefs));

    let log2_e = std::f64::consts::LOG2_E; // 1/ln(2)

    for (i, result) in results.into_iter().enumerate() {
        for j in 0..n_coefs {
            // Convert from natural log scale to log2 scale
            coefficients[[i, j]] = result.coefficients[j] * log2_e;
            // SE also converts: SE_log2 = SE_natlog * log2(e) = SE_natlog / ln(2)
            standard_errors[[i, j]] = result.standard_errors[j] * log2_e;
        }
    }

    Ok((coefficients, standard_errors))
}

/// Result from re-fitting a single gene with custom lambda
struct GlmRefitResult {
    /// Coefficients on natural log scale
    coefficients: Vec<f64>,
    /// Standard errors on natural log scale (from sandwich estimator)
    standard_errors: Vec<f64>,
}

/// Fit a single gene's GLM with custom per-coefficient ridge penalty.
/// Nearly identical to `fit_single_gene` but uses the provided lambda vector
/// instead of the hardcoded 1e-6/(ln2^2).
fn fit_single_gene_with_lambda(
    counts: ArrayView1<f64>,
    design: &Array2<f64>,
    size_factors: ArrayView1<f64>,
    alpha: f64,
    lambda: &[f64],
) -> GlmRefitResult {
    let n_samples = counts.len();
    let n_coefs = design.ncols();
    let max_iter = 100;
    let tol = 1e-8;

    // Initialize coefficients using OLS on log counts (same as fit_single_gene)
    // R C++ fitBeta: y_hat(j) = std::log(k(i,j) / nf(i,j) + 0.1);
    let log_counts: Vec<f64> = counts
        .iter()
        .zip(size_factors.iter())
        .map(|(&c, &s)| {
            let norm_ct = if s > 0.0 { c / s } else { 0.0 };
            (norm_ct + 0.1).ln()
        })
        .collect();

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
    let mut beta = solve_symmetric_system(&xtx, &xty, n_coefs);

    if beta.iter().any(|&b| !b.is_finite()) {
        let mean_count: f64 = counts
            .iter()
            .zip(size_factors.iter())
            .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
            .sum::<f64>()
            / n_samples as f64;
        beta = vec![0.0; n_coefs];
        beta[0] = (mean_count.max(0.1)).ln();
    }

    // Save OLS initial betas for optim fallback (same as fit_single_gene)
    let ols_initial_beta = beta.clone();

    let mut converged = false;
    let mut dev_old = 0.0f64;

    // Pre-allocate IRLS buffers outside loop to avoid repeated allocation
    let mut mus = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];
    let mut working_response = vec![0.0; n_samples];

    for _iter in 0..max_iter {
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let mu_raw = nb_mean(eta, size_factors[i]);
            let mu_clamped = mu_raw.max(MIN_MU);
            mus[i] = mu_clamped;
            weights[i] = nb_weight(mu_clamped, alpha);
            working_response[i] =
                (mu_clamped / size_factors[i]).ln() + (counts[i] - mu_clamped) / mu_clamped;
        }

        let new_beta =
            weighted_least_squares_ridge_lambda(design, &weights, &working_response, lambda);

        // R C++: if (sum(abs(beta_hat) > large) > 0) { iter(i) = maxit; break; }
        if new_beta.iter().any(|&b| b.abs() > MAX_LFC_BETA) {
            beta = new_beta;
            break;
        }

        // Compute mu from new beta
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * new_beta[j]).sum();
            mus[i] = nb_mean(eta, size_factors[i]).max(MIN_MU);
        }

        // Compute deviance
        let dev: f64 = mus.iter().zip(counts.iter())
            .map(|(&mu, &y)| {
                let r = 1.0 / alpha;
                -2.0 * nb_log_likelihood_for_deviance(y, mu, r)
            })
            .sum();

        // Convergence test
        let conv_test = (dev - dev_old).abs() / (dev.abs() + 0.1);
        if conv_test.is_nan() {
            beta = new_beta;
            break;
        }
        if _iter > 0 && conv_test < tol {
            beta = new_beta;
            converged = true;
            break;
        }
        dev_old = dev;
        beta = new_beta;
    }

    // L-BFGS-B fallback when IRLS did not converge (matches R DESeq2 optim fallback)
    // R: rowsForOptim <- which(!betaConv | !rowStable | !rowVarPositive)
    // Same pattern as fit_single_gene() but with ACTUAL shrinkage lambda

    // R: rowStable = all betas are finite
    let row_stable = beta.iter().all(|b| b.is_finite());

    // R: rowVarPositive = all diagonal elements of W are positive
    let row_var_positive = {
        let mut all_pos = true;
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let mu = nb_mean(eta, size_factors[i]).max(MIN_MU);
            let w = nb_weight(mu, alpha);
            if w <= 0.0 || !w.is_finite() {
                all_pos = false;
                break;
            }
        }
        all_pos
    };

    if !converged || !row_stable || !row_var_positive {
        let ln2 = std::f64::consts::LN_2;

        // Convert lambda from natural log scale to log2 scale:
        // Penalty: lambda_natlog * beta_natlog^2 = lambda_natlog * (beta_log2 * ln2)^2
        //        = (lambda_natlog * ln2^2) * beta_log2^2
        // So lambda_log2 = lambda_natlog * ln2^2
        let lambda_log2: Vec<f64> = lambda.iter().map(|&l| l * ln2 * ln2).collect();

        // Choose initial values for optim (same logic as fit_single_gene)
        let beta_log2: Vec<f64> = beta.iter().map(|&b| b / ln2).collect();
        let betas_within_bounds = beta_log2.iter().all(|&b| b.abs() < 30.0);

        let initial_for_optim = if row_stable && betas_within_bounds {
            beta_log2
        } else {
            ols_initial_beta
        };

        let counts_slice: Vec<f64> = counts.to_vec();
        let sf_slice: Vec<f64> = size_factors.to_vec();
        let optim_result = optim_nb_fallback_fitting(
            &counts_slice, &sf_slice, alpha, design, &initial_for_optim, &lambda_log2,
        );

        // Use optim result UNCONDITIONALLY (matches R fitNbinomGLMs.R:382)
        beta = optim_result.betas;
    }

    // Compute final weights for SE calculation
    let mut weights = vec![0.0; n_samples];
    for i in 0..n_samples {
        let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
        let mu = nb_mean(eta, size_factors[i]).max(MIN_MU);
        weights[i] = nb_weight(mu, alpha);
    }

    let standard_errors = calculate_standard_errors_with_lambda(design, &weights, lambda);

    GlmRefitResult {
        coefficients: beta,
        standard_errors,
    }
}

/// Weighted least squares with per-coefficient ridge penalty.
/// Like `weighted_least_squares_ridge` but uses a custom lambda vector
/// instead of the hardcoded 1e-6/(ln2^2).
fn weighted_least_squares_ridge_lambda(
    design: &Array2<f64>,
    weights: &[f64],
    response: &[f64],
    lambda: &[f64],
) -> Vec<f64> {
    let n_coefs = design.ncols();
    let mut xtwx = vec![0.0; n_coefs * n_coefs];
    for i in 0..design.nrows() {
        let w = weights[i];
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                xtwx[j * n_coefs + k] += w * design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Add per-coefficient ridge penalty
    for j in 0..n_coefs {
        xtwx[j * n_coefs + j] += lambda[j];
    }

    let mut xtwz = vec![0.0; n_coefs];
    for i in 0..design.nrows() {
        let w = weights[i];
        for j in 0..n_coefs {
            xtwz[j] += w * design[[i, j]] * response[i];
        }
    }

    solve_symmetric_system(&xtwx, &xtwz, n_coefs)
}

/// Calculate standard errors using sandwich estimator with per-coefficient lambda.
/// SE = sqrt(diag( (X'WX + Lambda)^{-1} X'WX (X'WX + Lambda)^{-1} ))
///
/// This matches R DESeq2's C++ fitBeta which uses the sandwich covariance.
fn calculate_standard_errors_with_lambda(
    design: &Array2<f64>,
    weights: &[f64],
    lambda: &[f64],
) -> Vec<f64> {
    let n_samples = design.nrows();
    let n_coefs = design.ncols();

    // Build X'WX
    let mut xtwx = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_samples {
        let w = weights[i];
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                xtwx[j * n_coefs + k] += w * design[[i, j]] * design[[i, k]];
            }
        }
    }

    // X'WX + Lambda (ridge)
    let mut xtwx_ridge = xtwx.clone();
    for j in 0..n_coefs {
        xtwx_ridge[j * n_coefs + j] += lambda[j];
    }

    // Invert (X'WX + Lambda)
    let xtwx_ridge_inv = invert_symmetric_matrix(&xtwx_ridge, n_coefs);

    // Sandwich: (X'WX + Lambda)^{-1} * X'WX * (X'WX + Lambda)^{-1}
    // temp = xtwx_ridge_inv * xtwx
    let mut temp = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_coefs {
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                temp[i * n_coefs + j] += xtwx_ridge_inv[i * n_coefs + k] * xtwx[k * n_coefs + j];
            }
        }
    }

    // sigma = temp * xtwx_ridge_inv
    let mut sigma = vec![0.0; n_coefs * n_coefs];
    for i in 0..n_coefs {
        for j in 0..n_coefs {
            for k in 0..n_coefs {
                sigma[i * n_coefs + j] += temp[i * n_coefs + k] * xtwx_ridge_inv[k * n_coefs + j];
            }
        }
    }

    // SE = sqrt(diag(sigma))
    (0..n_coefs)
        .map(|i| {
            if sigma[i * n_coefs + i] > 0.0 {
                sigma[i * n_coefs + i].sqrt()
            } else {
                f64::NAN
            }
        })
        .collect()
}