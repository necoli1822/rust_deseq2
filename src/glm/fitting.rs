//! GLM fitting using Iteratively Reweighted Least Squares (IRLS)

use ndarray::{Array1, Array2, Array3, ArrayView1};
use rayon::prelude::*;

use super::design::{create_design_matrix, DesignInfo};
use super::negative_binomial::{nb_mean, nb_weight, MAX_LFC_BETA, MIN_MU};
use crate::optim::nmath::{dnbinom_mu_log, dnorm_log, r_sum};
use crate::optim::{optim_lbfgsb, LbfgsbControl};
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
    if !dds.has_size_or_norm_factors() {
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

    // Create design matrix - the model matrix of a design formula, or the extended
    // design if multi-factor model or custom reference levels
    let has_custom_ref = !dds.reference_levels().is_empty();
    let (design, info) = if let Some(info) = dds.formula_design_info() {
        let design = dds.design_matrix().cloned().ok_or_else(|| DeseqError::InvalidDesignMatrix {
            reason: "the design formula has no model matrix".to_string(),
        })?;
        (design, info.clone())
    } else if dds.has_batch_effect() || dds.has_continuous() || has_custom_ref {
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
    let dispersions = dds.dispersions().unwrap();
    let n_genes = dds.n_genes();
    // R (fitNbinomGLMs): getAndCheckWeights() weights, normalised by row maximum.
    // Genes whose weights fail the check are allZero in R and never fitted; they
    // are fitted unweighted here and reported as NA by the tests.
    let weights = dds.normalized_weights();
    let weights_fail = dds.weights_fail();

    // Fit GLM for each gene in parallel
    let results: Vec<GlmFitResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts = counts.row(i);
            let alpha = dispersions[i];
            let w = gene_weights(weights.as_ref(), weights_fail, i);
            // R: normalizationFactors <- getSizeOrNormFactors(object)
            let nf = dds.gene_norm_factors(i).unwrap();
            fit_single_gene_weighted(gene_counts, &design, nf, alpha, params, w.as_deref())
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

/// Normalised observation weights of gene `i` for the GLM fits, or `None` without
/// weights or when the gene's weights fail getAndCheckWeights().
pub(crate) fn gene_weights(weights: Option<&Array2<f64>>, weights_fail: Option<&Vec<bool>>, i: usize) -> Option<Vec<f64>> {
    let w = weights?;
    if weights_fail.is_some_and(|f| f[i]) {
        return None;
    }
    Some(w.row(i).to_vec())
}

/// Fit the NB GLM of one gene without observation weights.
/// R equivalent: fitNbinomGLMs() for one row (C++ fitBeta, then fitNbinomGLMsOptim)
pub fn fit_single_gene(
    counts: ArrayView1<f64>,
    design: &Array2<f64>,
    size_factors: ArrayView1<f64>,
    alpha: f64,
    params: &GlmFitParams,
) -> GlmFitResult {
    fit_single_gene_weighted(counts, design, size_factors, alpha, params, None)
}

/// Fit the NB GLM of one gene with optional observation weights (normalised
/// by row maximum, as getAndCheckWeights() returns them).
/// `size_factors` is the gene's row of getSizeOrNormFactors(): the size
/// factors, or the gene's normalization factors (the fit's offset is their log).
/// R equivalent: fitNbinomGLMs() for one row with `useWeights`: the IRLS weights
/// are `weights * mu/(1 + alpha*mu)` (C++ fitBeta), the convergence deviance,
/// the logLike and the optim() objective are weighted sums, and the hat
/// diagonals and sandwich covariance use the weighted IRLS weights.
pub fn fit_single_gene_weighted(
    counts: ArrayView1<f64>,
    design: &Array2<f64>,
    size_factors: ArrayView1<f64>,
    alpha: f64,
    params: &GlmFitParams,
    obs_weights: Option<&[f64]>,
) -> GlmFitResult {
    fit_single_gene_from(counts, design, size_factors, alpha, params, obs_weights, None)
}

/// [`fit_single_gene_weighted`] from the starting coefficients `start` (natural log
/// scale) instead of the least-squares fit of `log(normalized counts + 0.1)`
/// R equivalent: fitNbinomGLMs() for a model matrix that is not full rank, which starts
/// from `beta_mat` = log(rowMeans(counts(object, normalized = TRUE))) for the
/// Intercept column and 0 otherwise, or 1 for every column without an intercept; the
/// ridge penalty (lambda = 1e-6) makes the IRLS and optim() fits well defined
pub fn fit_single_gene_from(
    counts: ArrayView1<f64>,
    design: &Array2<f64>,
    size_factors: ArrayView1<f64>,
    alpha: f64,
    params: &GlmFitParams,
    obs_weights: Option<&[f64]>,
    start: Option<&[f64]>,
) -> GlmFitResult {
    // IRLS weight of sample i: R `weights.row(i).t() % mu_hat/(1.0 + alpha_hat(i) * mu_hat)`
    let irls_weight = |i: usize, mu: f64| -> f64 {
        match obs_weights {
            Some(w) => w[i] * mu / (1.0 + alpha * mu),
            None => nb_weight(mu, alpha),
        }
    };
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
    let mut beta = match start {
        Some(b) => b.to_vec(),
        None => solve_symmetric_system(&xtx, &xty, n_coefs),
    };

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
            weights[i] = irls_weight(i, mu_clamped);
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
        // (with weights: -2.0 * weights(i,j) * Rf_dnbinom_mu(...))
        let dev: f64 = match obs_weights {
            None => mus.iter().zip(counts.iter())
                .map(|(&mu, &y)| {
                    let r = 1.0 / alpha;
                    -2.0 * nb_log_likelihood_for_deviance(y, mu, r)
                })
                .sum(),
            Some(w) => (0..n_samples)
                .map(|i| -2.0 * w[i] * nb_log_likelihood_for_deviance(counts[i], mus[i], 1.0 / alpha))
                .sum(),
        };

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

    // optim() fallback (fitNbinomGLMsOptim) for the rows R selects with
    // rowsForOptim <- which(!betaConv | !rowStable | !rowVarPositive).

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

    // R keeps the hat diagonals from the IRLS fit even for rows it then refits with
    // optim (fitNbinomGLMsOptim updates beta, SE and mu but not H), and Cook's
    // distances are computed from that H. `mus` holds the clamped mu the C++ loop
    // last computed, so these are the weights fitBeta used for hat_diagonals.
    let irls_hat = if needs_optim {
        let irls_weights: Vec<f64> = mus.iter().enumerate().map(|(i, &mu)| irls_weight(i, mu)).collect();
        Some(calculate_standard_errors_and_hat(design, &irls_weights).1)
    } else {
        None
    };

    let optim_fit = if needs_optim {
        // R: betaMatrix <- log2(exp(1)) * betaRes$beta_mat
        //    betaRow <- if (rowStable[row] & all(abs(betaMatrix[row,]) < large)) {
        //        betaMatrix[row,]
        //    } else {
        //        beta_mat[row,]
        //    }
        // DESeq2 1.50.2 uses beta_mat (the natural-log OLS start) as is, on the
        // log2 scale of the objective; later versions multiply it by log2(e).
        let beta_log2: Vec<f64> = beta.iter().map(|&b| std::f64::consts::LOG2_E * b).collect();
        let start = if row_stable && beta_log2.iter().all(|&b| b.abs() < 30.0) {
            beta_log2
        } else {
            ols_initial_beta.clone()
        };
        let counts_slice: Vec<f64> = counts.to_vec();
        let sf_slice: Vec<f64> = size_factors.to_vec();
        // R: lambda <- rep(1e-6, ncol(modelMatrix))  (log2 scale)
        let lambda_log2 = vec![1e-6; n_coefs];
        let fit = optim_nb_fallback_fitting(&counts_slice, &sf_slice, alpha, design, &start, &lambda_log2, obs_weights);
        // R: betaMatrix[row,] <- o$par (stored here on the natural-log scale)
        beta = fit.beta_log2.iter().map(|&p| p * std::f64::consts::LN_2).collect();
        // R: if (o$convergence == 0) betaConv[row] <- TRUE
        if fit.optim_converged {
            converged = true;
        }
        Some(fit)
    } else {
        None
    };

    let mut mus = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];
    let log_likelihood = if let Some(fit) = &optim_fit {
        // R (fitNbinomGLMsOptim): mu[row,] <- nf * 2^(x %*% o$par) (unclamped);
        // w <- diag((mu_row^-1 + alpha)^-1) and logLike from mu_row clamped at minmu.
        // With weights: w <- diag(weights[row,] * (mu_row^-1 + alpha)^-1)
        mus.copy_from_slice(&fit.mu);
        for (i, (w, &m)) in weights.iter_mut().zip(&fit.mu_clamped).enumerate() {
            *w = match obs_weights {
                Some(ow) => ow[i] * (1.0 / (1.0 / m + alpha)),
                None => 1.0 / (1.0 / m + alpha),
            };
        }
        fit.log_likelihood
    } else {
        // Weights (and so SE) use mu clamped at minmu, but the stored mu is the
        // unclamped nf * exp(X beta), as in R's fitNbinomGLMs; Cook's distances are
        // computed from it. R: logLike <- nbinomLogLike(counts, mu, ...) with the
        // unclamped mu.
        // With weights (nbinomLogLike): rowSums(weights * dnbinom(...)).
        let size = 1.0 / alpha;
        let mut ll = 0.0;
        for i in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|j| design[[i, j]] * beta[j]).sum();
            let mu = nb_mean(eta, size_factors[i]);
            mus[i] = mu;
            weights[i] = irls_weight(i, mu.max(MIN_MU));
            match obs_weights {
                Some(w) => ll += w[i] * nb_log_likelihood_for_deviance(counts[i], mu, size),
                None => ll += nb_log_likelihood_for_deviance(counts[i], mu, size),
            }
        }
        ll
    };

    let (standard_errors, final_hat, covariance) = calculate_standard_errors_and_hat(design, &weights);
    let hat_diagonals = irls_hat.unwrap_or(final_hat);

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
// optim() fallback for rows where IRLS did not converge (fitNbinomGLMsOptim)
// ==========================================================================

/// Result of DESeq2's `fitNbinomGLMsOptim()` for one row.
struct OptimFallbackResult {
    /// `o$par`, on the log2 scale (`betaMatrix[row,] <- o$par`).
    beta_log2: Vec<f64>,
    /// `o$convergence == 0` (R then sets `betaConv[row] <- TRUE`).
    optim_converged: bool,
    /// `mu_row <- as.numeric(nf * 2^(x %*% o$par))`, unclamped (`mu[row,]`).
    mu: Vec<f64>,
    /// `mu_row[mu_row < minmu] <- minmu`, used for the weights and logLike.
    mu_clamped: Vec<f64>,
    /// `logLike[row] <- sum(dnbinom(k, mu = mu_row, size = 1/alpha, log = TRUE))`
    /// with the clamped `mu_row`.
    log_likelihood: f64,
}

/// Row `i` of `x %*% p` as R evaluates it (BLAS `dgemv`): a column-ordered
/// fused multiply-add chain. This is exact for designs of 0/1 indicators; for
/// continuous covariates OpenBLAS's blocked kernel can round differently in
/// the last bit.
fn design_row_times(design: &Array2<f64>, i: usize, p: &[f64]) -> f64 {
    let mut eta = 0.0f64;
    for (j, &pj) in p.iter().enumerate() {
        eta = design[[i, j]].mul_add(pj, eta);
    }
    eta
}

/// `as.numeric(nf * 2^(x %*% p))` of `objectiveFn` in `fitNbinomGLMsOptim`.
fn optim_mu_row(design: &Array2<f64>, nf: &[f64], p: &[f64]) -> Vec<f64> {
    (0..nf.len())
        .map(|i| nf[i] * 2f64.powf(design_row_times(design, i, p)))
        .collect()
}

/// R: `fitNbinomGLMsOptim()` (fitNbinomGLMs.R) for a single row, without
/// weights:
///
/// ```r
/// objectiveFn <- function(p) {
///   mu_row <- as.numeric(nf * 2^(x %*% p))
///   logLike <- sum(dnbinom(k, mu = mu_row, size = 1/alpha, log = TRUE))
///   logPrior <- sum(dnorm(p, 0, sqrt(1/lambda), log = TRUE))
///   negLogPost <- -1 * (logLike + logPrior)
///   if (is.finite(negLogPost)) negLogPost else 10^300
/// }
/// o <- optim(betaRow, objectiveFn, method = "L-BFGS-B", lower = -large, upper = large)
/// ```
///
/// with `large = 30`, the finite-difference gradient of `optim()` (no `gr`)
/// and `minmu = 0.5`. `beta_row` and `lambda` are on the log2 scale. The
/// objective uses the ports of R's `dnbinom`, `dnorm` and `sum` in
/// [`crate::optim::nmath`], so the L-BFGS-B iterates follow R's.
///
/// R stops with an error if `optim()` does; that cannot happen here because
/// non-finite objective values are replaced by 1e300. Should it happen
/// anyway, the starting values are returned as not converged.
///
/// With observation weights (`useWeights`) the objective's and the stored
/// logLike are `sum(weights[row,] * logLikeVector)`.
fn optim_nb_fallback_fitting(
    counts: &[f64],
    size_factors: &[f64],
    alpha: f64,
    design: &Array2<f64>,
    beta_row: &[f64],
    lambda: &[f64],
    obs_weights: Option<&[f64]>,
) -> OptimFallbackResult {
    const LARGE: f64 = 30.0;
    const MINMU: f64 = MIN_MU;
    let size = 1.0 / alpha;
    let sd: Vec<f64> = lambda.iter().map(|&l| (1.0 / l).sqrt()).collect();

    let objective = |p: &[f64]| -> f64 {
        let mu_row = optim_mu_row(design, size_factors, p);
        let log_like = match obs_weights {
            Some(w) => r_sum((0..counts.len()).map(|i| w[i] * dnbinom_mu_log(counts[i], size, mu_row[i]))),
            None => r_sum((0..counts.len()).map(|i| dnbinom_mu_log(counts[i], size, mu_row[i]))),
        };
        let log_prior = r_sum(p.iter().zip(&sd).map(|(&pj, &s)| dnorm_log(pj, 0.0, s)));
        // R: -1 * (logLike + logPrior), which is exactly the negation
        let neg_log_post = -(log_like + log_prior);
        if neg_log_post.is_finite() {
            neg_log_post
        } else {
            1e300
        }
    };

    let (par, optim_converged) =
        match optim_lbfgsb(beta_row, objective, None, &[-LARGE], &[LARGE], &LbfgsbControl::default()) {
            Ok(o) => (o.par, o.convergence == 0),
            Err(_) => (beta_row.to_vec(), false),
        };

    let mu = optim_mu_row(design, size_factors, &par);
    let mu_clamped: Vec<f64> = mu.iter().map(|&m| if m < MINMU { MINMU } else { m }).collect();
    let log_likelihood = match obs_weights {
        Some(w) => r_sum((0..counts.len()).map(|i| w[i] * dnbinom_mu_log(counts[i], size, mu_clamped[i]))),
        None => r_sum((0..counts.len()).map(|i| dnbinom_mu_log(counts[i], size, mu_clamped[i]))),
    };

    OptimFallbackResult {
        beta_log2: par,
        optim_converged,
        mu,
        mu_clamped,
        log_likelihood,
    }
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
    refit_glm_with_prior_counts(dds, dds.counts().counts(), design, lambda)
}

/// [`refit_glm_with_prior`] on the given counts (genes x samples) instead of the
/// dataset's. After outlier replacement R's counts(dds) are the original counts
/// again, and lfcShrink(type = "normal") fits those with the refitted dispersions.
pub fn refit_glm_with_prior_counts(
    dds: &DESeqDataSet,
    counts: ndarray::ArrayView2<f64>,
    design: &Array2<f64>,
    lambda: &[f64],
) -> Result<(Array2<f64>, Array2<f64>)> {
    let n_genes = dds.n_genes();
    let n_coefs = design.ncols();

    if !dds.has_size_or_norm_factors() {
        return Err(DeseqError::InvalidInput {
            reason: "Size factors required for GLM refit".to_string(),
        });
    }
    let dispersions = dds.dispersions().ok_or_else(|| DeseqError::InvalidInput {
        reason: "Dispersions required for GLM refit".to_string(),
    })?;

    // Fit each gene in parallel with the custom lambda
    let results: Vec<GlmRefitResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts = counts.row(i);
            let alpha = dispersions[i];
            fit_single_gene_with_lambda(gene_counts, design, dds.gene_norm_factors(i).unwrap(), alpha, lambda)
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
        let beta_log2: Vec<f64> = beta.iter().map(|&b| std::f64::consts::LOG2_E * b).collect();
        let betas_within_bounds = beta_log2.iter().all(|&b| b.abs() < 30.0);

        let initial_for_optim = if row_stable && betas_within_bounds {
            beta_log2
        } else {
            ols_initial_beta
        };

        let counts_slice: Vec<f64> = counts.to_vec();
        let sf_slice: Vec<f64> = size_factors.to_vec();
        let optim_result = optim_nb_fallback_fitting(
            &counts_slice, &sf_slice, alpha, design, &initial_for_optim, &lambda_log2, None,
        );

        // Use optim result UNCONDITIONALLY (R: betaMatrix[row,] <- o$par)
        beta = optim_result.beta_log2.iter().map(|&p| p * ln2).collect();
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

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    const SF: [f64; 6] = [
        0.89427455600680239,
        1.1916654891192604,
        1.1639976379313646,
        0.57028785514879077,
        1.2315086281834942,
        1.1478657100736156,
    ];

    fn design_3v3() -> Array2<f64> {
        Array2::from_shape_fn((6, 2), |(i, j)| if j == 0 || i >= 3 { 1.0 } else { 0.0 })
    }

    /// `fitNbinomGLMsOptim` on single genes of the validation `poscounts`
    /// scenario, against R 4.5.3 / DESeq2 1.50.2 (`fitNbinomGLMs(dds[g, ])`
    /// with `trace(optim)`; printed with `%.17g`).
    #[test]
    fn optim_fallback_matches_r() {
        // (counts, dispersion, betaRow, o$par)
        type Case = (&'static [f64; 6], f64, [f64; 2], [f64; 2]);
        let cases: [Case; 3] = [
            (
                &[0.0, 0.0, 0.0, 2966.0, 0.0, 5402.0],
                7.4497229636364546,
                [-2.3025850929940468, 7.2061381472882697],
                [-15.672569955307745, 27.361696748687944],
            ),
            (
                &[761.0, 0.0, 0.0, 2049.0, 6583.0, 4544.0],
                5.8325453627524402,
                [0.71377437312718983, 7.6377114716374912],
                [8.1478023470345757, 3.922316996230288],
            ),
            (
                &[0.0, 0.0, 16515.0, 10319.0, 0.0, 9947.0],
                10.0,
                [1.651666959965705, 3.8709697428404657],
                [12.207331837082, 0.91549654179166051],
            ),
        ];
        let x = design_3v3();
        for (k, alpha, start, r_par) in cases {
            let fit = optim_nb_fallback_fitting(k, &SF, alpha, &x, &start, &[1e-6, 1e-6], None);
            assert!(fit.optim_converged);
            for j in 0..2 {
                let rel = (fit.beta_log2[j] - r_par[j]).abs() / r_par[j].abs();
                assert!(rel < 1e-12, "counts {k:?}: par {:?} vs R {r_par:?}", fit.beta_log2);
            }
        }
    }
}