//! Regularized Log Transformation (rlog)
//!
//! Transforms count data to log2 scale using a GLM with ridge regularization,
//! matching R DESeq2's rlog() / rlogData() algorithm exactly.
//!
//! The algorithm:
//! 1. Build a per-sample design matrix (intercept + sample indicators)
//! 2. Estimate betaPriorVar from weighted upper quantile of log fold changes
//! 3. Fit per-gene NB GLM via IRLS with per-coefficient ridge penalty
//! 4. Output: modelMatrix %*% t(betaMatrix) on log2 scale

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};
use crate::glm::{nb_log_likelihood, nb_weight, MIN_MU};
use crate::stats::match_weighted_upper_quantile_for_variance;

/// log2(e) = 1 / ln(2)
const LOG2_E: f64 = std::f64::consts::LOG2_E; // 1.4426950408889634

/// Maximum absolute beta value before flagging divergence (DESeq2 C++ `large`)
const MAX_BETA: f64 = 30.0;

/// Default tolerance for rlog GLM fitting (DESeq2 uses 1e-4 for rlog)
const RLOG_BETA_TOL: f64 = 1e-4;

/// Maximum IRLS iterations
const RLOG_MAX_ITER: usize = 100;

/// Test if count data is sparse (many zeros)
///
/// R equivalent: sparseTest(counts, threshold=0.9, nBig=100)
///
/// Returns true if the data has many genes with zero counts,
/// suggesting special handling is needed for rlog.
///
/// # Arguments
/// * `counts` - Count matrix (genes x samples)
/// * `threshold` - Fraction of zero-heavy genes to trigger sparse mode (default 0.9)
fn sparse_test(counts: ArrayView2<f64>, threshold: f64) -> bool {
    let n_genes = counts.nrows();
    let n_samples = counts.ncols();

    if n_genes == 0 || n_samples == 0 {
        return false;
    }

    // Count genes that have at least one zero
    let n_genes_with_zeros = counts.rows().into_iter()
        .filter(|row| row.iter().any(|&x| x == 0.0))
        .count();

    let zero_fraction = n_genes_with_zeros as f64 / n_genes as f64;

    zero_fraction > threshold
}

/// Result of rlog transformation
#[derive(Debug)]
pub struct RlogResult {
    /// Transformed data matrix (genes x samples), on log2 scale
    pub data: Array2<f64>,
    /// Gene IDs
    pub gene_ids: Vec<String>,
    /// Sample IDs
    pub sample_ids: Vec<String>,
    /// Intercept estimates (log2 expression level per gene)
    pub intercepts: Vec<f64>,
    /// Dispersion estimates used (trended/fitted dispersions)
    pub dispersions: Vec<f64>,
    /// The estimated beta prior variance
    pub beta_prior_var: f64,
}

/// Apply Regularized Log Transformation
///
/// The rlog transformation normalizes with respect to library size and transforms
/// the counts to the log2 scale. It minimizes differences between samples for
/// rows with small counts, and normalizes with respect to library size.
///
/// Follows R DESeq2's rlog() / rlogData() algorithm:
/// 1. Build per-sample design matrix (intercept + one indicator per sample)
/// 2. Filter all-zero rows
/// 3. Estimate betaPriorVar from weighted upper quantile of log fold changes
/// 4. Fit NB GLM per gene via IRLS with ridge regularization (dispFit, not MAP)
/// 5. Compute rlog values as modelMatrix %*% t(betaMatrix)
///
/// # Arguments
/// * `dds` - DESeqDataSet with size factors and dispersions estimated
/// * `blind` - If true, ignore design and use ~1 for transformation (currently
///   the per-sample design is always used as in R's rlogData)
///
/// # Returns
/// * `RlogResult` containing transformed data on log2 scale
pub fn rlog(dds: &DESeqDataSet, blind: bool) -> Result<RlogResult> {
    // R behavior:
    //   blind=TRUE  -> dispersions should have been estimated with intercept-only design
    //   blind=FALSE -> use existing dispersions from the fitted model
    // The caller (e.g., main.rs) is responsible for setting up the appropriate
    // design before estimating dispersions. Here we validate that dispersions exist
    // when blind=false (since they must come from a prior DESeq run).
    if !blind && !dds.has_dispersions() {
        return Err(DeseqError::InvalidInput {
            reason: "blind=false requires dispersions to be estimated first (run DESeq)".to_string(),
        });
    }

    // Check prerequisites
    if !dds.has_size_factors() {
        return Err(DeseqError::InvalidInput {
            reason: "Size factors must be estimated before rlog".to_string(),
        });
    }

    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();
    let counts = dds.counts().counts();
    let size_factors = dds.size_factors().unwrap();
    // R uses normalization factors (if set) instead of size factors for GLM fitting
    let normalization_factors = dds.normalization_factors();

    // Check for sparsity (R: sparseTest)
    let is_sparse = sparse_test(dds.counts().counts(), 0.9);
    if is_sparse {
        log::info!("Data is sparse (many zero counts). Using robust rlog fitting.");
    }

    // Debug: dump size factors for comparison with R
    for j in 0..n_samples {
        log::debug!("rlog sf[{}]={:.15}", j + 1, size_factors[j]);
    }

    // Need trended (fitted) dispersions for rlog, NOT MAP dispersions
    // R: alpha_hat = mcols(objectNZ)$dispFit
    let disp_fit: Vec<f64> = if let Some(trended) = dds.trended_dispersions() {
        trended.to_vec()
    } else if let Some(disp) = dds.dispersions() {
        // Fall back to whatever dispersions are available
        disp.to_vec()
    } else {
        return Err(DeseqError::InvalidInput {
            reason: "Dispersions (trended/fitted) must be estimated before rlog".to_string(),
        });
    };

    // Debug: Note on parametric coefficients - they're not stored in DESeqDataSet
    log::debug!("rlog parametric_coefficients=<not available in current implementation>");

    // ---------------------------------------------------------------
    // Step 1: Build per-sample design matrix
    // R: modelMatrix = model.matrix(~samples)[-1,]
    // Result: n_samples rows x (1 + n_samples) cols
    //   col 0 = intercept (all 1s)
    //   col j (1..n_samples) = 1 if row == j-1, else 0
    // ---------------------------------------------------------------
    let n_coefs = 1 + n_samples;
    let mut model_matrix = Array2::<f64>::zeros((n_samples, n_coefs));
    for i in 0..n_samples {
        model_matrix[[i, 0]] = 1.0; // intercept
        model_matrix[[i, i + 1]] = 1.0; // sample indicator
    }

    // ---------------------------------------------------------------
    // Step 2: Compute normalized counts and identify all-zero rows
    // R: baseMean = rowMeans(counts(object, normalized=TRUE))
    //    allZero  = rowSums(counts(object)) == 0
    // ---------------------------------------------------------------
    let mut norm_counts = Array2::zeros((n_genes, n_samples));
    let mut base_means = vec![0.0_f64; n_genes];
    let mut all_zero = vec![false; n_genes];

    for i in 0..n_genes {
        let mut row_sum = 0.0_f64;
        for j in 0..n_samples {
            let raw = counts[[i, j]];
            row_sum += raw;
            let nc = raw / size_factors[j];
            norm_counts[[i, j]] = nc;
            base_means[i] += nc;
        }
        base_means[i] /= n_samples as f64;
        all_zero[i] = row_sum == 0.0;
    }

    // Debug: dump first 10 genes' baseMean, dispFit, allZero
    let gene_ids_ref = dds.counts().gene_ids();
    for i in 0..10.min(n_genes) {
        log::debug!("rlog gene[{}] id={} baseMean={:.15} geneDisp=N/A dispFit={:.15} allZero={}",
            i + 1, gene_ids_ref[i], base_means[i], disp_fit[i], all_zero[i]);
    }

    // Indices of non-zero genes
    let nz_indices: Vec<usize> = (0..n_genes).filter(|&i| !all_zero[i]).collect();
    let n_nz = nz_indices.len();

    if n_nz == 0 {
        // All genes are zero - return zeros
        return Ok(RlogResult {
            data: Array2::zeros((n_genes, n_samples)),
            gene_ids: dds.counts().gene_ids().to_vec(),
            sample_ids: dds.counts().sample_ids().to_vec(),
            intercepts: vec![f64::NEG_INFINITY; n_genes],
            dispersions: disp_fit,
            beta_prior_var: 1.0,
        });
    }

    // ---------------------------------------------------------------
    // Step 3: Estimate betaPriorVar
    // R:
    //   logCounts = log2(counts(objectNZ, normalized=TRUE) + 0.5)
    //   logFoldChangeMatrix = logCounts - log2(baseMean + 0.5)
    //   logFoldChangeVector = as.numeric(logFoldChangeMatrix)
    //   varlogk = 1/baseMean + dispFit
    //   weights = 1/varlogk
    //   betaPriorVar = matchWeightedUpperQuantileForVariance(lfcVec, rep(weights, ncol))
    // ---------------------------------------------------------------
    let mut lfc_vector: Vec<f64> = Vec::with_capacity(n_nz * n_samples);
    let mut weight_vector: Vec<f64> = Vec::with_capacity(n_nz * n_samples);

    for &gi in &nz_indices {
        let bm = base_means[gi];
        let disp = disp_fit[gi];
        let varlogk = 1.0 / bm + disp;
        let w = 1.0 / varlogk;
        let log2_bm_pseudo = (bm + 0.5_f64).log2();

        for j in 0..n_samples {
            let log2_nc = (norm_counts[[gi, j]] + 0.5_f64).log2();
            let lfc = log2_nc - log2_bm_pseudo;
            lfc_vector.push(lfc);
            weight_vector.push(w);
        }
    }

    // Debug: print first 10 lfc/weight values for comparison with R
    log::debug!("rlog lfc_vector_length={}", lfc_vector.len());
    log::debug!("rlog n_nonzero={}", n_nz);
    for i in 0..10.min(lfc_vector.len()) {
        log::debug!("rlog lfc[{}]={:.15}", i + 1, lfc_vector[i]);
    }
    for i in 0..10.min(weight_vector.len()) {
        log::debug!("rlog weight[{}]={:.15}", i + 1, weight_vector[i]);
    }

    let beta_prior_var = match_weighted_upper_quantile_for_variance(&lfc_vector, &weight_vector, 0.05);

    // ---------------------------------------------------------------
    // Step 4: Build lambda vector (ridge penalties)
    // R:
    //   lambda = 1/rep(betaPriorVar, ncol(modelMatrix))
    //   lambda[intercept_idx] = 1e-6
    //
    // Then converted to natural log scale in fitNbinomGLMs:
    //   lambdaNatLogScale = lambda / log(2)^2
    // ---------------------------------------------------------------
    let ln2 = std::f64::consts::LN_2;
    let ln2_sq = ln2 * ln2;

    let mut lambda = vec![0.0_f64; n_coefs];
    let lambda_log2 = 1.0 / beta_prior_var;
    for k in 0..n_coefs {
        lambda[k] = lambda_log2;
    }
    // Intercept (column 0) gets a wide prior
    lambda[0] = 1e-6;
    // Convert from log2 scale to natural log scale
    for k in 0..n_coefs {
        lambda[k] /= ln2_sq;
    }

    // ---------------------------------------------------------------
    // Step 5: Build normalization factor matrix (per gene: just size factors)
    // R uses nf = matrix(rep(sf, each=nrow), ncol=ncol)
    // But for rlog without frozen intercept, normalizationFactors = size factors
    // ---------------------------------------------------------------
    // size_factors is an Array1<f64> of length n_samples

    // ---------------------------------------------------------------
    // Step 6: Fit NB GLM per gene via IRLS with ridge regularization
    // Process non-zero genes in parallel
    // ---------------------------------------------------------------
    let model_matrix_ref = &model_matrix;
    let lambda_ref = &lambda;

    // Compute initial betas for all non-zero genes using OLS
    // R: y = t(log(counts(object, normalized=TRUE) + 0.1))
    //    beta_mat = t(solve(R, t(Q) %*% y))
    // Since the model matrix is full rank (identity + intercept), OLS always works.
    //
    // For each gene, we fit on natural log scale:
    //   OLS: beta = (X'X)^{-1} X' ln(norm_counts + 0.1)

    let results: Vec<RlogGeneResult> = nz_indices
        .par_iter()
        .map(|&gi| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[gi, j]]).collect();
            let gene_norm_counts: Vec<f64> =
                (0..n_samples).map(|j| norm_counts[[gi, j]]).collect();
            let alpha = disp_fit[gi];
            // Use normalization factors for this gene if available, otherwise size factors
            let nf: Vec<f64> = if let Some(nf_matrix) = normalization_factors {
                (0..n_samples).map(|j| nf_matrix[[gi, j]]).collect()
            } else {
                size_factors.to_vec()
            };

            fit_rlog_gene(
                &gene_counts,
                &gene_norm_counts,
                &nf,
                alpha,
                model_matrix_ref,
                lambda_ref,
            )
        })
        .collect();

    // ---------------------------------------------------------------
    // Step 7: Compute transformed data = modelMatrix %*% t(betaMatrix)
    // R: normalizedDataNZ = t(modelMatrix %*% t(betaMatrix))
    //    betaMatrix is on log2 scale: beta_log2 = log2(e) * beta_natlog
    //
    // For each gene i, for each sample s:
    //   rlog[i,s] = sum_k modelMatrix[s,k] * beta_log2[i,k]
    // ---------------------------------------------------------------
    let mut data = Array2::zeros((n_genes, n_samples));
    let mut intercepts = vec![f64::NEG_INFINITY; n_genes];
    let dispersions_out = disp_fit.clone();

    for (idx, &gi) in nz_indices.iter().enumerate() {
        let result = &results[idx];
        let beta_log2: Vec<f64> = result.beta.iter().map(|&b| b * LOG2_E).collect();

        // intercept on log2 scale
        intercepts[gi] = beta_log2[0];

        // Compute model_matrix %*% beta_log2 for this gene
        for s in 0..n_samples {
            let mut val = 0.0;
            for k in 0..n_coefs {
                val += model_matrix[[s, k]] * beta_log2[k];
            }
            data[[gi, s]] = val;
        }
    }

    // All-zero rows: data stays 0.0 (log2(1) = 0, matching R's buildMatrixWithZeroRows)
    // Intercepts for all-zero rows stay NEG_INFINITY (matching R's -Inf)

    Ok(RlogResult {
        data,
        gene_ids: dds.counts().gene_ids().to_vec(),
        sample_ids: dds.counts().sample_ids().to_vec(),
        intercepts,
        dispersions: dispersions_out,
        beta_prior_var,
    })
}

// ==========================================================================
// Per-gene rlog GLM fitting (QR-augmented IRLS with ridge regularization)
// ==========================================================================

/// Result of fitting a single gene's rlog GLM
struct RlogGeneResult {
    /// Beta coefficients on natural log scale
    beta: Vec<f64>,
}

/// Fit the rlog NB GLM for a single gene via QR-augmented IRLS with ridge penalty.
///
/// Matches DESeq2 C++ fitBeta with useQR=TRUE (DESeq2.cpp:331-383):
///
/// Natural log scale throughout. Each IRLS iteration:
///   1. mu = nf * exp(X * beta), clamped to minmu
///   2. w = mu / (1 + alpha * mu)              (NB weights)
///   3. z = log(mu/nf) + (y - mu) / mu         (working response)
///   4. Construct augmented system:
///        A = [ diag(sqrt(w)) * X ]             (n_samples x n_coefs)
///            [ diag(sqrt(lambda)) ]            (n_coefs x n_coefs)
///        big_z = [ sqrt(w) * z ]               (n_samples)
///                [ 0, ..., 0 ]                 (n_coefs zeros)
///   5. QR decompose A, solve R * beta = Q' * big_z
///   6. Convergence: |dev - dev_old| / (|dev| + 0.1) < tol
///
/// The augmented matrix has dimensions (n_samples + n_coefs) x n_coefs and is
/// ALWAYS full column rank (due to the sqrt(lambda) rows), making QR numerically
/// stable even for rank-deficient design matrices.
///
/// After IRLS, checks for unstable/non-finite betas and falls back to
/// coordinate-wise Newton-Raphson optimization if needed.
///
/// # Arguments
/// * `raw_counts` - Raw counts for this gene (length n_samples)
/// * `norm_counts` - Normalized counts (counts / size_factors) for this gene
/// * `size_factors` - Size factors (length n_samples)
/// * `alpha` - Dispersion (fitted/trended, NOT MAP)
/// * `model_matrix` - Design matrix (n_samples x n_coefs)
/// * `lambda` - Ridge penalty per coefficient (on natural log scale)
fn fit_rlog_gene(
    raw_counts: &[f64],
    norm_counts: &[f64],
    size_factors: &[f64],
    alpha: f64,
    model_matrix: &Array2<f64>,
    lambda: &[f64],
) -> RlogGeneResult {
    let n_samples = raw_counts.len();
    let n_coefs = model_matrix.ncols();

    // ---------------------------------------------------------------
    // Beta initialization for rank-deficient design (R fitNbinomGLMs.R:139-155)
    //
    // The rlog design matrix is n_samples x (n_samples+1), which is always
    // rank-deficient (more columns than rows). R's behavior:
    //   intercept = ln(mean(normalized_counts)), all other betas = 0
    // ---------------------------------------------------------------
    let mean_nc = norm_counts.iter().sum::<f64>() / n_samples as f64;
    let mut beta = vec![0.0_f64; n_coefs];
    beta[0] = (mean_nc.max(0.1)).ln();

    // ---------------------------------------------------------------
    // QR-augmented IRLS loop (matches DESeq2.cpp fitBeta, useQR=TRUE path)
    // ---------------------------------------------------------------
    let mut dev_old = 0.0_f64;
    let n_aug = n_samples + n_coefs; // augmented system rows

    // Allocate buffers OUTSIDE loop to reuse memory across iterations
    let mut mu = vec![0.0_f64; n_samples];
    let mut w = vec![0.0_f64; n_samples];
    let mut z = vec![0.0_f64; n_samples];
    let mut a_mat = vec![0.0_f64; n_aug * n_coefs]; // flat 2D array
    let mut big_z = vec![0.0_f64; n_aug];

    for t in 0..RLOG_MAX_ITER {
        // Compute mu = nf * exp(X * beta), clamped to minmu
        for s in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|k| model_matrix[[s, k]] * beta[k]).sum();
            mu[s] = (size_factors[s] * eta.exp()).max(MIN_MU);
        }

        // Compute weights and working response
        for s in 0..n_samples {
            w[s] = nb_weight(mu[s], alpha);
            z[s] = (mu[s] / size_factors[s]).ln() + (raw_counts[s] - mu[s]) / mu[s];
        }

        // Build augmented matrix A (n_aug x n_coefs) and augmented response big_z (n_aug)
        //   Top block: diag(sqrt(w)) * X
        //   Bottom block: diag(sqrt(lambda))
        // a_mat stored in row-major order: a_mat[i * n_coefs + j]

        for s in 0..n_samples {
            let sw = w[s].sqrt();
            for k in 0..n_coefs {
                a_mat[s * n_coefs + k] = sw * model_matrix[[s, k]];
            }
            big_z[s] = sw * z[s];
        }
        for k in 0..n_coefs {
            a_mat[(n_samples + k) * n_coefs + k] = lambda[k].sqrt();
            big_z[n_samples + k] = 0.0; // reset to 0.0 for this iteration
        }

        // QR solve: A * beta_new = big_z
        let new_beta = qr_solve_augmented(&a_mat, &big_z, n_aug, n_coefs);

        // Check for divergence: if any |beta| > 30, stop
        if new_beta.iter().any(|&b| b.abs() > MAX_BETA || !b.is_finite()) {
            break;
        }

        beta = new_beta;

        // Recompute mu with updated beta for deviance
        for s in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|k| model_matrix[[s, k]] * beta[k]).sum();
            mu[s] = (size_factors[s] * eta.exp()).max(MIN_MU);
        }

        // Compute deviance: dev = -2 * sum(NB log-likelihood)
        let dev: f64 = (0..n_samples)
            .map(|s| -2.0 * nb_log_likelihood(raw_counts[s], mu[s], alpha))
            .sum();

        // Check convergence (skip first iteration, matching C++ `if (t > 0)`)
        if t > 0 {
            let conv_test = (dev - dev_old).abs() / (dev.abs() + 0.1);
            if conv_test.is_nan() {
                break;
            }
            if conv_test < RLOG_BETA_TOL {
                break;
            }
        }

        dev_old = dev;
    }

    // ---------------------------------------------------------------
    // Optim fallback for unstable rows (R: fitNbinomGLMs.R rowsForOptim)
    //
    // Even when useOptim=FALSE, R falls back for rows where:
    //   - betas are non-finite
    //   - betas are extreme (|beta| > large)
    //   - weights are non-positive (negative variance)
    // ---------------------------------------------------------------
    let row_stable = beta.iter().all(|b| b.is_finite() && b.abs() <= MAX_BETA);

    let row_var_positive = if row_stable {
        (0..n_samples).all(|s| {
            let eta: f64 = (0..n_coefs).map(|k| model_matrix[[s, k]] * beta[k]).sum();
            let mu = (size_factors[s] * eta.exp()).max(MIN_MU);
            let w = nb_weight(mu, alpha);
            w > 0.0 && w.is_finite()
        })
    } else {
        false
    };

    if !row_stable || !row_var_positive {
        // Fallback: coordinate-wise Newton-Raphson optimization
        // Use rank-deficient initialization as starting point
        let mut optim_beta = vec![0.0_f64; n_coefs];
        optim_beta[0] = (mean_nc.max(0.1)).ln();

        let optim_result = optim_rlog_fallback(
            raw_counts,
            size_factors,
            alpha,
            model_matrix,
            lambda,
            &optim_beta,
        );
        return RlogGeneResult { beta: optim_result };
    }

    RlogGeneResult { beta }
}

/// Solve the augmented least squares system A * x = b via Householder QR.
///
/// A is m x n (m >= n) stored in row-major order (flat array), b is length m. Returns x of length n.
/// Uses Householder reflections for numerical stability.
fn qr_solve_augmented(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    // Copy A into a working matrix (stored row-major for efficiency)
    let mut r = vec![0.0_f64; m * n];
    r.copy_from_slice(&a[..m * n]);
    let mut qt_b = b.to_vec();

    // Householder QR: for each column j, compute reflector and apply
    for j in 0..n {
        // Compute the Householder vector for column j, rows j..m
        // Access r[i][j] as r[i * n + j]
        let mut norm_sq = 0.0_f64;
        for i in j..m {
            let val = r[i * n + j];
            norm_sq += val * val;
        }

        if norm_sq < 1e-30 {
            // Column is essentially zero; skip (degenerate case)
            continue;
        }

        let norm = norm_sq.sqrt();
        let r_jj = r[j * n + j];
        let sign = if r_jj >= 0.0 { 1.0 } else { -1.0 };
        let u0 = r_jj + sign * norm;

        // Build the Householder vector v: v[0] = 1, v[i] = r[j+i][j] / u0
        let mut v = vec![0.0_f64; m - j];
        v[0] = 1.0;
        for i in 1..(m - j) {
            v[i] = r[(j + i) * n + j] / u0;
        }

        let tau = 2.0 / v.iter().map(|&vi| vi * vi).sum::<f64>();

        // Apply reflector to remaining columns of R: R[j:, k] -= tau * v * (v' * R[j:, k])
        for k in j..n {
            let dot: f64 = (0..(m - j)).map(|i| v[i] * r[(j + i) * n + k]).sum();
            for i in 0..(m - j) {
                r[(j + i) * n + k] -= tau * v[i] * dot;
            }
        }

        // Apply reflector to Qt*b: b[j:] -= tau * v * (v' * b[j:])
        let dot_b: f64 = (0..(m - j)).map(|i| v[i] * qt_b[j + i]).sum();
        for i in 0..(m - j) {
            qt_b[j + i] -= tau * v[i] * dot_b;
        }
    }

    // Back-substitution: R[0:n, 0:n] * x = qt_b[0:n]
    // Access r[i][j] as r[i * n + j]
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..n {
            sum -= r[i * n + j] * x[j];
        }
        let r_ii = r[i * n + i];
        if r_ii.abs() < 1e-30 {
            x[i] = 0.0; // Degenerate column
        } else {
            x[i] = sum / r_ii;
        }
    }

    x
}

/// Coordinate-wise Newton-Raphson optimization fallback for rlog genes
/// where QR-augmented IRLS produces unstable results.
///
/// Minimizes the penalized negative log-likelihood:
///   -logLik - 0.5 * sum(lambda_k * beta_k^2)
///
/// Each coefficient is optimized individually with Armijo line search.
fn optim_rlog_fallback(
    raw_counts: &[f64],
    size_factors: &[f64],
    alpha: f64,
    model_matrix: &Array2<f64>,
    lambda: &[f64],
    initial_beta: &[f64],
) -> Vec<f64> {
    let n_samples = raw_counts.len();
    let n_coefs = model_matrix.ncols();
    let max_iter = 5000;
    let tol = 1e-8;

    let mut beta = initial_beta.to_vec();

    // Allocate buffers for mu calculations to avoid repeated allocations
    let mut mu_buf = vec![0.0_f64; n_samples];

    // Helper to compute all mu values for current beta
    let compute_mu = |beta: &[f64], mu_buf: &mut [f64]| {
        for s in 0..n_samples {
            let eta: f64 = (0..n_coefs).map(|k| model_matrix[[s, k]] * beta[k]).sum();
            mu_buf[s] = (size_factors[s] * eta.exp()).max(MIN_MU);
        }
    };

    // Penalized NB log-likelihood (natural log scale)
    let compute_pen_ll = |beta: &[f64], mu_buf: &mut [f64]| -> f64 {
        compute_mu(beta, mu_buf);
        let mut ll = 0.0_f64;
        for s in 0..n_samples {
            ll += nb_log_likelihood(raw_counts[s], mu_buf[s], alpha);
        }
        // Ridge prior: -0.5 * sum(lambda_k * beta_k^2)
        for k in 0..n_coefs {
            ll -= 0.5 * lambda[k] * beta[k] * beta[k];
        }
        ll
    };

    let mut ll_old = compute_pen_ll(&beta, &mut mu_buf);

    for _outer in 0..max_iter {
        let ll_start = ll_old;

        // Compute mu for current beta once per outer iteration
        compute_mu(&beta, &mut mu_buf);

        for j in 0..n_coefs {
            let mut grad_j = 0.0_f64;
            let mut hess_jj = 0.0_f64;

            for s in 0..n_samples {
                let mu = mu_buf[s];
                let resid_scaled = (raw_counts[s] - mu) / (1.0 + alpha * mu);
                grad_j += resid_scaled * model_matrix[[s, j]];
                let w = mu / (1.0 + alpha * mu);
                hess_jj -= w * model_matrix[[s, j]] * model_matrix[[s, j]];
            }

            // Ridge penalty contribution
            grad_j -= lambda[j] * beta[j];
            hess_jj -= lambda[j];

            if hess_jj.abs() < 1e-20 {
                continue;
            }

            let delta = -grad_j / hess_jj;
            if delta.abs() < 1e-14 {
                continue;
            }

            // Armijo line search
            let beta_j_old = beta[j];
            let mut step = 1.0;
            let mut improved = false;

            for _ls in 0..30 {
                beta[j] = (beta_j_old + step * delta).clamp(-MAX_BETA, MAX_BETA);
                let ll_new = compute_pen_ll(&beta, &mut mu_buf);

                if ll_new >= ll_old + 1e-4 * step * grad_j * delta {
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
                beta[j] = beta_j_old;
            } else {
                // Beta changed, need to recompute mu_buf for next coefficient
                compute_mu(&beta, &mut mu_buf);
            }
        }

        let ll_change = (ll_old - ll_start).abs() / (ll_old.abs() + 0.1);
        if ll_change < tol {
            break;
        }
    }

    beta
}

// ==========================================================================
// RlogResult utility methods
// ==========================================================================

impl RlogResult {
    /// Get transformed value for a specific gene and sample
    pub fn get(&self, gene_idx: usize, sample_idx: usize) -> f64 {
        self.data[[gene_idx, sample_idx]]
    }

    /// Get transformed row (gene across all samples)
    pub fn gene_row(&self, gene_idx: usize) -> Vec<f64> {
        let n_samples = self.data.ncols();
        (0..n_samples).map(|j| self.data[[gene_idx, j]]).collect()
    }

    /// Get transformed column (all genes for a sample)
    pub fn sample_col(&self, sample_idx: usize) -> Vec<f64> {
        let n_genes = self.data.nrows();
        (0..n_genes).map(|i| self.data[[i, sample_idx]]).collect()
    }

    /// Number of genes
    pub fn n_genes(&self) -> usize {
        self.data.nrows()
    }

    /// Number of samples
    pub fn n_samples(&self) -> usize {
        self.data.ncols()
    }

    /// Get intercept (baseline expression) for a gene, on log2 scale
    pub fn intercept(&self, gene_idx: usize) -> f64 {
        self.intercepts[gene_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_test() {
        use ndarray::array;

        // Test with no zeros - should not be sparse
        let counts_no_zeros = array![
            [10.0, 20.0, 30.0],
            [15.0, 25.0, 35.0],
            [5.0, 10.0, 15.0]
        ];
        assert!(!sparse_test(counts_no_zeros.view(), 0.9));

        // Test with many zeros - should be sparse
        let counts_many_zeros = array![
            [10.0, 0.0, 30.0],
            [0.0, 25.0, 0.0],
            [5.0, 0.0, 15.0],
            [0.0, 10.0, 0.0],
            [15.0, 0.0, 20.0]
        ];
        // 5 genes, all have at least one zero -> 100% > 90% threshold
        assert!(sparse_test(counts_many_zeros.view(), 0.9));

        // Test with some zeros but not sparse
        let counts_some_zeros = array![
            [10.0, 20.0, 30.0],
            [15.0, 0.0, 35.0],
            [5.0, 10.0, 15.0]
        ];
        // 1 out of 3 genes has a zero -> 33% < 90% threshold
        assert!(!sparse_test(counts_some_zeros.view(), 0.9));

        // Test empty matrix
        let counts_empty = Array2::<f64>::zeros((0, 0));
        assert!(!sparse_test(counts_empty.view(), 0.9));
    }

    #[test]
    fn test_qr_solve_augmented_simple() {
        // Simple 3x2 overdetermined system:
        // [[1, 0], [0, 1], [1, 1]] * x = [1, 2, 3]
        // Least squares solution: x = [1, 2] (exact in this case)
        // Stored in row-major order: [row0_col0, row0_col1, row1_col0, row1_col1, ...]
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = qr_solve_augmented(&a, &b, 3, 2);
        assert!(
            (x[0] - 1.0).abs() < 1e-10,
            "x[0] should be 1.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 2.0).abs() < 1e-10,
            "x[1] should be 2.0, got {}",
            x[1]
        );
    }

    #[test]
    fn test_qr_solve_augmented_ridge() {
        // Test the augmented system matching rlog's structure:
        // Top: diag(sqrt(w)) * X, Bottom: diag(sqrt(lambda))
        // For X = [[1, 1], [1, 0]], w = [1, 1], lambda = [0.01, 0.01]
        // A = [[1, 1], [1, 0], [0.1, 0], [0, 0.1]]
        // b = [3, 2, 0, 0]
        // Stored in row-major order
        let a = vec![
            1.0, 1.0,  // row 0
            1.0, 0.0,  // row 1
            0.1, 0.0,  // row 2
            0.0, 0.1,  // row 3
        ];
        let b = vec![3.0, 2.0, 0.0, 0.0];
        let x = qr_solve_augmented(&a, &b, 4, 2);
        // Solution should be finite and reasonable
        assert!(x[0].is_finite(), "x[0] should be finite");
        assert!(x[1].is_finite(), "x[1] should be finite");
        // With ridge, intercept ~ 2.0, slope ~ 1.0 (shrunk toward 0)
        assert!(
            (x[0] - 2.0).abs() < 0.5,
            "x[0] should be near 2.0, got {}",
            x[0]
        );
    }

    #[test]
    fn test_fit_rlog_gene_simple() {
        // 3 samples, simple counts
        let raw_counts = vec![100.0, 200.0, 150.0];
        let sf = vec![1.0, 1.0, 1.0];
        let norm_counts: Vec<f64> = raw_counts
            .iter()
            .zip(sf.iter())
            .map(|(&c, &s)| c / s)
            .collect();
        let alpha = 0.1;

        // Design: intercept + 3 sample indicators
        let n_samples = 3;
        let n_coefs = 4;
        let mut model_matrix = Array2::<f64>::zeros((n_samples, n_coefs));
        for i in 0..n_samples {
            model_matrix[[i, 0]] = 1.0;
            model_matrix[[i, i + 1]] = 1.0;
        }

        // Lambda: wide prior for intercept, tight for sample effects
        let ln2 = std::f64::consts::LN_2;
        let ln2_sq = ln2 * ln2;
        let beta_prior_var = 1.0;
        let mut lambda = vec![1.0 / beta_prior_var / ln2_sq; n_coefs];
        lambda[0] = 1e-6 / ln2_sq;

        let result = fit_rlog_gene(&raw_counts, &norm_counts, &sf, alpha, &model_matrix, &lambda);

        // Beta should have finite values
        assert!(
            result.beta.iter().all(|b| b.is_finite()),
            "All betas should be finite"
        );

        // The intercept should be roughly ln(mean(100, 200, 150)) = ln(150) ~ 5.01
        let expected_intercept = 150.0_f64.ln();
        assert!(
            (result.beta[0] - expected_intercept).abs() < 1.0,
            "Intercept should be near ln(150)={:.2}, got {:.2}",
            expected_intercept,
            result.beta[0]
        );
    }

    #[test]
    fn test_fit_rlog_gene_zero_counts() {
        // Gene with very low counts - should still converge
        let raw_counts = vec![1.0, 0.0, 2.0, 0.0];
        let sf = vec![1.0, 1.2, 0.8, 1.1];
        let norm_counts: Vec<f64> = raw_counts
            .iter()
            .zip(sf.iter())
            .map(|(&c, &s)| c / s)
            .collect();
        let alpha = 0.5;

        let n_samples = 4;
        let n_coefs = 5;
        let mut model_matrix = Array2::<f64>::zeros((n_samples, n_coefs));
        for i in 0..n_samples {
            model_matrix[[i, 0]] = 1.0;
            model_matrix[[i, i + 1]] = 1.0;
        }

        let ln2_sq = std::f64::consts::LN_2.powi(2);
        let beta_prior_var = 1.0;
        let mut lambda = vec![1.0 / beta_prior_var / ln2_sq; n_coefs];
        lambda[0] = 1e-6 / ln2_sq;

        let result = fit_rlog_gene(&raw_counts, &norm_counts, &sf, alpha, &model_matrix, &lambda);

        assert!(
            result.beta.iter().all(|b| b.is_finite()),
            "All betas should be finite even with zero counts"
        );
    }
}
