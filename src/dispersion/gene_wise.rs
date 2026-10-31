//! Gene-wise dispersion estimation using Cox-Reid adjusted profile likelihood
//!
//! This module implements the dispersion estimation method used by DESeq2,
//! which uses alternating optimization between dispersion and GLM coefficients.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::data::DESeqDataSet;
use crate::dispersion::DispersionParams;
use crate::error::{DeseqError, Result};
use crate::glm::check_full_rank;

/// Check if the design matrix represents a simple group model
/// (number of unique row patterns equals number of columns).
/// When true, DESeq2 uses linear model (OLS) instead of NB IRLS for mu estimation.
fn is_linear_mu(design: &Array2<f64>) -> bool {
    let n = design.nrows();
    let p = design.ncols();

    // Collect unique row patterns
    let mut unique_rows: Vec<Vec<i64>> = Vec::new();
    for i in 0..n {
        // Convert to integer bits for exact comparison (design matrices have 0/1 entries)
        let row: Vec<i64> = (0..p).map(|j| (design[[i, j]] * 1000.0).round() as i64).collect();
        if !unique_rows.contains(&row) {
            unique_rows.push(row);
        }
    }

    unique_rows.len() == p
}

/// Observation weights of one gene as the dispersion fits use them
/// (`weightsSEXP` of DESeq2.cpp `fitDisp` / `fitDispGrid`), with the model matrix
/// of the Cox-Reid term restricted to the samples with weight > weightThreshold.
#[derive(Debug, Clone)]
pub struct DispWeights {
    /// Weights multiplying each sample's log-likelihood term
    pub w: Vec<f64>,
    /// R: x.rows(find(wts > weightThreshold)), then x.cols(find(sum(abs(x)) > 0))
    pub cr_design: Array2<f64>,
    /// Samples kept in the Cox-Reid term
    pub cr_rows: Vec<usize>,
}

impl DispWeights {
    pub fn new(w: Vec<f64>, design: &Array2<f64>) -> Self {
        let (cr_design, cr_rows) =
            crate::data::weights::cr_design(design, &w, crate::data::weights::WEIGHT_THRESHOLD);
        Self { w, cr_design, cr_rows }
    }

    /// mu of the samples kept in the Cox-Reid term
    pub(crate) fn cr_mu(&self, mu: &[f64]) -> Vec<f64> {
        self.cr_rows.iter().map(|&i| mu[i]).collect()
    }
}

/// Observation weights of one gene for estimateDispersionsGeneEst()
#[derive(Debug, Clone)]
pub struct GeneEstWeights {
    /// Normalised weights, used by fitNbinomGLMs() for mu
    pub glm: Vec<f64>,
    /// R: pmax(weights, 1e-6), used by fitDisp() / fitDispGrid()
    pub disp: DispWeights,
    /// The gene's raw weights times its normalized counts (getBaseMeansAndVariances),
    /// whose mean and variance give momentsDispEstimate()
    pub base_mean_counts: Vec<f64>,
}

impl GeneEstWeights {
    /// `normalized` are the gene's row-max-normalised weights and
    /// `base_mean_counts` its raw weights times its normalized counts.
    pub fn new(normalized: Vec<f64>, base_mean_counts: Vec<f64>, design: &Array2<f64>) -> Self {
        let floored: Vec<f64> = normalized
            .iter()
            .map(|&x| x.max(crate::data::weights::GENE_EST_MIN_WEIGHT))
            .collect();
        Self { glm: normalized, disp: DispWeights::new(floored, design), base_mean_counts }
    }
}

/// Estimate gene-wise dispersions using Cox-Reid adjusted profile likelihood
/// R equivalent: estimateDispersionsGeneEst() in core.R
/// with alternating optimization (matching R DESeq2)
pub fn estimate_gene_dispersions(dds: &mut DESeqDataSet, params: &DispersionParams) -> Result<()> {
    if !dds.has_size_or_norm_factors() {
        return Err(DeseqError::DispersionEstimationFailed {
            gene_id: "N/A".to_string(),
            reason: "Size factors must be estimated first".to_string(),
        });
    }

    let counts = dds.counts().counts();
    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();

    // Create design matrix - use full design for multi-factor models
    let design = create_design_matrix_for_dispersion(dds)?;

    // R's checkForExperimentalReplicates: warn if nrow == ncol (no replicates)
    if design.nrows() == design.ncols() {
        log::warn!(
            "No replicates detected (nrow(modelMatrix) == ncol(modelMatrix) == {}). \
             Dispersion estimates may be unreliable.",
            design.nrows()
        );
    }

    // Observation weights (R: getAndCheckWeights): normalised by row maximum;
    // genes whose weights do not allow estimation are treated as allZero.
    let weights = dds.normalized_weights();
    let weights_fail: Option<Vec<bool>> = weights.as_ref().map(|w| {
        let ok = crate::data::weights::weights_ok(w, &design);
        let n_fail = ok.iter().filter(|&&o| !o).count();
        if n_fail > 0 {
            log::warn!(
                "for {} row(s), the weights as supplied won't allow parameter estimation, \
                 producing a degenerate design matrix; these rows are treated as if they \
                 contained all zeros",
                n_fail
            );
        }
        ok.into_iter().map(|o| !o).collect()
    });
    let base_mean_counts = dds.base_mean_counts().map(|c| c.into_owned());

    // xim of momentsDispEstimate(objectNZ): mean(1/sizeFactors), or with
    // normalization factors mean(1/colMeans(nf)) over the genes that are fitted
    // (not allZero, which includes the genes whose weights fail)
    let nz_genes: Vec<usize> = (0..n_genes)
        .filter(|&i| {
            counts.row(i).iter().any(|&c| c != 0.0) && !weights_fail.as_ref().is_some_and(|f| f[i])
        })
        .collect();
    let xim = dds.moments_xim(&nz_genes).unwrap();

    // Check if design supports linear mu (simple group model)
    // R: linearMu is forced FALSE when weights are used
    let use_linear_mu = is_linear_mu(&design) && weights.is_none();

    // Estimate dispersions in parallel using Cox-Reid likelihood
    // Returns (dispersion, mu) for each gene - mu is needed for MAP estimation
    let results: Vec<(f64, Vec<f64>)> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[i, j]]).collect();
            let gene_weights = match (&weights, &weights_fail, &base_mean_counts) {
                (Some(w), Some(fail), Some(bm)) => {
                    if fail[i] {
                        return (f64::NAN, vec![0.0; n_samples]);
                    }
                    Some(GeneEstWeights::new(w.row(i).to_vec(), bm.row(i).to_vec(), &design))
                }
                _ => None,
            };
            let nf = dds.gene_norm_factors(i).unwrap().to_vec();
            estimate_dispersion_gene(&gene_counts, &nf, &design, xim, n_samples, use_linear_mu, params, gene_weights.as_ref())
        })
        .collect();

    // Separate dispersions and mu values
    let dispersions: Vec<f64> = results.iter().map(|(d, _)| *d).collect();

    // Build mu matrix (n_genes x n_samples) - DESeq2 stores this in assays[["mu"]]
    let mut mu_matrix = Array2::zeros((n_genes, n_samples));
    for (i, (_, mu)) in results.iter().enumerate() {
        for (j, &mu_val) in mu.iter().enumerate() {
            mu_matrix[[i, j]] = mu_val;
        }
    }

    let dispersion_array = Array1::from_vec(dispersions);
    dds.set_gene_dispersions(dispersion_array)?;
    dds.set_mu(mu_matrix)?;
    if let Some(fail) = weights_fail {
        dds.set_weights_fail(fail);
    }

    // Store the design matrix for use in MAP dispersion estimation
    dds.set_design_matrix(design)?;

    Ok(())
}

/// Create design matrix for dispersion estimation
/// Supports both simple two-group and multi-factor designs
///
/// If a design matrix has already been set on the DESeqDataSet (e.g., for blind
/// VST/rlog where we use an intercept-only ~1 design), use that instead of
/// building one from metadata.
fn create_design_matrix_for_dispersion(dds: &DESeqDataSet) -> Result<Array2<f64>> {
    // If design matrix is already set (e.g., for blind VST), use it
    if let Some(dm) = dds.design_matrix() {
        return Ok(dm.clone());
    }

    let metadata = dds.sample_metadata();
    let n_samples = dds.n_samples();
    let factors = dds.factors();
    let continuous = dds.continuous_vars();
    let ref_levels = dds.reference_levels();
    let design_var = dds.design_variable();

    // Check if main factor has more than 2 levels or custom reference level
    let main_levels = metadata.get_levels(design_var)?;
    let has_custom_ref = ref_levels.contains_key(design_var);
    let is_multi_level = main_levels.len() > 2;

    // If only design variable (no batch effects) and simple two-level factor
    // with no custom reference, use simple design matrix
    if factors.is_empty() && continuous.is_empty() && !is_multi_level && !has_custom_ref {
        let conditions: Vec<String> = metadata
            .condition(design_var)
            .cloned()
            .unwrap_or_else(|| vec!["unknown".to_string(); n_samples]);
        return Ok(create_simple_design_matrix(&conditions, n_samples));
    }

    // Multi-factor design: create full design matrix matching DESeq2
    // Formula order: ~ covariate1 + covariate2 + ... + design_variable
    let mut n_cols = 1; // intercept
    let mut factor_info: Vec<(String, Vec<String>, String)> = Vec::new(); // (factor_name, non_ref_levels, ref_level)

    // First process additional factors (batch effects/covariates) in order
    for factor in factors {
        let levels = metadata.get_levels(factor)?;
        let ref_level = ref_levels
            .get(factor)
            .cloned()
            .unwrap_or_else(|| levels[0].clone());

        let non_ref_levels: Vec<String> = levels
            .iter()
            .filter(|l| **l != ref_level)
            .cloned()
            .collect();

        n_cols += non_ref_levels.len();
        factor_info.push((factor.to_string(), non_ref_levels, ref_level));
    }

    // Then add the main design variable (last in formula order)
    {
        let levels = metadata.get_levels(design_var)?;
        let ref_level = ref_levels
            .get(design_var)
            .cloned()
            .unwrap_or_else(|| levels[0].clone());

        let non_ref_levels: Vec<String> = levels
            .iter()
            .filter(|l| **l != ref_level)
            .cloned()
            .collect();

        n_cols += non_ref_levels.len();
        factor_info.push((design_var.to_string(), non_ref_levels, ref_level));
    }

    // Add continuous variables
    n_cols += continuous.len();

    // Build design matrix
    let mut design = Array2::zeros((n_samples, n_cols));

    for i in 0..n_samples {
        let mut col = 0;

        // Intercept
        design[[i, col]] = 1.0;
        col += 1;

        // Factor columns (batch effects first, then main design variable)
        for (factor, non_ref_levels, _ref_level) in &factor_info {
            let sample_value = metadata.get_value(factor, i)?;

            for level in non_ref_levels {
                design[[i, col]] = if sample_value == *level { 1.0 } else { 0.0 };
                col += 1;
            }
        }

        // Continuous columns
        for cont in continuous {
            design[[i, col]] = metadata.get_continuous_value(cont, i)?;
            col += 1;
        }
    }

    check_full_rank(&design)?;
    Ok(design)
}

/// Create a simple design matrix for two-group comparison
fn create_simple_design_matrix(conditions: &[String], n_samples: usize) -> Array2<f64> {
    let mut unique: Vec<String> = conditions.iter().cloned().collect();
    unique.sort_by(|a, b| crate::data::r_collate(a, b));
    unique.dedup();

    // Single level (intercept-only design ~1): return 1-column matrix
    if unique.len() <= 1 {
        let mut design = Array2::zeros((n_samples, 1));
        for i in 0..n_samples {
            design[[i, 0]] = 1.0;
        }
        return design;
    }

    let reference = unique.first().cloned().unwrap_or_default();

    let mut design = Array2::zeros((n_samples, 2));
    for i in 0..n_samples {
        design[[i, 0]] = 1.0; // Intercept
        if conditions[i] != reference {
            design[[i, 1]] = 1.0;
        }
    }
    design
}

/// DESeq2's roughDispEstimate: moment-based dispersion estimator
/// This matches R DESeq2's roughDispEstimate function exactly
/// Formula: alpha = sum(((y - mu)^2 - mu) / mu^2) / (m - p)
fn rough_disp_estimate(
    counts: &[f64],
    size_factors: &[f64],
    design: &Array2<f64>,
) -> f64 {
    let n = counts.len();
    let p = design.ncols();

    // Normalized counts
    let normalized: Vec<f64> = counts
        .iter()
        .zip(size_factors.iter())
        .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.1 })
        .collect();

    // Fit linear model to get mu (DESeq2 uses linearModelMu on normalized counts)
    // This is a weighted least squares fit to get group means
    let mu = linear_model_mu(&normalized, design);

    // Calculate dispersion: sum(((y - mu)^2 - mu) / mu^2) / (m - p)
    let mut sum_term = 0.0;
    for i in 0..n {
        let mu_i = mu[i].max(1.0);  // DESeq2 uses pmax(mu, 1)
        let y = normalized[i];
        sum_term += ((y - mu_i).powi(2) - mu_i) / (mu_i * mu_i);
    }

    let alpha = sum_term / (n - p) as f64;
    alpha.max(0.0)  // DESeq2 uses pmax(est, 0)
}

/// Solve a system of linear equations Ax = b using Gaussian elimination with partial pivoting
/// Returns the solution vector x, or None if the matrix is singular
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }

    // Create augmented matrix [A|b]
    let mut aug: Vec<Vec<f64>> = a.iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.push(b[i]);
            new_row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular matrix
        }

        // Swap rows
        aug.swap(col, max_row);

        // Eliminate
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
}

/// Fit linear model to get mu values (group means for each sample)
/// Matches DESeq2's linearModelMu function - used in roughDispEstimate with pmax(mu, 1)
fn linear_model_mu(normalized: &[f64], design: &Array2<f64>) -> Vec<f64> {
    let n = normalized.len();
    let p = design.ncols();

    // Solve normal equations: (X'X)^-1 X'y
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];

    for i in 0..n {
        for j in 0..p {
            xty[j] += design[[i, j]] * normalized[i];
            for k in 0..p {
                xtx[j][k] += design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Use optimized 2x2 solution for simple designs, general solver for larger
    let beta = if p == 2 {
        let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
        if det.abs() > 1e-10 {
            vec![
                (xtx[1][1] * xty[0] - xtx[0][1] * xty[1]) / det,
                (xtx[0][0] * xty[1] - xtx[1][0] * xty[0]) / det,
            ]
        } else {
            vec![normalized.iter().sum::<f64>() / n as f64, 0.0]
        }
    } else {
        solve_linear_system(&xtx, &xty).unwrap_or_else(|| {
            let mut fallback = vec![0.0; p];
            fallback[0] = normalized.iter().sum::<f64>() / n as f64;
            fallback
        })
    };

    // Calculate fitted mu values
    (0..n)
        .map(|i| {
            let mut mu = 0.0;
            for j in 0..p {
                mu += design[[i, j]] * beta[j];
            }
            mu.max(1.0)  // DESeq2 uses pmax(mu, 1) in roughDispEstimate
        })
        .collect()
}

/// Fit linear model to get mu values for dispersion optimization
/// Matches DESeq2's linearModelMuNormalized - NO pmax clamping here
fn linear_model_mu_for_dispersion(normalized: &[f64], design: &Array2<f64>) -> Vec<f64> {
    let n = normalized.len();
    let p = design.ncols();

    // Solve normal equations: (X'X)^-1 X'y
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];

    for i in 0..n {
        for j in 0..p {
            xty[j] += design[[i, j]] * normalized[i];
            for k in 0..p {
                xtx[j][k] += design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Use optimized 2x2 solution for simple designs, general solver for larger
    let beta = if p == 2 {
        let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
        if det.abs() > 1e-10 {
            vec![
                (xtx[1][1] * xty[0] - xtx[0][1] * xty[1]) / det,
                (xtx[0][0] * xty[1] - xtx[1][0] * xty[0]) / det,
            ]
        } else {
            vec![normalized.iter().sum::<f64>() / n as f64, 0.0]
        }
    } else {
        solve_linear_system(&xtx, &xty).unwrap_or_else(|| {
            let mut fallback = vec![0.0; p];
            fallback[0] = normalized.iter().sum::<f64>() / n as f64;
            fallback
        })
    };

    // Calculate fitted mu values - no clamping for dispersion optimization
    (0..n)
        .map(|i| {
            let mut mu = 0.0;
            for j in 0..p {
                mu += design[[i, j]] * beta[j];
            }
            mu
        })
        .collect()
}

/// Compute mu using linear model (OLS) on normalized counts, matching R's linearModelMuNormalized.
/// mu_normalized = X * (X'X)^{-1} * X' * normalized_counts
/// mu = mu_normalized * size_factors, clamped to minmu (0.5)
///
/// This is used for gene-wise dispersion estimation when the design matrix represents
/// a simple group model (is_linear_mu == true). R's DESeq2 uses this instead of NB IRLS.
fn linear_model_mu_for_gene_dispersion(counts: &[f64], size_factors: &[f64], design: &Array2<f64>) -> Vec<f64> {
    let n = counts.len();
    let p = design.ncols();
    const MINMU: f64 = 0.5;

    // Normalized counts (y)
    let normalized: Vec<f64> = counts.iter().zip(size_factors)
        .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
        .collect();

    // Compute X'X
    let mut xtx = vec![vec![0.0; p]; p];
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                xtx[j][k] += design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Compute X'y
    let mut xty = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            xty[j] += design[[i, j]] * normalized[i];
        }
    }

    // Solve (X'X) beta = X'y
    let beta = if p == 2 {
        let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
        if det.abs() > 1e-10 {
            vec![
                (xtx[1][1] * xty[0] - xtx[0][1] * xty[1]) / det,
                (xtx[0][0] * xty[1] - xtx[1][0] * xty[0]) / det,
            ]
        } else {
            let mean_norm = normalized.iter().sum::<f64>() / n as f64;
            vec![mean_norm, 0.0]
        }
    } else {
        solve_linear_system(&xtx, &xty).unwrap_or_else(|| {
            let mean_norm = normalized.iter().sum::<f64>() / n as f64;
            let mut b = vec![0.0; p];
            b[0] = mean_norm;
            b
        })
    };

    // mu_normalized = X * beta, then mu = mu_normalized * size_factors
    (0..n).map(|i| {
        let mut mu_norm = 0.0;
        for j in 0..p {
            mu_norm += design[[i, j]] * beta[j];
        }
        (mu_norm * size_factors[i]).max(MINMU)
    }).collect()
}

/// Fitted means for gene-wise dispersion estimation when the design is not a
/// simple group model.
/// R (estimateDispersionsGeneEst): fit <- fitNbinomGLMs(object, alpha_hat, modelMatrix)
/// with its defaults (betaTol 1e-8, maxit 100, optim fallback), then
/// fitMu[fitMu < minmu] <- minmu. Reuse the GLM fit rather than a separate IRLS, so
/// the convergence rule, ridge and fallback are the same ones R uses.
fn fit_nb_glm_mu(counts: &[f64], size_factors: &[f64], design: &Array2<f64>, alpha: f64, weights: Option<&[f64]>) -> Vec<f64> {
    const MINMU: f64 = 0.5;
    let fit = crate::glm::fit_single_gene_weighted(
        ndarray::ArrayView1::from(counts),
        design,
        ndarray::ArrayView1::from(size_factors),
        alpha,
        &crate::glm::GlmFitParams::default(),
        weights,
    );
    fit.mu.iter().map(|&m| m.max(MINMU)).collect()
}

/// DESeq2's momentsDispEstimate: variance-based dispersion estimator
/// Formula: (baseVar - xim * baseMean) / baseMean^2
fn moments_disp_estimate(normalized: &[f64], xim: f64) -> f64 {
    let n = normalized.len() as f64;
    let base_mean: f64 = normalized.iter().sum::<f64>() / n;

    // Sample variance (unbiased estimator with n-1 denominator)
    let base_var: f64 = if n > 1.0 {
        normalized.iter().map(|&x| (x - base_mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };

    // momentsDispEstimate = (baseVar - xim * baseMean) / baseMean^2
    if base_mean > 1e-10 {
        (base_var - xim * base_mean) / (base_mean * base_mean)
    } else {
        // R returns Inf for very low expression genes, which gets clamped to maxDisp
        f64::INFINITY
    }
}

/// Estimate gene-wise dispersion using DESeq2's approach:
/// R equivalent: fitDisp() via fitDispWrapper() in core.R
/// 1. Calculate roughDispEstimate and momentsDispEstimate
/// 2. alpha_init = min(roughDisp, momentsDisp)
/// 3. Run MLE optimization starting from alpha_init
/// 4. Apply noIncrease condition: if log-posterior doesn't improve, return alpha_init
/// 5. If line search didn't converge (hit max iterations), fall back to grid search
///
/// Returns (dispersion, mu) where mu is the expected counts for this gene across samples.
/// The mu values are needed for MAP dispersion estimation (DESeq2 reuses them).
///
/// With observation weights, momentsDispEstimate() uses the weighted baseMean and
/// baseVar, mu comes from the weighted GLM (`use_linear_mu` must then be false),
/// and the line search and grid use the weighted log posterior; roughDispEstimate()
/// stays unweighted, as in R.
#[allow(clippy::too_many_arguments)]
pub fn estimate_dispersion_gene(
    counts: &[f64],
    size_factors: &[f64],
    design: &Array2<f64>,
    xim: f64,
    n_samples: usize,
    use_linear_mu: bool,
    params: &DispersionParams,
    weights: Option<&GeneEstWeights>,
) -> (f64, Vec<f64>) {
    // R filters out allZero genes: if all counts are zero, return NaN dispersion
    let all_zero = counts.iter().all(|&c| c == 0.0);
    if all_zero {
        let mu = vec![0.0; n_samples];
        return (f64::NAN, mu);
    }

    let min_disp = params.min_disp;
    let max_disp = (n_samples as f64).max(10.0);
    let maxit = params.maxit;

    // Normalized counts
    let normalized: Vec<f64> = counts
        .iter()
        .zip(size_factors.iter())
        .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.1 })
        .collect();

    // Step 1: Calculate roughDispEstimate (DESeq2's moment-based estimator)
    let rough_disp = rough_disp_estimate(counts, size_factors, design);

    // Step 2: Calculate momentsDispEstimate
    // R: momentsDispEstimate() from mcols baseMean/baseVar (weighted when weights are set)
    let moments_disp = match weights {
        Some(w) => moments_disp_estimate(&w.base_mean_counts, xim),
        None => moments_disp_estimate(&normalized, xim),
    };
    let disp_w = weights.map(|w| &w.disp);

    // Step 3: alpha_init = min(roughDisp, momentsDisp), clamped to [minDisp, maxDisp]
    let alpha_init = rough_disp.min(moments_disp).max(min_disp).min(max_disp);

    // Step 4: Calculate mu
    // When design is a simple group model (use_linear_mu), DESeq2 uses OLS linear model
    // on normalized counts (linearModelMuNormalized) instead of NB IRLS.
    // For non-group designs, NB GLM (IRLS) fitting ensures mu > 0 through log link.
    let mu = if use_linear_mu {
        linear_model_mu_for_gene_dispersion(counts, size_factors, design)
    } else {
        fit_nb_glm_mu(counts, size_factors, design, alpha_init, weights.map(|w| w.glm.as_slice()))
    };

    // Step 5: Optimize dispersion with FIXED mu using DESeq2's Armijo line search
    // Returns (alpha_mle, initial_lp, last_lp, iter_count)
    let (alpha_mle, initial_lp, last_lp, iter_count) = optimize_dispersion_with_mu_deseq2_full(counts, design, &mu, alpha_init, max_disp, params, disp_w);

    // Step 6: Apply noIncrease condition (DESeq2 R code line 828-831)
    // noIncrease <- last_lp < initial_lp + abs(initial_lp)/1e6
    // If noIncrease, return alpha_init instead of alpha_mle
    let no_increase = last_lp < initial_lp + initial_lp.abs() / 1e6;

    let mut result = if no_increase {
        alpha_init
    } else {
        alpha_mle
    };

    // Step 7: DESeq2's refitDisp condition - if line search didn't converge, use grid search
    // dispGeneEstConv <- dispIter < maxit & !(dispIter == 1)
    // refitDisp <- !dispGeneEstConv & dispGeneEst > minDisp*10
    let disp_gene_est_conv = iter_count < maxit && iter_count != 1;
    let refit_disp = !disp_gene_est_conv && result > min_disp * 10.0;

    if refit_disp {
        // Fall back to grid search (DESeq2's fitDispGrid)
        let grid_result = grid_search_dispersion(counts, design, &mu, min_disp, max_disp, disp_w);
        result = grid_result;
    }

    (result.max(min_disp).min(max_disp), mu)
}

/// Grid search for dispersion when line search doesn't converge
/// Matches DESeq2's fitDispGrid function
fn grid_search_dispersion(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    min_disp: f64,
    max_disp: f64,
    weights: Option<&DispWeights>,
) -> f64 {
    let min_log_alpha = min_disp.ln();
    let max_log_alpha = max_disp.ln();
    let n_grid = 20;

    // Create coarse grid
    let delta = (max_log_alpha - min_log_alpha) / (n_grid - 1) as f64;
    let coarse_grid: Vec<f64> = (0..n_grid)
        .map(|i| min_log_alpha + i as f64 * delta)
        .collect();

    // Evaluate log-posterior on coarse grid
    let coarse_lp: Vec<f64> = coarse_grid.iter()
        .map(|&log_alpha| log_posterior_deseq2(counts, design, mu, log_alpha, weights))
        .collect();

    // Find maximum on coarse grid
    let (best_idx, &_best_log_alpha) = coarse_grid.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let lp_a = coarse_lp[coarse_grid.iter().position(|x| x == *a).unwrap()];
            let lp_b = coarse_lp[coarse_grid.iter().position(|x| x == *b).unwrap()];
            lp_a.partial_cmp(&lp_b).unwrap()
        })
        .unwrap();

    let best_coarse_log_alpha = coarse_grid[best_idx];

    // Create fine grid around best coarse value
    let fine_min = best_coarse_log_alpha - delta;
    let fine_max = best_coarse_log_alpha + delta;
    let fine_delta = (fine_max - fine_min) / (n_grid - 1) as f64;
    let fine_grid: Vec<f64> = (0..n_grid)
        .map(|i| fine_min + i as f64 * fine_delta)
        .collect();

    // Evaluate log-posterior on fine grid
    let fine_lp: Vec<f64> = fine_grid.iter()
        .map(|&log_alpha| log_posterior_deseq2(counts, design, mu, log_alpha, weights))
        .collect();

    // Find maximum on fine grid
    let best_fine_idx = fine_lp.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    fine_grid[best_fine_idx].exp()
}


/// Optimize dispersion using DESeq2's exact Armijo line search algorithm
/// This matches the C++ implementation in DESeq2
/// Returns: (alpha_mle, initial_lp, last_lp, iter_count)
fn optimize_dispersion_with_mu_deseq2_full(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    alpha_init: f64,
    max_disp: f64,
    params: &DispersionParams,
    weights: Option<&DispWeights>,
) -> (f64, f64, f64, usize) {
    optimize_dispersion_with_mu_deseq2_internal_full(counts, design, mu, alpha_init, false, max_disp, params, weights)
}

/// Internal optimization function with optional debug output - returns iter_count
#[allow(clippy::too_many_arguments)]
fn optimize_dispersion_with_mu_deseq2_internal_full(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    alpha_init: f64,
    debug: bool,
    max_disp: f64,
    params: &DispersionParams,
    weights: Option<&DispWeights>,
) -> (f64, f64, f64, usize) {
    // DESeq2 parameters (from C++ source), now configurable via params
    let min_log_alpha_hard = -30.0;  // C++ hard bound for proposals
    let min_log_alpha = (params.min_disp / 10.0).ln();  // R uses log(minDisp/10) ≈ -20.72
    let max_log_alpha = max_disp.ln();
    let epsilon = 1.0e-4;
    let maxit = params.maxit;
    let tol = params.disp_tol;
    let kappa_0 = params.kappa_0;

    // Initialize
    let mut log_alpha = alpha_init.max(1e-10).ln().max(min_log_alpha_hard).min(max_log_alpha);
    let initial_lp = log_posterior_deseq2(counts, design, mu, log_alpha, weights);
    let mut lp = initial_lp;
    let mut dlp = d_log_posterior_deseq2_analytical(counts, design, mu, log_alpha, weights);
    let mut kappa = kappa_0;

    if debug {
        eprintln!("  Initial: log_alpha={:.4}, alpha={:.6e}, lp={:.4}, dlp={:.6}",
                  log_alpha, log_alpha.exp(), lp, dlp);
    }

    let mut iter_accept = 0;
    let mut final_iter = maxit;  // Track the final iteration count

    for iter in 0..maxit {
        // DESeq2 C++ style: adjust kappa if proposal goes out of bounds
        // Note: R uses hardcoded bounds of -30.0 and 10.0 for LOG-SCALE proposals
        let a_propose_raw = log_alpha + kappa * dlp;

        if a_propose_raw < min_log_alpha_hard {
            kappa = (min_log_alpha_hard - log_alpha) / dlp;
        }
        if a_propose_raw > 10.0 {
            kappa = (10.0 - log_alpha) / dlp;
        }

        let a_propose = log_alpha + kappa * dlp;

        // Evaluate theta(kappa) = -log_posterior (we minimize theta)
        let lp_propose = log_posterior_deseq2(counts, design, mu, a_propose, weights);
        let theta_kappa = -lp_propose;
        let theta_hat_kappa = -lp - kappa * epsilon * dlp * dlp;

        // Armijo condition: theta_kappa <= theta_hat_kappa
        if theta_kappa <= theta_hat_kappa {
            // Accept step
            iter_accept += 1;
            log_alpha = a_propose;
            let lpnew = lp_propose;
            let change = lpnew - lp;

            if debug {
                eprintln!("  Iter {}: ACCEPT log_alpha={:.4}, alpha={:.6e}, lp={:.4}, change={:.6e}, dlp={:.6}, kappa={:.4}",
                          iter, log_alpha, log_alpha.exp(), lpnew, change, dlp, kappa);
            }

            // Check convergence
            if change < tol {
                lp = lpnew;
                final_iter = iter + 1;
                if debug {
                    eprintln!("  Converged: change < tol");
                }
                break;
            }

            // Check if at minimum bound
            if log_alpha < min_log_alpha {
                final_iter = iter + 1;
                if debug {
                    eprintln!("  At minimum bound");
                }
                break;
            }

            lp = lpnew;
            dlp = d_log_posterior_deseq2_analytical(counts, design, mu, log_alpha, weights);

            // Adjust step size (DESeq2 style)
            kappa = (kappa * 1.1).min(kappa_0);
            if iter_accept % 5 == 0 {
                kappa /= 2.0;
            }
        } else {
            // Reject step, reduce kappa
            if debug && iter < 10 {
                eprintln!("  Iter {}: REJECT, reducing kappa to {:.4}", iter, kappa / 2.0);
            }
            kappa /= 2.0;
        }
    }

    if debug {
        eprintln!("  Final: alpha={:.6e}, initial_lp={:.4}, final_lp={:.4}, iter={}",
                  log_alpha.exp(), initial_lp, lp, final_iter);
    }

    // Note: max_disp will be enforced by caller
    let alpha_mle = log_alpha.exp().max(params.min_disp);
    (alpha_mle, initial_lp, lp, final_iter)
}

/// Analytical derivative of log-posterior w.r.t. log(alpha)
/// Matches DESeq2's C++ implementation exactly
///
/// With observation weights (`useWeights`), each sample's likelihood term is
/// multiplied by its weight and the Cox-Reid term uses only the samples with
/// weight > weightThreshold (its own weights are not multiplied by them).
fn d_log_posterior_deseq2_analytical(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    log_alpha: f64,
    weights: Option<&DispWeights>,
) -> f64 {
    let alpha = log_alpha.exp();
    let alpha_sq = alpha * alpha;
    let n = counts.len();

    let mut dll_dalpha = 0.0;

    for i in 0..n {
        let y = counts[i];
        let mu_i = mu[i].max(1e-10);

        // d/dalpha of lgamma(y + 1/alpha) = digamma(y + 1/alpha) * (-1/alpha^2)
        let term1 = -digamma(y + 1.0 / alpha) / alpha_sq;

        // d/dalpha of -lgamma(1/alpha) = digamma(1/alpha) / alpha^2
        let term2 = digamma(1.0 / alpha) / alpha_sq;

        // d/dalpha of -y*log(mu + 1/alpha) = y / (alpha^2 * (mu + 1/alpha))
        let term3 = y / (alpha_sq * (mu_i + 1.0 / alpha));

        // d/dalpha of -(1/alpha)*log(1 + mu*alpha)
        // = (1/alpha^2)*log(1 + mu*alpha) - mu / (alpha * (1 + mu*alpha))
        let term4 = (1.0 + mu_i * alpha).ln() / alpha_sq - mu_i / (alpha * (1.0 + mu_i * alpha));

        match weights {
            Some(w) => dll_dalpha += w.w[i] * (term1 + term2 + term3 + term4),
            None => dll_dalpha += term1 + term2 + term3 + term4,
        }
    }

    // Convert to d/d(log_alpha) by multiplying by alpha
    let dll_dlog_alpha = dll_dalpha * alpha;

    let dcr_dalpha = match weights {
        Some(w) => d_cox_reid_dalpha(&w.cr_design, &w.cr_mu(mu), alpha),
        None => d_cox_reid_dalpha(design, mu, alpha),
    };

    // Convert CR derivative to d/d(log_alpha) by multiplying by alpha
    // This matches R's formula: return (ll_part + cr_term) * alpha
    let dcr_dlog_alpha = dcr_dalpha * alpha;

    dll_dlog_alpha + dcr_dlog_alpha
}

/// Derivative w.r.t. alpha of the Cox-Reid term -0.5 * log|X'WX|, w_i = 1/(1/mu_i + alpha)
/// R: the `cr_term` of DESeq2.cpp `dlog_posterior`
fn d_cox_reid_dalpha(design: &Array2<f64>, mu: &[f64], alpha: f64) -> f64 {
    let n = mu.len();

    // Analytical Cox-Reid derivative matching DESeq2's C++ implementation
    // CR term = -0.5 * log|X'WX| where w_i = 1/(1/mu_i + alpha)
    // d(CR)/dalpha = -0.5 * tr(B^{-1} * dB/dalpha) where B = X'WX
    // dw_i/dalpha = -w_i^2, so dB/dalpha = X' * diag(dw) * X
    let p = design.ncols();

    if p == 2 {
        // Optimized 2x2 path (most common: simple two-condition design)
        let mut b = [[0.0f64; 2]; 2];
        let mut db = [[0.0f64; 2]; 2];
        for i in 0..n {
            let mu_i = mu[i].max(1e-10);
            let w_i = 1.0 / (1.0 / mu_i + alpha);
            let dw_i = -w_i * w_i; // dw/dalpha = -1/(1/mu + alpha)^2 = -w^2
            for j in 0..2 {
                for k in 0..2 {
                    let xjxk = design[[i, j]] * design[[i, k]];
                    b[j][k] += w_i * xjxk;
                    db[j][k] += dw_i * xjxk;
                }
            }
        }
        let det_b = b[0][0] * b[1][1] - b[0][1] * b[1][0];
        // R: -0.5 * det(b) * trace(b^-1 db) / det(b), no cutoff on det(b)
        if det_b != 0.0 {
            // tr(B^{-1} dB) = (B[1][1]*dB[0][0] + B[0][0]*dB[1][1]
            //                  - B[0][1]*dB[1][0] - B[1][0]*dB[0][1]) / det(B)
            let tr_binv_db = (b[1][1] * db[0][0] + b[0][0] * db[1][1]
                - b[0][1] * db[1][0] - b[1][0] * db[0][1])
                / det_b;
            -0.5 * tr_binv_db
        } else {
            0.0
        }
    } else {
        // General p > 2 case
        let mut b = vec![vec![0.0f64; p]; p];
        let mut db = vec![vec![0.0f64; p]; p];
        for i in 0..n {
            let mu_i = mu[i].max(1e-10);
            let w_i = 1.0 / (1.0 / mu_i + alpha);
            let dw_i = -w_i * w_i;
            for j in 0..p {
                for k in 0..p {
                    let xjxk = design[[i, j]] * design[[i, k]];
                    b[j][k] += w_i * xjxk;
                    db[j][k] += dw_i * xjxk;
                }
            }
        }
        // Compute tr(B^{-1} * dB) by solving B * X = dB column by column
        // tr(B^{-1} * dB) = sum of diagonal elements of B^{-1} * dB
        // We solve B * x_col = db_col for each column, then sum x_col[col]
        let mut trace = 0.0;
        for col in 0..p {
            let db_col: Vec<f64> = (0..p).map(|row| db[row][col]).collect();
            if let Some(x_col) = solve_linear_system(&b, &db_col) {
                trace += x_col[col];
            }
            // If singular, contribution is 0
        }
        -0.5 * trace
    }
}

/// DESeq2's log-posterior function (without prior for gene estimation)
///
/// With observation weights the likelihood is `sum(w * ll_i)` and the Cox-Reid
/// term uses only the samples with weight > weightThreshold (DESeq2.cpp
/// `log_posterior` with `useWeights`).
fn log_posterior_deseq2(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    log_alpha: f64,
    obs_weights: Option<&DispWeights>,
) -> f64 {
    let n = counts.len();
    let alpha = log_alpha.exp();
    let alpha_inv = 1.0 / alpha;

    if let Some(w) = obs_weights {
        let mut ll_part = 0.0;
        for i in 0..n {
            let y = counts[i];
            let mu_i = mu[i].max(1e-10);
            let ll_i = lgamma(y + alpha_inv) - lgamma(alpha_inv) - y * (mu_i + alpha_inv).ln()
                - alpha_inv * (1.0 + mu_i * alpha).ln();
            ll_part += w.w[i] * ll_i;
        }
        let cr_mu = w.cr_mu(mu);
        let cr_w: Vec<f64> = cr_mu.iter().map(|&m| 1.0 / (1.0 / m.max(1e-10) + alpha)).collect();
        return ll_part + cox_reid_term(&w.cr_design, &cr_w);
    }

    let mut ll_part = 0.0;
    let mut weights = vec![0.0; n];

    for i in 0..n {
        let y = counts[i];
        let mu_i = mu[i].max(1e-10);

        // DESeq2's exact log-likelihood formula:
        // ll = lgamma(y + 1/alpha) - lgamma(1/alpha) - y*log(mu + 1/alpha) - (1/alpha)*log(1 + mu*alpha)
        // Evaluate each sample's term first, then add it, as R's vectorised
        // sum(lgamma(y + 1/a) - lgamma(1/a) - y*log(mu + 1/a) - 1/a*log(1 + mu*a)) does;
        // accumulating the pieces separately rounds differently, which matters where
        // the line search's Armijo test is decided at rounding level (lp ~ 1e5).
        ll_part += lgamma(y + alpha_inv) - lgamma(alpha_inv) - y * (mu_i + alpha_inv).ln()
            - alpha_inv * (1.0 + mu_i * alpha).ln();

        // Weight for Cox-Reid adjustment: w = 1 / (1/mu + alpha)
        weights[i] = 1.0 / (1.0 / mu_i + alpha);
    }

    ll_part + cox_reid_term(design, &weights)
}

/// Cox-Reid adjustment -0.5 * log|X'WX| for GLM weights `weights` (w_i = 1/(1/mu_i + alpha))
fn cox_reid_term(design: &Array2<f64>, weights: &[f64]) -> f64 {
    let n = weights.len();
    let p = design.ncols();

    // Cox-Reid adjustment: -0.5 * log|X'WX|  (DESeq2.cpp: cr_term = -0.5 * log(det(b)))
    // R applies no cutoff. Cutting the term off below det = 1e-10 made the
    // log-posterior jump wherever many samples have small weights, e.g. 11 groups of
    // 3 at mu = minmu gives det = (3/(2 + alpha))^11 < 1e-10 for alpha > ~22, which
    // stopped the search there instead of at R's maxDisp.
    let log_det = if p == 2 {
        let mut xtwx = [[0.0; 2]; 2];
        for i in 0..n {
            for j in 0..2 {
                for k in 0..2 {
                    xtwx[j][k] += weights[i] * design[[i, j]] * design[[i, k]];
                }
            }
        }
        let det = xtwx[0][0] * xtwx[1][1] - xtwx[0][1] * xtwx[1][0];
        if det > 0.0 { Some(det.ln()) } else { None }
    } else {
        let mut xtwx = vec![vec![0.0; p]; p];
        for i in 0..n {
            for j in 0..p {
                for k in 0..p {
                    xtwx[j][k] += weights[i] * design[[i, j]] * design[[i, k]];
                }
            }
        }
        log_determinant_positive(&xtwx)
    };
    match log_det {
        Some(ld) => -0.5 * ld,
        None => 0.0,
    }
}

/// log(det(A)) for a matrix with positive determinant, from the LU pivots
/// (summing logs avoids the underflow of multiplying many small pivots);
/// `None` when A is singular or its determinant is not positive.
fn log_determinant_positive(a: &[Vec<f64>]) -> Option<f64> {
    let n = a.len();
    let mut lu: Vec<Vec<f64>> = a.to_vec();
    let mut sign = 1.0;
    let mut log_det = 0.0;
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if lu[row][col].abs() > lu[max_row][col].abs() {
                max_row = row;
            }
        }
        if lu[max_row][col] == 0.0 {
            return None;
        }
        if max_row != col {
            lu.swap(col, max_row);
            sign = -sign;
        }
        let pivot = lu[col][col];
        if pivot < 0.0 {
            sign = -sign;
        }
        log_det += pivot.abs().ln();
        for row in (col + 1)..n {
            let factor = lu[row][col] / pivot;
            for j in col..n {
                lu[row][j] -= factor * lu[col][j];
            }
        }
    }
    if sign > 0.0 { Some(log_det) } else { None }
}

/// Digamma function
fn digamma(x: f64) -> f64 {
    statrs::function::gamma::digamma(x)
}

/// Log-gamma function
fn lgamma(x: f64) -> f64 {
    // R's lgammafn: statrs's Lanczos ln_gamma loses ~1e-6 absolute in
    // lgamma(y + 1/alpha) - lgamma(1/alpha) when 1/alpha ~ 1e8, enough to change
    // which point of the flat low-dispersion end of the grid search wins.
    crate::optim::nmath::lgammafn(x)
}

/// Debug function to trace optimization for specific genes
pub fn debug_gene_dispersion(
    counts: &[f64],
    size_factors: &[f64],
    conditions: &[String],
    gene_id: &str,
) {
    let n_samples = counts.len();
    let design = create_simple_design_matrix(conditions, n_samples);

    // Pre-compute xim
    let xim: f64 = size_factors.iter().map(|&s| 1.0 / s.max(1e-10)).sum::<f64>() / n_samples as f64;

    // Normalized counts
    let normalized: Vec<f64> = counts
        .iter()
        .zip(size_factors.iter())
        .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.1 })
        .collect();

    eprintln!("\n=== Debugging gene: {} ===", gene_id);
    eprintln!("Counts: {:?}", counts);
    eprintln!("Size factors: {:?}", size_factors);
    eprintln!("Normalized: {:?}", normalized);

    // Step 1: Calculate roughDispEstimate
    let rough_disp = rough_disp_estimate(counts, size_factors, &design);
    eprintln!("roughDispEstimate: {:.8}", rough_disp);

    // Step 2: Calculate momentsDispEstimate
    let moments_disp = moments_disp_estimate(&normalized, xim);
    eprintln!("momentsDispEstimate: {:.8}", moments_disp);

    // Step 3: alpha_init
    let max_disp = (n_samples as f64).max(10.0);
    let alpha_init = rough_disp.min(moments_disp).max(1e-8).min(max_disp);
    eprintln!("max_disp: {:.8}", max_disp);
    eprintln!("alpha_init: {:.8}", alpha_init);
    eprintln!("log(alpha_init): {:.4}", alpha_init.ln());

    // Calculate mu using linear model (group means) - matching DESeq2 exactly
    let linear_mu = linear_model_mu_for_dispersion(&normalized, &design);
    let mu: Vec<f64> = (0..n_samples)
        .map(|i| linear_mu[i] * size_factors[i])
        .collect();

    eprintln!("linear_mu (normalized): {:?}", linear_mu);
    eprintln!("mu: {:?}", mu);

    // Run optimization with debug output (uses default params for debug)
    let debug_params = DispersionParams::default();
    eprintln!("\nOptimization trace:");
    let (_alpha_mle, initial_lp, last_lp, iter_count) = optimize_dispersion_with_mu_deseq2_internal_full(
        counts, &design, &mu, alpha_init, true, max_disp, &debug_params, None
    );

    // Check noIncrease condition
    let no_increase = last_lp < initial_lp + initial_lp.abs() / 1e6;
    eprintln!("\nnoIncrease condition:");
    eprintln!("  last_lp={:.6}, initial_lp={:.6}", last_lp, initial_lp);
    eprintln!("  threshold: initial_lp + |initial_lp|/1e6 = {:.6}", initial_lp + initial_lp.abs() / 1e6);
    eprintln!("  noIncrease: {}", no_increase);

    // Check refitDisp condition
    let min_disp = debug_params.min_disp;
    let maxit = debug_params.maxit;
    let disp_gene_est_conv = iter_count < maxit && iter_count != 1;
    let current_result = if no_increase { alpha_init } else { _alpha_mle };
    let refit_disp = !disp_gene_est_conv && current_result > min_disp * 10.0;
    eprintln!("\nrefitDisp condition:");
    eprintln!("  iter_count={}, maxit={}", iter_count, maxit);
    eprintln!("  dispGeneEstConv: {} (iter < maxit && iter != 1)", disp_gene_est_conv);
    eprintln!("  current_result: {:.6e}", current_result);
    eprintln!("  refitDisp: {}", refit_disp);

    if refit_disp {
        eprintln!("\nRunning grid search fallback...");
        let grid_result = grid_search_dispersion(counts, &design, &mu, min_disp, max_disp, None);
        eprintln!("Grid search result: {:.6e}", grid_result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, SampleMetadata};
    use crate::normalization::{estimate_size_factors, SizeFactorMethod};
    use ndarray::array;

    #[test]
    fn test_debug_gene_0201() {
        // Data from R debug output
        let counts = vec![25.0, 24.0, 25.0, 10.0, 20.0, 11.0];
        let size_factors = vec![1.0223, 1.0194, 1.052, 1.0169, 0.9944, 1.0543];
        let conditions = vec![
            "control".to_string(),
            "control".to_string(),
            "control".to_string(),
            "treated".to_string(),
            "treated".to_string(),
            "treated".to_string(),
        ];

        debug_gene_dispersion(&counts, &size_factors, &conditions, "gene_0201");
    }

    #[test]
    fn test_debug_gene_0317() {
        // Data from R debug output
        let counts = vec![34.0, 32.0, 35.0, 13.0, 5.0, 15.0];
        let size_factors = vec![1.0223, 1.0194, 1.052, 1.0169, 0.9944, 1.0543];
        let conditions = vec![
            "control".to_string(),
            "control".to_string(),
            "control".to_string(),
            "treated".to_string(),
            "treated".to_string(),
            "treated".to_string(),
        ];

        debug_gene_dispersion(&counts, &size_factors, &conditions, "gene_0317");
    }

    #[test]
    fn test_gene_wise_dispersion() {
        let counts = CountMatrix::new(
            array![
                [100.0, 120.0, 90.0, 110.0, 95.0, 105.0],
                [500.0, 550.0, 480.0, 520.0, 490.0, 510.0],
                [50.0, 45.0, 55.0, 48.0, 52.0, 50.0]
            ],
            vec!["gene1".to_string(), "gene2".to_string(), "gene3".to_string()],
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
                "condition",
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                ],
            )
            .unwrap();

        let mut dds = DESeqDataSet::new(counts, metadata, "condition").unwrap();
        estimate_size_factors(&mut dds, SizeFactorMethod::Ratio).unwrap();
        estimate_gene_dispersions(&mut dds, &DispersionParams::default()).unwrap();

        let dispersions = dds.gene_dispersions().unwrap();
        assert_eq!(dispersions.len(), 3);
        assert!(dispersions.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_max_disp_formula() {
        // Test that max_disp = max(n_samples, 10) is used correctly
        // For 6 samples: max_disp should be 10.0
        // For 15 samples: max_disp should be 15.0
        // For 20 samples: max_disp should be 20.0

        // Test with 6 samples (max_disp should be 10.0)
        let counts = vec![25.0, 24.0, 25.0, 10.0, 20.0, 11.0];
        let size_factors = vec![1.0223, 1.0194, 1.052, 1.0169, 0.9944, 1.0543];
        let conditions = vec![
            "control".to_string(),
            "control".to_string(),
            "control".to_string(),
            "treated".to_string(),
            "treated".to_string(),
            "treated".to_string(),
        ];
        let design = create_simple_design_matrix(&conditions, 6);
        let xim: f64 = size_factors.iter().map(|&s: &f64| 1.0 / s.max(1e-10)).sum::<f64>() / 6.0;

        let use_linear = is_linear_mu(&design);
        let params = DispersionParams::default();
        let (disp, _mu) = estimate_dispersion_gene(&counts, &size_factors, &design, xim, 6, use_linear, &params, None);
        // Dispersion should be clamped to max_disp = max(6, 10) = 10.0
        assert!(disp <= 10.0);

        // Test with 15 samples (max_disp should be 15.0)
        let counts_15 = vec![25.0; 15];
        let size_factors_15 = vec![1.0; 15];
        let mut conditions_15 = vec!["A".to_string(); 8];
        conditions_15.extend(vec!["B".to_string(); 7]);
        let design_15 = create_simple_design_matrix(&conditions_15, 15);
        let xim_15: f64 = size_factors_15.iter().map(|&s: &f64| 1.0 / s.max(1e-10)).sum::<f64>() / 15.0;

        let use_linear_15 = is_linear_mu(&design_15);
        let (disp_15, _mu_15) = estimate_dispersion_gene(&counts_15, &size_factors_15, &design_15, xim_15, 15, use_linear_15, &params, None);
        // Dispersion should be clamped to max_disp = max(15, 10) = 15.0
        assert!(disp_15 <= 15.0);

        // Verify the formula: max_disp should be max(n_samples, 10)
        assert_eq!((6_f64).max(10.0), 10.0);
        assert_eq!((15_f64).max(10.0), 15.0);
        assert_eq!((20_f64).max(10.0), 20.0);
    }
}
