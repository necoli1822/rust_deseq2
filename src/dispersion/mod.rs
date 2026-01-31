//! Dispersion estimation for negative binomial models

mod gene_wise;
mod map;
mod trend;

pub use gene_wise::{estimate_gene_dispersions, debug_gene_dispersion, estimate_dispersion_gene};
pub use map::{estimate_map_dispersions, fit_map_dispersion, estimate_prior_variance_with_var};
pub use trend::{fit_dispersion_trend, TrendFitMethod};

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};
use crate::glm::{fit_single_gene, GlmFitParams, GlmFitResult};

/// Configurable parameters for dispersion estimation.
/// R equivalent: parameters of estimateDispersionsGeneEst() and estimateDispersionsMAP() in core.R
#[derive(Debug, Clone)]
pub struct DispersionParams {
    /// Minimum dispersion value. R: estimateDispersionsGeneEst(minDisp=1e-8)
    pub min_disp: f64,
    /// Dispersion convergence tolerance. R: estimateDispersionsGeneEst(dispTol=1e-6)
    pub disp_tol: f64,
    /// Initial step size for line search. R: estimateDispersionsGeneEst(kappa_0=1)
    pub kappa_0: f64,
    /// Maximum iterations for dispersion optimization. R: estimateDispersionsGeneEst(maxit=100)
    pub maxit: usize,
    /// Outlier SD threshold for MAP shrinkage. R: estimateDispersionsMAP(outlierSD=2)
    pub outlier_sd: f64,
}

impl Default for DispersionParams {
    fn default() -> Self {
        Self {
            min_disp: 1e-8,
            disp_tol: 1e-6,
            kappa_0: 1.0,
            maxit: 100,
            outlier_sd: 2.0,
        }
    }
}

/// Estimate all dispersions (gene-wise, trended, and MAP)
/// R equivalent: `estimateDispersions()` in methods.R
pub fn estimate_dispersions(dds: &mut DESeqDataSet, fit_type: TrendFitMethod, params: &DispersionParams) -> Result<()> {
    // R: checkForExperimentalReplicates — if design matrix has as many or more
    // columns as rows, there are no replicates for dispersion estimation.
    if let Some(dm) = dds.design_matrix() {
        if dm.nrows() <= dm.ncols() {
            return Err(DeseqError::InvalidInput {
                reason: "design formula has as many or more parameters than samples; no replicates for dispersion estimation".to_string(),
            });
        }
    }

    // Step 1: Gene-wise MLE
    estimate_gene_dispersions(dds, params)?;

    // Step 2: Trend fitting
    fit_dispersion_trend(dds, fit_type)?;

    // Step 3: MAP shrinkage
    estimate_map_dispersions(dds, params)?;

    Ok(())
}

/// Refit only the flagged genes (R's refitWithoutOutliers).
/// R equivalent: `refitWithoutOutliers()` in core.R
/// Matches R DESeq2 behavior:
/// 1. Re-estimate gene-wise dispersions for ONLY flagged genes
/// 2. Reuse the ORIGINAL dispersion trend function and prior variance
/// 3. Re-estimate MAP dispersions for ONLY flagged genes
/// 4. Refit GLM for ONLY flagged genes
/// 5. Write results back into the corresponding rows of the full DDS
pub fn refit_without_outliers(
    dds: &mut DESeqDataSet,
    flagged_gene_indices: &[usize],
) -> Result<()> {
    use rayon::prelude::*;

    if flagged_gene_indices.is_empty() {
        return Ok(());
    }

    let n_samples = dds.n_samples();

    // Get required data
    let counts = dds.counts().counts().to_owned();
    let size_factors = dds.size_factors().ok_or_else(|| DeseqError::DispersionEstimationFailed {
        gene_id: "N/A".to_string(),
        reason: "Size factors required for refit".to_string(),
    })?.to_owned();
    let design = dds.design_matrix().ok_or_else(|| DeseqError::DispersionEstimationFailed {
        gene_id: "N/A".to_string(),
        reason: "Design matrix required for refit".to_string(),
    })?.clone();

    // Get ORIGINAL dispersion function (trend coefficients) - DO NOT re-estimate
    let disp_fn = dds.dispersion_function().ok_or_else(|| DeseqError::DispersionEstimationFailed {
        gene_id: "N/A".to_string(),
        reason: "Dispersion function required for refit".to_string(),
    })?;

    // Get ORIGINAL prior variance - DO NOT re-estimate
    let prior_var = dds.dispersion_prior_var().ok_or_else(|| DeseqError::DispersionEstimationFailed {
        gene_id: "N/A".to_string(),
        reason: "Dispersion prior variance required for refit. Run full estimate_dispersions first.".to_string(),
    })?;

    let sf_slice = size_factors.as_slice().unwrap();
    let xim: f64 = sf_slice.iter().map(|&s| 1.0 / s.max(1e-10)).sum::<f64>() / n_samples as f64;

    // Check if design supports linear mu
    let use_linear_mu = {
        let n = design.nrows();
        let p = design.ncols();
        let mut unique_rows: Vec<Vec<i64>> = Vec::new();
        for i in 0..n {
            let row: Vec<i64> = (0..p).map(|j| (design[[i, j]] * 1000.0).round() as i64).collect();
            if !unique_rows.contains(&row) {
                unique_rows.push(row);
            }
        }
        unique_rows.len() == p
    };

    let max_disp = (n_samples as f64).max(10.0);

    // Step 1: Re-estimate gene-wise dispersions for ONLY flagged genes
    log::info!("Refitting {} flagged genes (gene-wise dispersions)...", flagged_gene_indices.len());
    let gene_results: Vec<(usize, f64, Vec<f64>)> = flagged_gene_indices
        .par_iter()
        .map(|&gene_idx| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[gene_idx, j]]).collect();
            let (disp, mu) = estimate_dispersion_gene(
                &gene_counts, sf_slice, &design, xim, n_samples, use_linear_mu, &DispersionParams::default()
            );
            (gene_idx, disp, mu)
        })
        .collect();

    // Write gene-wise results back
    let mut gene_dispersions = dds.gene_dispersions().unwrap().to_owned();
    let mut mu_matrix = dds.mu().unwrap().to_owned();
    for &(gene_idx, disp, ref mu) in &gene_results {
        gene_dispersions[gene_idx] = disp;
        for (j, &mu_val) in mu.iter().enumerate() {
            mu_matrix[[gene_idx, j]] = mu_val;
        }
    }
    dds.set_gene_dispersions(gene_dispersions)?;
    dds.set_mu(mu_matrix)?;

    // Step 2: Compute trended dispersions using ORIGINAL dispersion function
    // disp(mean) = asymptDisp + extraPois / mean
    let (asympt_disp, extra_pois) = disp_fn;
    let base_means: Vec<f64> = flagged_gene_indices.iter().map(|&gene_idx| {
        let norm_counts: Vec<f64> = (0..n_samples)
            .map(|j| if sf_slice[j] > 0.0 { counts[[gene_idx, j]] / sf_slice[j] } else { 0.0 })
            .collect();
        norm_counts.iter().sum::<f64>() / n_samples as f64
    }).collect();

    let mut trended = dds.trended_dispersions().unwrap().to_owned();
    for (idx, &gene_idx) in flagged_gene_indices.iter().enumerate() {
        let mean = base_means[idx].max(1e-10);
        trended[gene_idx] = asympt_disp + extra_pois / mean;
    }
    dds.set_trended_dispersions(trended)?;

    // Step 3: Re-estimate MAP dispersions for ONLY flagged genes using ORIGINAL prior_var
    log::info!("Refitting {} flagged genes (MAP dispersions)...", flagged_gene_indices.len());
    let gene_dispersions = dds.gene_dispersions().unwrap();
    let trended_dispersions = dds.trended_dispersions().unwrap();
    let mu_matrix_ref = dds.mu().unwrap();

    let map_results: Vec<(usize, f64)> = flagged_gene_indices
        .par_iter()
        .map(|&gene_idx| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[gene_idx, j]]).collect();
            let gene_disp = gene_dispersions[gene_idx];
            let trend_disp = trended_dispersions[gene_idx];
            let mu: Vec<f64> = (0..n_samples).map(|j| mu_matrix_ref[[gene_idx, j]]).collect();

            let map_disp = fit_map_dispersion(
                &gene_counts, &design, &mu, gene_disp, trend_disp, prior_var, max_disp
            );
            (gene_idx, map_disp)
        })
        .collect();

    // Apply outlier detection for flagged genes using varLogDispEsts from full dataset
    let n_coef = design.ncols();
    let (_, var_log_disp_ests) = estimate_prior_variance_with_var(
        gene_dispersions.as_slice().unwrap(),
        trended_dispersions.as_slice().unwrap(),
        n_samples,
        n_coef,
    );
    let outlier_threshold = 2.0 * var_log_disp_ests.sqrt();

    let mut map_dispersions = dds.map_dispersions().unwrap().to_owned();
    for &(gene_idx, map_disp) in &map_results {
        let gene_disp = gene_dispersions[gene_idx];
        let trend_disp = trended_dispersions[gene_idx];

        // Apply outlier detection (same as in estimate_map_dispersions)
        if gene_disp.is_finite() && trend_disp.is_finite() && gene_disp > 0.0 && trend_disp > 0.0 {
            let log_diff = gene_disp.ln() - trend_disp.ln();
            if log_diff > outlier_threshold {
                map_dispersions[gene_idx] = gene_disp;
            } else {
                map_dispersions[gene_idx] = map_disp;
            }
        } else {
            map_dispersions[gene_idx] = map_disp;
        }
    }
    dds.set_map_dispersions(map_dispersions)?;

    // Step 4: Refit GLM for ONLY flagged genes
    log::info!("Refitting {} flagged genes (GLM)...", flagged_gene_indices.len());
    let dispersions = dds.dispersions().unwrap().to_owned();
    let n_coefs = design.ncols();

    let glm_results: Vec<(usize, GlmFitResult)> = flagged_gene_indices
        .par_iter()
        .map(|&gene_idx| {
            let gene_counts = counts.row(gene_idx);
            let alpha = dispersions[gene_idx];
            let result = fit_single_gene(gene_counts, &design, size_factors.view(), alpha, &GlmFitParams::default());
            (gene_idx, result)
        })
        .collect();

    // Write GLM results back
    let mut coefficients = dds.coefficients().unwrap().to_owned();
    let mut standard_errors = dds.standard_errors().unwrap().to_owned();
    let mut covariances = dds.covariances().unwrap().to_owned();
    let mut hat_diagonals = dds.hat_diagonals().unwrap().to_owned();
    let mut mu_final = dds.mu().unwrap().to_owned();

    // Collect convergence status and deviance from refit results
    let mut refit_convergence: Vec<(usize, bool)> = Vec::new();
    let mut deviance = dds.deviance().map(|d| d.to_owned()).unwrap_or_else(|| {
        use ndarray::Array1;
        Array1::zeros(dds.n_genes())
    });

    for (gene_idx, result) in glm_results {
        refit_convergence.push((gene_idx, result.converged));
        for j in 0..n_coefs {
            coefficients[[gene_idx, j]] = result.coefficients[j];
            standard_errors[[gene_idx, j]] = result.standard_errors[j];
            for k in 0..n_coefs {
                covariances[[gene_idx, j, k]] = result.covariance[j * n_coefs + k];
            }
        }
        for j in 0..n_samples {
            hat_diagonals[[gene_idx, j]] = result.hat_diagonals[j];
            mu_final[[gene_idx, j]] = result.mu[j];
        }
        // Update deviance for refitted gene
        deviance[gene_idx] = -2.0 * result.log_likelihood;
    }

    dds.set_coefficients(coefficients)?;
    dds.set_standard_errors(standard_errors)?;
    dds.set_covariances(covariances)?;
    dds.set_hat_diagonals(hat_diagonals)?;
    dds.set_mu(mu_final)?;
    dds.set_deviance(deviance)?;

    // Update convergence status for refitted genes
    if let Some(existing_converged) = dds.converged() {
        let mut converged = existing_converged.clone();
        for (gene_idx, conv) in refit_convergence {
            converged[gene_idx] = conv;
        }
        dds.set_converged(converged);
    }

    Ok(())
}
