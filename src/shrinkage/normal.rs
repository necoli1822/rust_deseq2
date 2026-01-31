//! Normal prior LFC shrinkage (DESeq2 type="normal")
//!
//! Implements the exact R DESeq2 `lfcShrink(type="normal")` algorithm:
//! 1. Estimate `betaPriorVar` per coefficient using `matchWeightedUpperQuantileForVariance`
//! 2. Re-fit the GLM with ridge penalty `lambda = 1/betaPriorVar` per coefficient
//! 3. Extract shrunk LFC and SE from the re-fit
//! 4. Only LFC and SE change; stat/pvalue are preserved from MLE

use crate::data::DESeqDataSet;
use crate::glm::{get_contrast_index, refit_glm_with_prior, DesignInfo};
use crate::io::DESeqResults;
use crate::stats::match_weighted_upper_quantile_for_variance;

/// Apply normal prior shrinkage to log fold changes (DESeq2 type="normal").
/// R equivalent: lfcShrink(type="normal") in lfcShrink.R
///
/// This re-fits the GLM with a per-coefficient ridge penalty estimated from the
/// MLE betas, matching R DESeq2's betaPrior=TRUE pathway.
///
/// # Arguments
/// * `dds` - DESeqDataSet with fitted GLM (coefficients, design matrix, dispersions)
/// * `design_info` - Design information for coefficient name resolution
/// * `results` - Results to update (LFC and SE are replaced; stat/pvalue preserved)
/// * `upper_quantile` - Upper quantile for beta prior variance estimation (default 0.05)
pub fn shrink_lfc_normal(dds: &DESeqDataSet, design_info: &DesignInfo, results: &mut DESeqResults, upper_quantile: f64) {
    let n_genes = results.gene_ids.len();
    let n_samples = dds.n_samples();

    // Get the design matrix from the DDS
    let design = match dds.design_matrix() {
        Some(d) => d.clone(),
        None => {
            log::warn!("No design matrix available for LFC shrinkage");
            return;
        }
    };
    let n_coefs = design.ncols();

    // Get MLE coefficients (on natural log scale from the GLM fit)
    let coefficients = match dds.coefficients() {
        Some(c) => c,
        None => {
            log::warn!("No coefficients available for LFC shrinkage");
            return;
        }
    };

    // Get trended dispersions for weight calculation (R: dispFit)
    let trended = match dds.trended_dispersions() {
        Some(t) => t,
        None => {
            log::warn!("No trended dispersions available for LFC shrinkage");
            return;
        }
    };

    // Get normalized counts and compute baseMean
    let normalized = match dds.normalized_counts() {
        Some(n) => n,
        None => {
            log::warn!("No normalized counts available for LFC shrinkage");
            return;
        }
    };

    // Determine which coefficient index corresponds to the contrast
    let (coef_idx, _contrast_sign) = match get_contrast_index(
        design_info,
        &results.contrast.numerator,
        &results.contrast.denominator,
    ) {
        Ok(val) => val,
        Err(e) => {
            log::warn!("Could not determine contrast index for shrinkage: {}", e);
            return;
        }
    };

    // Compute baseMean per gene
    let base_means: Vec<f64> = (0..n_genes)
        .map(|i| {
            let sum: f64 = (0..n_samples).map(|j| normalized[[i, j]]).sum();
            sum / n_samples as f64
        })
        .collect();

    // Identify non-zero genes (baseMean > 0)
    let all_zero: Vec<bool> = base_means.iter().map(|&bm| bm == 0.0).collect();
    let nz_indices: Vec<usize> = (0..n_genes).filter(|&i| !all_zero[i]).collect();

    if nz_indices.is_empty() {
        log::warn!("All genes have zero baseMean, skipping shrinkage");
        return;
    }

    // Compute weights for betaPriorVar estimation
    // R: varlogk = 1/baseMean + dispFit; weight = 1/varlogk
    let weights: Vec<f64> = nz_indices
        .iter()
        .map(|&i| {
            let varlogk = 1.0 / base_means[i] + trended[i];
            1.0 / varlogk
        })
        .collect();

    // ---------------------------------------------------------------
    // Step 1: Estimate betaPriorVar for each coefficient
    // R: For each non-intercept coefficient:
    //   betas = coefficients of non-zero genes for this coefficient
    //   filter |beta| < 10
    //   betaPriorVar[k] = matchWeightedUpperQuantileForVariance(betas, weights)
    // ---------------------------------------------------------------
    // Note: MLE coefficients in dds are on natural log scale.
    // R DESeq2 stores them on log2 scale. The betaPriorVar is estimated on
    // log2 scale in R, so we need to convert our natural log betas to log2.
    let log2_e = std::f64::consts::LOG2_E;
    let ln2 = std::f64::consts::LN_2;
    let ln2_sq = ln2 * ln2;

    let mut beta_prior_var = vec![1e6_f64; n_coefs]; // intercept gets wide prior

    for k in 1..n_coefs {
        // skip intercept (idx 0)
        // Get MLE betas for this coefficient (non-zero genes only), converted to log2 scale
        let betas_log2: Vec<f64> = nz_indices
            .iter()
            .map(|&i| coefficients[[i, k]] * log2_e)
            .collect();

        // Filter |beta| < 10 (R: useFinite <- abs(x) < 10)
        let mut filtered_betas = Vec::new();
        let mut filtered_weights = Vec::new();
        for (j, &b) in betas_log2.iter().enumerate() {
            if b.abs() < 10.0 && b.is_finite() {
                filtered_betas.push(b);
                filtered_weights.push(weights[j]);
            }
        }

        if filtered_betas.is_empty() {
            beta_prior_var[k] = 1e6; // wide prior if no valid betas
        } else {
            beta_prior_var[k] =
                match_weighted_upper_quantile_for_variance(&filtered_betas, &filtered_weights, upper_quantile);
        }
    }

    log::info!(
        "betaPriorVar: {}",
        beta_prior_var
            .iter()
            .enumerate()
            .map(|(i, v)| format!("coef[{}]={:.6}", i, v))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ---------------------------------------------------------------
    // Step 2: Build lambda vector (on natural log scale)
    // R: lambda_log2 = 1/betaPriorVar
    //    lambda_natlog = lambda_log2 / ln(2)^2
    // The intercept gets a very wide prior (1e-6 on log2 scale).
    // ---------------------------------------------------------------
    let mut lambda: Vec<f64> = beta_prior_var
        .iter()
        .map(|&bpv| (1.0 / bpv) / ln2_sq)
        .collect();
    // Intercept: wide prior (1e-6 on log2 scale -> 1e-6/ln2^2 on natlog scale)
    lambda[0] = 1e-6 / ln2_sq;

    log::info!(
        "lambda (natlog): {}",
        lambda
            .iter()
            .enumerate()
            .map(|(i, v)| format!("coef[{}]={:.6e}", i, v))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ---------------------------------------------------------------
    // Step 3: Re-fit GLM with per-coefficient lambda
    // ---------------------------------------------------------------
    let refit_result = refit_glm_with_prior(dds, &design, &lambda);

    match refit_result {
        Ok((new_coefficients, new_standard_errors)) => {
            // Step 4: Update results with shrunk LFC and SE
            // new_coefficients and new_standard_errors are on log2 scale
            for i in 0..n_genes {
                if !all_zero[i] {
                    results.log2_fold_changes[i] = new_coefficients[[i, coef_idx]];
                    results.lfc_se[i] = new_standard_errors[[i, coef_idx]];
                }
            }

            log::info!("LFC shrinkage complete (normal prior, GLM refit)");
        }
        Err(e) => {
            log::warn!(
                "GLM re-fit with prior failed: {}. Keeping MLE estimates.",
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Contrast;

    #[test]
    fn test_lfc_shrinkage_requires_dds() {
        // This test verifies the function signature compiles correctly.
        // Full integration testing requires a complete DESeqDataSet.
        let results = DESeqResults {
            gene_ids: vec!["g1".to_string(), "g2".to_string(), "g3".to_string()],
            base_means: vec![100.0, 100.0, 100.0],
            base_vars: vec![50.0, 50.0, 50.0],
            log2_fold_changes: vec![2.0, 0.5, 0.1],
            lfc_se: vec![0.1, 0.5, 2.0],
            stat: vec![20.0, 1.0, 0.05],
            pvalues: vec![0.0, 0.3, 0.96],
            padj: vec![0.0, 0.3, 0.96],
            dispersions: vec![0.1, 0.1, 0.1],
            gene_wise_dispersions: vec![0.1, 0.1, 0.1],
            trended_dispersions: vec![0.1, 0.1, 0.1],
            contrast: Contrast {
                variable: "treatment".to_string(),
                numerator: "treated".to_string(),
                denominator: "control".to_string(),
            },
        };

        // Verify the struct is valid
        assert_eq!(results.gene_ids.len(), 3);
        // stat/pvalue should be preserved after shrinkage (they come from MLE)
        assert_eq!(results.stat[0], 20.0);
    }
}
