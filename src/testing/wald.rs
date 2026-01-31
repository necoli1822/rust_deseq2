//! Wald test for differential expression

use ndarray::Axis;

use super::fdr::benjamini_hochberg;
use super::pvalue::calculate_pvalue;
use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};
use crate::glm::{get_contrast_index, DesignInfo};
use crate::io::{Contrast, DESeqResults};

/// Perform Wald test for a specific contrast
/// R equivalent: nbinomWaldTest()
///
/// When `lfc_threshold > 0`, performs a test of H0: |LFC| <= threshold
/// using the "greaterAbs" alternative hypothesis (R's default when lfcThreshold > 0).
/// R equivalent: results(dds, lfcThreshold=..., altHypothesis="greaterAbs")
///   stat = sign(log2FC) * max(0, (|log2FC| - threshold) / lfcSE)
///   pvalue = min(1, 2 * pnorm(-|stat|))
pub fn wald_test(
    dds: &DESeqDataSet,
    design_info: &DesignInfo,
    contrast: Contrast,
    _alpha: f64,
    use_t: bool,
    lfc_threshold: f64,
) -> Result<DESeqResults> {
    // Check if GLM has been fitted
    let coefficients = dds.coefficients().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "GLM must be fitted before testing".to_string(),
    })?;

    let standard_errors = dds.standard_errors().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Standard errors not available".to_string(),
    })?;

    // Find the coefficient index and sign for the contrast
    // sign is -1.0 when the user requested the reverse direction (numerator=reference)
    let (coef_idx, contrast_sign) = get_contrast_index(design_info, &contrast.numerator, &contrast.denominator)?;

    let n_genes = dds.n_genes();
    let gene_ids = dds.counts().gene_ids().to_vec();

    // Calculate base means and base vars (mean and variance of normalized counts)
    let (base_means, base_vars): (Vec<f64>, Vec<f64>) = if let Some(norm_counts) = dds.normalized_counts() {
        let n_samples_f = dds.n_samples() as f64;
        let means: Vec<f64> = norm_counts
            .axis_iter(Axis(0))
            .map(|row| row.sum() / n_samples_f)
            .collect();
        let vars: Vec<f64> = norm_counts
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, row)| {
                let mean = means[i];
                row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_samples_f - 1.0)
            })
            .collect();
        (means, vars)
    } else {
        (vec![f64::NAN; n_genes], vec![f64::NAN; n_genes])
    };

    // Extract log fold changes and standard errors
    // Apply contrast_sign to handle reverse contrasts (numerator=reference -> negate)
    // R sets LFC=0, SE=0 for allZero genes (baseMean == 0)
    let log2_fold_changes: Vec<f64> = (0..n_genes)
        .map(|i| {
            if base_means[i] == 0.0 {
                return 0.0;
            }
            let lfc = coefficients[[i, coef_idx]] * contrast_sign;
            // Convert from natural log to log2
            lfc / 2.0_f64.ln()
        })
        .collect();

    let lfc_se: Vec<f64> = (0..n_genes)
        .map(|i| {
            if base_means[i] == 0.0 {
                return 0.0;
            }
            let se = standard_errors[[i, coef_idx]];
            // Convert from natural log to log2
            se / 2.0_f64.ln()
        })
        .collect();

    // Calculate Wald statistics
    // When lfc_threshold > 0 (greaterAbs alternative hypothesis):
    //   R: stat = sign(log2FC) * max(0, (|log2FC| - threshold) / lfcSE)
    //   pvalue = min(1, 2 * pnorm(-|stat|))
    // When lfc_threshold == 0 (standard Wald test):
    //   stat = beta / se (in natural log scale, same ratio as log2 scale)
    let stat: Vec<f64> = (0..n_genes)
        .map(|i| {
            // R sets pvalue/padj to NA for allZero genes (baseMean == 0)
            if base_means[i] == 0.0 {
                return f64::NAN;
            }
            let beta = coefficients[[i, coef_idx]] * contrast_sign;
            let se = standard_errors[[i, coef_idx]];
            if se > 0.0 && se.is_finite() {
                if lfc_threshold > 0.0 {
                    // greaterAbs: test H0: |log2FC| <= threshold
                    let lfc_log2 = beta / 2.0_f64.ln();
                    let se_log2 = se / 2.0_f64.ln();
                    let abs_lfc = lfc_log2.abs();
                    if abs_lfc < lfc_threshold {
                        0.0
                    } else {
                        lfc_log2.signum() * (abs_lfc - lfc_threshold) / se_log2
                    }
                } else {
                    beta / se
                }
            } else {
                f64::NAN
            }
        })
        .collect();

    // Calculate p-values
    let pvalues: Vec<f64> = if use_t {
        let df = dds.n_samples() as f64 - design_info.coef_names.len() as f64;
        stat.iter()
            .map(|&z| super::pvalue::calculate_pvalue_t(z, df))
            .collect()
    } else {
        stat.iter().map(|&z| calculate_pvalue(z)).collect()
    };

    // Apply BH correction
    let padj = benjamini_hochberg(&pvalues);

    // Get dispersions (MAP)
    let dispersions: Vec<f64> = if let Some(disp) = dds.dispersions() {
        disp.to_vec()
    } else {
        vec![f64::NAN; n_genes]
    };

    // Get gene-wise dispersions
    let gene_wise_dispersions: Vec<f64> = if let Some(disp) = dds.gene_dispersions() {
        disp.to_vec()
    } else {
        vec![f64::NAN; n_genes]
    };

    // Get trended dispersions
    let trended_dispersions: Vec<f64> = if let Some(disp) = dds.trended_dispersions() {
        disp.to_vec()
    } else {
        vec![f64::NAN; n_genes]
    };

    Ok(DESeqResults {
        gene_ids,
        base_means,
        base_vars,
        log2_fold_changes,
        lfc_se,
        stat,
        pvalues,
        padj,
        dispersions,
        gene_wise_dispersions,
        trended_dispersions,
        contrast,
    })
}

/// Perform Wald test with extended contrast specification
/// R equivalent: nbinomWaldTest() with numeric/list contrast in core.R
/// Supports DESeq2-style contrast types
///
/// When `lfc_threshold > 0`, performs greaterAbs test (see `wald_test` docs).
pub fn wald_test_extended(
    dds: &DESeqDataSet,
    design_info: &DesignInfo,
    contrast: crate::io::ContrastSpec,
    _alpha: f64,
    use_t: bool,
    lfc_threshold: f64,
) -> Result<DESeqResults> {
    use crate::io::ContrastSpec;

    // Check if GLM has been fitted
    let coefficients = dds.coefficients().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "GLM must be fitted before testing".to_string(),
    })?;

    let standard_errors = dds.standard_errors().ok_or_else(|| DeseqError::InvalidContrast {
        reason: "Standard errors not available".to_string(),
    })?;

    // Try to get covariance matrices (newly added), fall back to diagonal approximation if missing
    let covariances = dds.covariances();

    let n_genes = dds.n_genes();
    let n_coefs = coefficients.ncols();
    let gene_ids = dds.counts().gene_ids().to_vec();

    // Convert contrast spec to contrast vector
    let (contrast_vec, contrast_description) = match &contrast {
        ContrastSpec::Simple {
            variable,
            numerator,
            denominator,
        } => {
            // Find coefficient index and sign for simple contrast
            let (coef_idx, sign) =
                get_contrast_index(design_info, numerator, denominator)?;
            let mut vec = vec![0.0; n_coefs];
            vec[coef_idx] = sign;
            let desc = format!("{} {} vs {}", variable, numerator, denominator);
            (vec, desc)
        }

        ContrastSpec::Name(coef_name) => {
            // Find coefficient by name
            let coef_idx = design_info
                .coef_names
                .iter()
                .position(|n| n == coef_name)
                .ok_or_else(|| DeseqError::InvalidContrast {
                    reason: format!(
                        "Coefficient '{}' not found. Available: {:?}",
                        coef_name, design_info.coef_names
                    ),
                })?;
            let mut vec = vec![0.0; n_coefs];
            vec[coef_idx] = 1.0;
            (vec, coef_name.clone())
        }

        ContrastSpec::Numeric(weights) => {
            if weights.len() != n_coefs {
                return Err(DeseqError::InvalidContrast {
                    reason: format!(
                        "Contrast vector length ({}) doesn't match number of coefficients ({})",
                        weights.len(),
                        n_coefs
                    ),
                });
            }
            let desc = format!("numeric contrast: {:?}", weights);
            (weights.clone(), desc)
        }

        ContrastSpec::List {
            numerator_coefs,
            denominator_coefs,
            list_values,
        } => {
            let mut vec = vec![0.0; n_coefs];
            let (num_weight, denom_weight) = list_values;

            // Add numerator weight for numerator coefficients
            for coef_name in numerator_coefs {
                let idx = design_info
                    .coef_names
                    .iter()
                    .position(|n| n == coef_name)
                    .ok_or_else(|| DeseqError::InvalidContrast {
                        reason: format!("Coefficient '{}' not found", coef_name),
                    })?;
                vec[idx] += num_weight;
            }

            // Add denominator weight for denominator coefficients
            for coef_name in denominator_coefs {
                let idx = design_info
                    .coef_names
                    .iter()
                    .position(|n| n == coef_name)
                    .ok_or_else(|| DeseqError::InvalidContrast {
                        reason: format!("Coefficient '{}' not found", coef_name),
                    })?;
                vec[idx] += denom_weight;
            }

            let desc = format!(
                "({}) - ({})",
                numerator_coefs.join(" + "),
                denominator_coefs.join(" + ")
            );
            (vec, desc)
        }
    };

    // Calculate base means and base vars
    let (base_means, base_vars): (Vec<f64>, Vec<f64>) = if let Some(norm_counts) = dds.normalized_counts() {
        let n_samples_f = dds.n_samples() as f64;
        let means: Vec<f64> = norm_counts
            .axis_iter(ndarray::Axis(0))
            .map(|row| row.sum() / n_samples_f)
            .collect();
        let vars: Vec<f64> = norm_counts
            .axis_iter(ndarray::Axis(0))
            .enumerate()
            .map(|(i, row)| {
                let mean = means[i];
                row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_samples_f - 1.0)
            })
            .collect();
        (means, vars)
    } else {
        (vec![f64::NAN; n_genes], vec![f64::NAN; n_genes])
    };

    // Calculate log2 fold changes using contrast vector
    // LFC = sum(contrast_vec[j] * beta[j]) for j in coefficients
    // R sets LFC=0 for allZero genes (baseMean == 0)
    let log2_fold_changes: Vec<f64> = (0..n_genes)
        .map(|i| {
            if base_means[i] == 0.0 {
                return 0.0;
            }
            let lfc: f64 = (0..n_coefs)
                .map(|j| contrast_vec[j] * coefficients[[i, j]])
                .sum();
            // Convert from natural log to log2
            lfc / 2.0_f64.ln()
        })
        .collect();

    // Calculate standard errors using contrast vector
    // If covariance matrix is available: SE = sqrt(contrast' * Cov * contrast)
    // Otherwise, fallback to diagonal approximation: SE = sqrt(sum(contrast^2 * se^2))
    // R sets SE=0 for allZero genes (baseMean == 0)
    let lfc_se: Vec<f64> = (0..n_genes)
        .map(|i| {
            if base_means[i] == 0.0 {
                return 0.0;
            }
            let var = if let Some(covs) = covariances {
                // covs is Array3 (n_genes x n_coefs x n_coefs)
                // c' * Cov * c = sum_j sum_k c_j * Cov_jk * c_k
                let mut sum = 0.0;
                for j in 0..n_coefs {
                    for k in 0..n_coefs {
                        sum += contrast_vec[j] * covs[[i, j, k]] * contrast_vec[k];
                    }
                }
                sum
            } else {
                // Fallback: diagonal approximation
                (0..n_coefs)
                    .map(|j| {
                        let se = standard_errors[[i, j]];
                        contrast_vec[j] * contrast_vec[j] * se * se
                    })
                    .sum()
            };
            
            // Convert from natural log to log2
            if var > 0.0 {
                var.sqrt() / 2.0_f64.ln()
            } else {
                f64::NAN
            }
        })
        .collect();

    // Calculate Wald statistics
    // R computes stat/pvalue for non-converged genes too (just flags betaConv=FALSE).
    // Only allZero genes (baseMean == 0) get NA.
    let stat: Vec<f64> = (0..n_genes)
        .map(|i| {
            // R sets pvalue/padj to NA for allZero genes (baseMean == 0)
            if base_means[i] == 0.0 {
                return f64::NAN;
            }

            let lfc_nat: f64 = (0..n_coefs)
                .map(|j| contrast_vec[j] * coefficients[[i, j]])
                .sum();

            let var_nat = if let Some(covs) = covariances {
                let mut sum = 0.0;
                for j in 0..n_coefs {
                    for k in 0..n_coefs {
                        sum += contrast_vec[j] * covs[[i, j, k]] * contrast_vec[k];
                    }
                }
                sum
            } else {
                (0..n_coefs)
                    .map(|j| {
                        let se = standard_errors[[i, j]];
                        contrast_vec[j] * contrast_vec[j] * se * se
                    })
                    .sum()
            };

            let se_nat = if var_nat > 0.0 { var_nat.sqrt() } else { 0.0 };

            if se_nat > 0.0 && se_nat.is_finite() {
                if lfc_threshold > 0.0 {
                    // greaterAbs: test H0: |log2FC| <= threshold
                    let lfc_log2 = lfc_nat / 2.0_f64.ln();
                    let se_log2 = se_nat / 2.0_f64.ln();
                    let abs_lfc = lfc_log2.abs();
                    if abs_lfc < lfc_threshold {
                        0.0
                    } else {
                        lfc_log2.signum() * (abs_lfc - lfc_threshold) / se_log2
                    }
                } else {
                    lfc_nat / se_nat
                }
            } else {
                f64::NAN
            }
        })
        .collect();

    // Calculate p-values
    let pvalues: Vec<f64> = if use_t {
        let df = dds.n_samples() as f64 - design_info.coef_names.len() as f64;
        stat.iter()
            .map(|&z| super::pvalue::calculate_pvalue_t(z, df))
            .collect()
    } else {
        stat.iter().map(|&z| calculate_pvalue(z)).collect()
    };

    // Apply BH correction
    let padj = benjamini_hochberg(&pvalues);

    // Get dispersions
    let dispersions: Vec<f64> = if let Some(disp) = dds.dispersions() {
        disp.to_vec()
    } else {
        vec![f64::NAN; n_genes]
    };

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

    // Create legacy contrast for results struct
    let legacy_contrast = contrast.to_legacy().unwrap_or_else(|| Contrast {
        variable: "contrast".to_string(),
        numerator: contrast_description.clone(),
        denominator: "0".to_string(),
    });

    Ok(DESeqResults {
        gene_ids,
        base_means,
        base_vars,
        log2_fold_changes,
        lfc_se,
        stat,
        pvalues,
        padj,
        dispersions,
        gene_wise_dispersions,
        trended_dispersions,
        contrast: legacy_contrast,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, SampleMetadata};
    use crate::dispersion::{estimate_dispersions, DispersionParams, TrendFitMethod};
    use crate::glm::{fit_glm, GlmFitParams};
    use crate::normalization::{estimate_size_factors, SizeFactorMethod};
    use ndarray::{array, Array1};

    #[test]
    fn test_wald_test() {
        // Create dataset with clear differential expression
        let counts = CountMatrix::new(
            array![
                [100.0, 110.0, 90.0, 400.0, 420.0, 380.0],  // ~4x up
                [500.0, 520.0, 480.0, 500.0, 510.0, 490.0], // no change
                [200.0, 210.0, 190.0, 50.0, 55.0, 45.0]     // ~4x down
            ],
            vec!["gene_up".to_string(), "gene_nc".to_string(), "gene_down".to_string()],
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
        let info = fit_glm(&mut dds, &GlmFitParams::default()).unwrap();

        let contrast = Contrast {
            variable: "treatment".to_string(),
            numerator: "treated".to_string(),
            denominator: "control".to_string(),
        };

        let results = wald_test(&dds, &info, contrast, 0.05, false, 0.0).unwrap();

        // Check log2 fold changes
        // gene_up: ~2 (4x increase)
        assert!(results.log2_fold_changes[0] > 1.5);

        // gene_nc: ~0
        assert!(results.log2_fold_changes[1].abs() < 0.5);

        // gene_down: ~-2 (4x decrease)
        assert!(results.log2_fold_changes[2] < -1.5);

        // All p-values should be valid
        for p in &results.pvalues {
            if p.is_finite() {
                assert!(*p >= 0.0 && *p <= 1.0);
            }
        }
    }

    #[test]
    fn test_wald_numeric_contrast() {
        use crate::io::ContrastSpec;
        
        // Setup simple 2-group dataset
        let counts = CountMatrix::new(
            array![
                [100.0, 100.0, 400.0, 400.0], // Up
                [500.0, 500.0, 500.0, 500.0], // No change
            ],
            vec!["gene_up".to_string(), "gene_nc".to_string()],
            vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()],
        ).unwrap();

        let mut metadata = SampleMetadata::new(vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()]);
        metadata.add_condition("condition", vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()]).unwrap();

        let mut dds = DESeqDataSet::new(counts, metadata, "condition").unwrap();
        
        // Force size factors to 1.0 to avoid normalization artifacts in this test
        let n_samples = dds.n_samples();
        dds.set_size_factors(Array1::from_elem(n_samples, 1.0)).unwrap();
        
        estimate_dispersions(&mut dds, TrendFitMethod::Mean, &DispersionParams::default()).unwrap();
        let info = fit_glm(&mut dds, &GlmFitParams::default()).unwrap();

        // Design matrix has Intercept and condition_B_vs_A
        // Coefs: [Intercept, beta_B]
        // Standard contrast "B vs A" corresponds to numeric vector [0, 1]
        let contrast_vec = vec![0.0, 1.0];
        let spec = ContrastSpec::Numeric(contrast_vec.clone());

        let results = wald_test_extended(&dds, &info, spec, 0.05, false, 0.0).unwrap();

        // Check gene_up (should be ~2 log2FC)
        assert!(results.log2_fold_changes[0] > 1.5);
        assert!(results.padj[0] < 0.05);

        // Check gene_nc (should be ~0 log2FC)
        assert!(results.log2_fold_changes[1].abs() < 0.5);

        // Verify that SE is calculated (not NaN)
        assert!(results.lfc_se[0].is_finite());
        assert!(results.lfc_se[0] > 0.0);
        
        // Test with a linear combination: 0.5 * Intercept + 0.5 * beta_B
        // This corresponds to the average expression of group B (on log scale)
        let contrast_vec_avg = vec![0.5, 0.5];
        let spec_avg = ContrastSpec::Numeric(contrast_vec_avg);
        let results_avg = wald_test_extended(&dds, &info, spec_avg, 0.05, false, 0.0).unwrap();
        
        assert!(results_avg.log2_fold_changes[0].is_finite());
        assert!(results_avg.lfc_se[0].is_finite());
    }

    #[test]
    fn test_wald_use_t_parameter() {
        // Create small dataset to test t-distribution (more conservative with small n)
        let counts = CountMatrix::new(
            array![
                [100.0, 110.0, 90.0, 400.0, 420.0, 380.0],  // Up-regulated
            ],
            vec!["gene_up".to_string()],
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
        let info = fit_glm(&mut dds, &GlmFitParams::default()).unwrap();

        let contrast = Contrast {
            variable: "treatment".to_string(),
            numerator: "treated".to_string(),
            denominator: "control".to_string(),
        };

        // Test with normal distribution (use_t=false)
        let results_normal = wald_test(&dds, &info, contrast.clone(), 0.05, false, 0.0).unwrap();

        // Test with t-distribution (use_t=true)
        let results_t = wald_test(&dds, &info, contrast, 0.05, true, 0.0).unwrap();

        // Log2FC and stat should be identical
        assert_eq!(results_normal.log2_fold_changes[0], results_t.log2_fold_changes[0]);
        assert_eq!(results_normal.stat[0], results_t.stat[0]);

        // P-value should be larger (more conservative) with t-distribution
        // df = n_samples - n_coefs = 6 - 2 = 4 (small df, so noticeable difference)
        assert!(results_t.pvalues[0] > results_normal.pvalues[0]);
    }
}
