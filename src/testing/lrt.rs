//! Likelihood Ratio Test for differential expression
//!
//! LRT compares nested models (full vs reduced) to test for differential expression.
//! It's particularly useful for testing multiple coefficients at once, or when
//! testing factors with more than two levels.
//!
//! Algorithm (matches R DESeq2 nbinomLRT):
//! 1. Fit full and reduced NB GLMs for every gene with `fit_single_gene`
//!    (fitNbinomGLMs: IRLS, then the optim() L-BFGS-B fallback where needed)
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
use crate::glm::{fit_single_gene_from, fit_single_gene_weighted, GlmFitParams};
use crate::io::{Contrast, DESeqResults};

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
    // R names the intercept column of a formula's model matrix "Intercept"; here a
    // first column of ones stands for it
    let intercept = design_reduced.ncols() > 0 && design_reduced.column(0).iter().all(|&v| v == 1.0);
    likelihood_ratio_test_reduced(dds, design_full, design_reduced, intercept)
}

/// [`likelihood_ratio_test`] with the reduced model's intercept given: R starts the fit
/// of a reduced model matrix that is not full rank (qr()$rank < ncol) from
/// log(rowMeans(normalized counts)) for the Intercept column and 0 elsewhere, or from
/// 1 for every column if `reduced_has_intercept` is false (fitNbinomGLMs())
/// R equivalent: nbinomLRT(object, full, reduced) with formulas
pub fn likelihood_ratio_test_reduced(
    dds: &DESeqDataSet,
    design_full: &Array2<f64>,
    design_reduced: &Array2<f64>,
    reduced_has_intercept: bool,
) -> Result<DESeqResults> {
    let counts = dds.counts().counts();
    if !dds.has_size_or_norm_factors() {
        return Err(DeseqError::InvalidContrast {
            reason: "Size factors required for LRT".to_string(),
        });
    }
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
    // R: qrx <- qr(modelMatrix); if (qrx$rank == ncol(modelMatrix)) least-squares start
    let reduced_full_rank = crate::glm::rfuns::Qr::new(design_reduced).rank == design_reduced.ncols();

    // Determine if the reduced model is intercept-only (single column of 1s)
    let is_intercept_only = design_reduced.ncols() == 1 && {
        let col = design_reduced.column(0);
        col.iter().all(|&v| (v - 1.0).abs() < 1e-12)
    };

    // Observation weights: normalised for the fits (getAndCheckWeights), raw for
    // baseMean/baseVar (getBaseMeansAndVariances); weightsFail genes are allZero.
    let raw_weights = dds.weights();
    let weights = dds.normalized_weights();
    let all_zero = dds.all_zero();

    // R's nbinomLRT refits both full and reduced models from scratch
    // using fitNbinomGLMs (which includes IRLS + optim fallback).

    // Fit both models for all genes in parallel
    let lrt_results: Vec<LrtGeneResult> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[i, j]]).collect();
            let alpha = dispersions[i];
            // R: getSizeOrNormFactors(objectNZ) row of this gene
            let sf: Vec<f64> = dds.gene_norm_factors(i).unwrap().to_vec();

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

            // Base mean and base var of normalized counts
            // (R getBaseMeansAndVariances: times the raw weights when present)
            let norm_counts: Vec<f64> = gene_counts
                .iter()
                .zip(sf.iter())
                .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
                .enumerate()
                .map(|(j, x)| match raw_weights {
                    Some(raw) => raw[[i, j]] * x,
                    None => x,
                })
                .collect();
            let base_mean: f64 = norm_counts.iter().sum::<f64>() / n_samples as f64;
            let base_var: f64 = norm_counts.iter()
                .map(|&x| (x - base_mean).powi(2))
                .sum::<f64>() / (n_samples as f64 - 1.0);

            // Genes whose weights fail getAndCheckWeights() are allZero in R:
            // baseMean is reported, every test statistic is NA
            if all_zero[i] {
                return LrtGeneResult {
                    base_mean,
                    base_var,
                    lrt_stat: f64::NAN,
                    log2_fold_change: f64::NAN,
                    lfc_se: f64::NAN,
                    betas_full: vec![f64::NAN; df_full],
                    mu_full: vec![f64::NAN; n_samples],
                };
            }

            // R's nbinomLRT refits the full model with fitNbinomGLMs (IRLS from
            // the OLS start, then the optim() fallback of fitNbinomGLMsOptim for
            // the rows that need it), which is what fit_single_gene ports.
            let gene_view = ndarray::ArrayView1::from(&gene_counts[..]);
            let sf_view = ndarray::ArrayView1::from(&sf[..]);
            let gene_w = crate::glm::gene_weights(weights.as_ref(), dds.weights_fail(), i);
            let gene_w = gene_w.as_deref();
            let full_result = fit_single_gene_weighted(gene_view, design_full, sf_view, alpha, &GlmFitParams::default(), gene_w);
            let full_ll = full_result.log_likelihood;

            // Fit reduced model
            // For intercept-only models, R DESeq2 uses a closed-form shortcut
            // that bypasses IRLS entirely (fitNbinomGLMs.R lines 104-137):
            //   beta0 = log2(mean(counts / size_factors))
            //   mu_j  = size_factor_j * mean(counts / size_factors)
            //   logLike = sum(dnbinom(k_j, mu=mu_j, size=1/alpha, log=TRUE))
            // No IRLS, no optim fallback. The intercept-only NB MLE for the
            // log-link is the arithmetic mean of normalized counts.
            let reduced_ll = if let (true, Some(w)) = (is_intercept_only, gene_w) {
                // With weights: betaMatrix <- log2(rowSums(weights * normCounts) / rowSums(weights)),
                // mu <- nf * 2^betaMatrix, logLike <- rowSums(weights * dnbinom(k, mu, 1/alpha))
                let wsum_norm: f64 = gene_counts
                    .iter()
                    .zip(sf.iter())
                    .zip(w)
                    .map(|((&c, &s), &wj)| wj * if s > 0.0 { c / s } else { 0.0 })
                    .sum();
                let mean_norm = wsum_norm / w.iter().sum::<f64>();
                let beta = mean_norm.log2();
                let mut ll = 0.0;
                for j in 0..n_samples {
                    let mu_j = sf[j] * 2f64.powf(beta);
                    ll += w[j] * nb_log_likelihood(gene_counts[j], mu_j, alpha);
                }
                ll
            } else if is_intercept_only {
                // mean_norm = mean(k / sf) -- arithmetic mean of normalized counts
                let mean_norm: f64 = gene_counts
                    .iter()
                    .zip(sf.iter())
                    .map(|(&c, &s)| if s > 0.0 { c / s } else { 0.0 })
                    .sum::<f64>()
                    / n_samples as f64;

                // mu_j = sf_j * mean_norm (UNCLAMPED, matching R); logLike from it
                let mut ll = 0.0;
                for j in 0..n_samples {
                    let mu_j = sf[j] * mean_norm; // unclamped
                    ll += nb_log_likelihood(gene_counts[j], mu_j, alpha);
                }
                ll
            } else {
                let gene_view = ndarray::ArrayView1::from(&gene_counts[..]);
                let sf_view = ndarray::ArrayView1::from(&sf[..]);
                // R (fitNbinomGLMs, model matrix not full rank): beta_mat starts at 0 with
                // log(rowMeans(counts(object, normalized = TRUE))) for the Intercept,
                // or at 1 without an intercept
                let start: Option<Vec<f64>> = (!reduced_full_rank).then(|| {
                    if reduced_has_intercept {
                        let norm: Vec<f64> = gene_counts.iter().zip(&sf).map(|(&c, &s)| c / s).collect();
                        let mut b = vec![0.0; design_reduced.ncols()];
                        b[0] = (crate::optim::nmath::r_sum(norm.iter().copied()) / n_samples as f64).ln();
                        b
                    } else {
                        vec![1.0; design_reduced.ncols()]
                    }
                });
                fit_single_gene_from(gene_view, design_reduced, sf_view, alpha, &GlmFitParams::default(), gene_w, start.as_deref())
                    .log_likelihood
            };

            // LRT statistic
            let lrt_stat = 2.0 * (full_ll - reduced_ll);
            // Note: R does not clamp LRT stat to >= 0. Negative values shouldn't
            // happen theoretically but can arise from numerical issues. We clamp
            // to 0 for safety; this is a minor deviation from R behavior.
            let lrt_stat = lrt_stat.max(0.0);

            // Log2 fold change: use full model's last non-intercept coefficient
            // Convert from natural log to log2 scale: beta_natlog * log2(e)
            let log2_e = std::f64::consts::LOG2_E;
            let full_betas = &full_result.coefficients;
            let log2_fold_change = if full_betas.len() > 1 {
                // Use the last non-intercept coefficient (the main effect)
                full_betas[full_betas.len() - 1] * log2_e
            } else {
                0.0
            };

            // SE from the full model fit (R: betaSE of fitNbinomGLMs)
            let se_vec = &full_result.standard_errors;
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
                betas_full: full_result.coefficients,
                mu_full: full_result.mu,
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
