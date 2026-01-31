//! Cook's distance outlier filtering
//!
//! Cook's distance measures the influence of each observation on the fitted model.
//! Observations with high Cook's distance may be outliers that unduly influence results.
//!
//! This implementation exactly matches DESeq2's approach:
//! - Uses hat diagonals from GLM fitting
//! - Uses robust method of moments dispersion (trimmed variance)
//! - Formula: cooks = PearsonResSq/p * H/(1 - H)^2

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::data::DESeqDataSet;
use crate::error::Result;
use crate::io::DESeqResults;

/// Calculate Cook's distance for all genes and samples
/// R equivalent: calculateCooksDistance() in core.R
/// Matches DESeq2's calculateCooksDistance function exactly
///
/// Returns a matrix of Cook's distances (genes x samples)
pub fn calculate_cooks_distance(dds: &DESeqDataSet) -> Result<Array2<f64>> {
    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();

    // Get mu (fitted values) from the dataset
    let mu = match dds.mu() {
        Some(m) => m,
        None => return Ok(Array2::zeros((n_genes, n_samples))),
    };

    // Get hat diagonals from GLM fitting
    let hat_diagonals = match dds.hat_diagonals() {
        Some(h) => h,
        None => return Ok(Array2::zeros((n_genes, n_samples))),
    };

    // Get design matrix for number of parameters
    let design = match dds.design_matrix() {
        Some(d) => d,
        None => return Ok(Array2::zeros((n_genes, n_samples))),
    };
    let p = design.ncols();

    // Calculate robust method of moments dispersion (DESeq2's robustMethodOfMomentsDisp)
    let dispersions = robust_method_of_moments_disp(dds);

    // Get normalized counts for Pearson residual calculation
    let counts = dds.counts().counts();

    // Calculate Cook's distance for each gene in parallel
    // DESeq2 formula: cooks = PearsonResSq/p * H/(1 - H)^2
    let cooks_flat: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .flat_map(|i| {
            let alpha = dispersions[i];
            (0..n_samples)
                .map(|j| {
                    let y = counts[[i, j]];
                    let mu_ij = mu[[i, j]];
                    let h = hat_diagonals[[i, j]];

                    // Variance for negative binomial: V = mu + alpha * mu^2
                    let v = mu_ij + alpha * mu_ij * mu_ij;

                    // Pearson residual squared: (y - mu)^2 / V
                    let pearson_resid_sq = if v > 0.0 {
                        (y - mu_ij).powi(2) / v
                    } else {
                        0.0
                    };

                    // Cook's distance: PearsonResSq/p * H/(1 - H)^2
                    if h.is_finite() && h < 1.0 && pearson_resid_sq.is_finite() {
                        pearson_resid_sq / p as f64 * h / (1.0 - h).powi(2)
                    } else {
                        f64::NAN
                    }
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    let mut cooks = Array2::zeros((n_genes, n_samples));
    for i in 0..n_genes {
        for j in 0..n_samples {
            cooks[[i, j]] = cooks_flat[i * n_samples + j];
        }
    }

    Ok(cooks)
}

/// Calculate robust method of moments dispersion
/// R equivalent: robustMethodOfMomentsDisp() in core.R
/// Matches DESeq2's robustMethodOfMomentsDisp function
///
/// Uses trimmed cell variance within conditions, then calculates
/// dispersion as: alpha = (variance - mean) / mean^2
///
/// IMPORTANT: For multi-factor designs, cells are determined by hashing
/// the model matrix rows, not just the design variable. This ensures
/// correct cell assignment for designs like ~ solvent + antibiotic.
pub fn robust_method_of_moments_disp(dds: &DESeqDataSet) -> Vec<f64> {
    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();
    let min_disp = 0.04;

    // Get normalized counts
    let norm_counts = match dds.normalized_counts() {
        Some(nc) => nc.clone(),
        None => {
            // Fall back to raw counts / size factors
            let counts = dds.counts().counts();
            let sf = dds.size_factors().map(|s| s.to_vec()).unwrap_or_else(|| vec![1.0; n_samples]);
            let mut nc = counts.to_owned();
            for i in 0..n_genes {
                for j in 0..n_samples {
                    if sf[j] > 0.0 {
                        nc[[i, j]] /= sf[j];
                    }
                }
            }
            nc
        }
    };

    // DESeq2 uses model matrix rows to determine cells:
    // cells <- apply(modelMatrix,1,paste0,collapse="")
    // This ensures correct cell assignment for multi-factor designs
    let cells: Vec<String> = if let Some(design_matrix) = dds.design_matrix() {
        // Hash each row of the model matrix to create cell identifiers
        (0..n_samples)
            .map(|j| {
                let row: Vec<String> = (0..design_matrix.ncols())
                    .map(|k| format!("{:.6}", design_matrix[[j, k]]))
                    .collect();
                row.join("_")
            })
            .collect()
    } else {
        // Fall back to design variable if no model matrix available
        let design_var = dds.design_variable();
        dds.sample_metadata()
            .condition(design_var)
            .cloned()
            .unwrap_or_else(|| vec!["unknown".to_string(); n_samples])
    };

    // Group samples by cell
    let mut cell_samples: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, cell) in cells.iter().enumerate() {
        cell_samples.entry(cell.clone()).or_default().push(i);
    }

    // Check if we have 3 or more samples in any cell (DESeq2: nOrMoreInCell)
    let has_three_or_more = cell_samples.values().any(|v| v.len() >= 3);

    // Calculate variance for each gene
    let variances: Vec<f64> = if has_three_or_more {
        // Use trimmed cell variance within cells
        trimmed_cell_variance(&norm_counts, &cells, &cell_samples)
    } else {
        // Use trimmed variance across all samples
        trimmed_variance(&norm_counts)
    };

    // Calculate row means
    let means: Vec<f64> = (0..n_genes)
        .map(|i| {
            let sum: f64 = (0..n_samples).map(|j| norm_counts[[i, j]]).sum();
            sum / n_samples as f64
        })
        .collect();

    // Calculate dispersion: alpha = (v - m) / m^2
    (0..n_genes)
        .map(|i| {
            let v = variances[i];
            let m = means[i];
            if m > 0.0 {
                let alpha = (v - m) / (m * m);
                alpha.max(min_disp)
            } else {
                min_disp
            }
        })
        .collect()
}

/// Calculate trimmed cell variance within each condition
/// Matches DESeq2's trimmedCellVariance function
fn trimmed_cell_variance(
    norm_counts: &Array2<f64>,
    conditions: &[String],
    condition_samples: &std::collections::HashMap<String, Vec<usize>>,
) -> Vec<f64> {
    let n_genes = norm_counts.nrows();

    // Get unique conditions in order
    let mut unique_conditions: Vec<String> = Vec::new();
    for cond in conditions {
        if !unique_conditions.contains(cond) {
            unique_conditions.push(cond.clone());
        }
    }

    // Filter to conditions with >= 3 samples
    let valid_conditions: Vec<&String> = unique_conditions
        .iter()
        .filter(|c| condition_samples.get(*c).map(|v| v.len()).unwrap_or(0) >= 3)
        .collect();

    if valid_conditions.is_empty() {
        return trimmed_variance(norm_counts);
    }

    // Calculate trimmed cell variance for each gene
    (0..n_genes)
        .map(|gene_idx| {
            let mut max_var = 0.0_f64;

            for cond in &valid_conditions {
                let sample_indices = condition_samples.get(*cond).unwrap();
                let n = sample_indices.len();

                // Determine trim ratio based on sample size
                // DESeq2: trimratio = c(1/3, 1/4, 1/8) for n in (0,3.5], (3.5,23.5], (23.5,Inf)
                let (trim_ratio, scale_c) = if n <= 3 {
                    (1.0 / 3.0, 2.04)
                } else if n <= 23 {
                    (1.0 / 4.0, 1.86)
                } else {
                    (1.0 / 8.0, 1.51)
                };

                // Get values for this condition
                let mut values: Vec<f64> = sample_indices
                    .iter()
                    .map(|&j| norm_counts[[gene_idx, j]])
                    .collect();

                // Calculate trimmed mean
                let cell_mean = trimmed_mean(&mut values, trim_ratio);

                // Calculate squared errors
                let sq_errors: Vec<f64> = sample_indices
                    .iter()
                    .map(|&j| (norm_counts[[gene_idx, j]] - cell_mean).powi(2))
                    .collect();

                // Calculate trimmed mean of squared errors
                let mut sq_errors_copy = sq_errors.clone();
                let trimmed_mse = trimmed_mean(&mut sq_errors_copy, trim_ratio);

                // Scale to get variance estimate
                let var_est = scale_c * trimmed_mse;

                if var_est > max_var {
                    max_var = var_est;
                }
            }

            max_var
        })
        .collect()
}

/// Calculate trimmed variance across all samples
/// Matches DESeq2's trimmedVariance function
fn trimmed_variance(norm_counts: &Array2<f64>) -> Vec<f64> {
    let n_genes = norm_counts.nrows();
    let n_samples = norm_counts.ncols();
    let trim_ratio = 1.0 / 8.0;
    let scale_c = 1.51;

    (0..n_genes)
        .map(|gene_idx| {
            let values: Vec<f64> = (0..n_samples)
                .map(|j| norm_counts[[gene_idx, j]])
                .collect();

            // Calculate trimmed mean
            let rm = trimmed_mean(&mut values.clone(), trim_ratio);

            // Calculate squared errors
            let sq_errors: Vec<f64> = values.iter().map(|&v| (v - rm).powi(2)).collect();

            // Calculate trimmed mean of squared errors
            let mut sq_errors_copy = sq_errors.clone();
            let trimmed_mse = trimmed_mean(&mut sq_errors_copy, trim_ratio);

            scale_c * trimmed_mse
        })
        .collect()
}

/// Calculate trimmed mean with given trim ratio
fn trimmed_mean(values: &mut Vec<f64>, trim_ratio: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Number of values to trim from each end
    let k = (n as f64 * trim_ratio).floor() as usize;

    if k * 2 >= n {
        // If trimming would remove all values, return simple mean
        return values.iter().sum::<f64>() / n as f64;
    }

    let trimmed: &[f64] = &values[k..n - k];
    if trimmed.is_empty() {
        values.iter().sum::<f64>() / n as f64
    } else {
        trimmed.iter().sum::<f64>() / trimmed.len() as f64
    }
}

/// Get the default Cook's distance cutoff
/// R equivalent: qf(0.99, p, m-p) in DESeq() pipeline
///
/// Uses the F distribution: F(p, n-p) at the 0.99 quantile
pub fn default_cooks_cutoff(n_samples: usize, n_coefs: usize) -> f64 {
    let df1 = n_coefs as f64;
    let df2 = (n_samples - n_coefs) as f64;

    if df2 <= 0.0 {
        return f64::INFINITY;
    }

    // Use F distribution quantile at 0.99
    match FisherSnedecor::new(df1, df2) {
        Ok(f_dist) => f_dist.inverse_cdf(0.99),
        Err(_) => 4.0 / n_samples as f64, // Fallback: 4/n rule of thumb
    }
}

/// Filter results by Cook's distance
/// R equivalent: recordCooksCutoff + filtering logic in DESeq() pipeline
///
/// Sets p-values to NA for genes where any sample exceeds the Cook's distance cutoff.
/// Matches DESeq2's behavior including the "dontFilter" heuristic for two-level designs.
pub fn filter_by_cooks(results: &mut DESeqResults, cooks: &Array2<f64>, cutoff: f64) {
    filter_by_cooks_with_counts(results, cooks, cutoff, None, false)
}

/// Filter results by Cook's distance with optional DESeq2 dontFilter heuristic
///
/// For two-level factor designs, DESeq2 has a special case:
/// If the outlier sample has a low count and 3+ other samples have higher counts,
/// the gene is NOT filtered (the outlier is on the "low side" of the distribution).
///
/// R equivalent: recordCooksCutoff + dontFilter heuristic in results.R
///
/// Parameters:
/// - results: The results to filter (pvalues will be set to NA)
/// - cooks: Cook's distance matrix (genes x samples)
/// - cutoff: Cook's distance cutoff
/// - counts: Optional raw count matrix for dontFilter heuristic
/// - is_two_level_design: If true and counts provided, apply dontFilter heuristic
pub fn filter_by_cooks_with_counts(
    results: &mut DESeqResults,
    cooks: &Array2<f64>,
    cutoff: f64,
    counts: Option<&Array2<f64>>,
    is_two_level_design: bool,
) {
    let n_genes = results.gene_ids.len();
    let n_samples = cooks.ncols();

    for i in 0..n_genes {
        // Find if there's an outlier and which sample it is
        let mut max_cooks = 0.0_f64;
        let mut max_cooks_sample = 0;

        for j in 0..n_samples {
            let c = cooks[[i, j]];
            if c.is_finite() && c > max_cooks {
                max_cooks = c;
                max_cooks_sample = j;
            }
        }

        // Check if this gene has a Cook's outlier
        if max_cooks > cutoff {
            // DESeq2's dontFilter heuristic (results.R lines 572-593):
            // This heuristic ONLY applies when:
            // 1. Design has exactly ONE variable
            // 2. That variable is a factor with exactly 2 levels (nlevels(var) == 2)
            //
            // For two-level designs: don't filter if 3+ samples have counts > outlier sample's count
            // For multi-level designs: always filter (set to NA) when max_cooks > cutoff
            let should_filter = if is_two_level_design {
                // Apply dontFilter heuristic only for two-level designs
                if let Some(count_matrix) = counts {
                    let outlier_count = count_matrix[[i, max_cooks_sample]];

                    // Count samples with counts > outlier_count
                    let n_greater = (0..n_samples)
                        .filter(|&j| count_matrix[[i, j]] > outlier_count)
                        .count();

                    // DESeq2: if n_greater >= 3, don't filter
                    n_greater < 3
                } else {
                    true // No counts provided, filter normally
                }
            } else {
                // For multi-level designs, always filter when max_cooks > cutoff
                true
            };

            if should_filter {
                results.pvalues[i] = f64::NAN;
                results.padj[i] = f64::NAN;
            }
        }
    }
}

/// Get maximum Cook's distance for each gene
/// R equivalent: apply(assays(dds)[["cooks"]], 1, max) in DESeq() pipeline
pub fn max_cooks_per_gene(cooks: &Array2<f64>) -> Array1<f64> {
    let n_genes = cooks.nrows();
    let n_samples = cooks.ncols();

    let mut max_cooks = Array1::zeros(n_genes);
    for i in 0..n_genes {
        let mut max_val = 0.0_f64;
        for j in 0..n_samples {
            let c = cooks[[i, j]];
            if c.is_finite() && c > max_val {
                max_val = c;
            }
        }
        max_cooks[i] = max_val;
    }

    max_cooks
}

/// Replace outlier counts using DESeq2's exact method
/// R equivalent: replaceOutliers() in core.R
/// Returns (modified_genes, flagged_genes) where:
/// - modified_genes: genes where counts were actually replaced
/// - flagged_genes: ALL genes where ANY sample has Cook's > cutoff (for refit, matching R)
///
/// DESeq2's replaceOutliers logic:
/// - trimBaseMean = apply(normalized_counts, 1, mean, trim=0.2) - trimmed mean across ALL samples
/// - replacement = trimBaseMean * sizeFactor[outlier_sample]
/// - Only replaces in samples with >= minReplicatesForReplace replicates in their condition
/// - R uses as.integer() which truncates (floor), not round
pub fn replace_outliers(
    dds: &mut crate::data::DESeqDataSet,
    cooks: &Array2<f64>,
    cutoff: f64,
    min_replicates_for_replace: usize,
    trim: f64,
) -> crate::error::Result<(Vec<usize>, Vec<usize>)> {
    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();

    // Check if we have enough replicates overall
    if n_samples < min_replicates_for_replace {
        return Ok((Vec::new(), Vec::new()));
    }

    // Get size factors
    let size_factors = match dds.size_factors() {
        Some(sf) => sf.to_vec(),
        None => vec![1.0; n_samples],
    };

    // Get normalized counts for trimmed mean calculation
    let norm_counts = match dds.normalized_counts() {
        Some(nc) => nc.clone(),
        None => {
            let counts = dds.counts().counts();
            let mut nc = counts.to_owned();
            for i in 0..n_genes {
                for j in 0..n_samples {
                    if size_factors[j] > 0.0 {
                        nc[[i, j]] /= size_factors[j];
                    }
                }
            }
            nc
        }
    };

    // Check which samples are replaceable (have minReplicatesForReplace in their condition)
    let design_var = dds.design_variable().to_string();
    let conditions: Vec<String> = dds
        .sample_metadata()
        .condition(&design_var)
        .cloned()
        .unwrap_or_else(|| vec!["unknown".to_string(); n_samples]);

    // Count samples per condition
    let mut condition_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for cond in &conditions {
        *condition_counts.entry(cond.clone()).or_insert(0) += 1;
    }

    // Samples are replaceable if their condition has >= minReplicatesForReplace samples
    let replaceable: Vec<bool> = conditions
        .iter()
        .map(|c| condition_counts.get(c).copied().unwrap_or(0) >= min_replicates_for_replace)
        .collect();

    // Calculate trimmed base mean for each gene
    // DESeq2: trimBaseMean = apply(counts(object, normalized=TRUE), 1, mean, trim=0.2)
    let mut trim_base_means = vec![0.0; n_genes];
    for gene_idx in 0..n_genes {
        let mut values: Vec<f64> = (0..n_samples)
            .map(|j| norm_counts[[gene_idx, j]])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Number of values to trim from each end (20%)
        let k = (n_samples as f64 * trim).floor() as usize;
        let trimmed = &values[k..n_samples - k];
        if !trimmed.is_empty() {
            trim_base_means[gene_idx] = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
        } else {
            trim_base_means[gene_idx] = values.iter().sum::<f64>() / n_samples as f64;
        }
    }

    let mut modified_genes = Vec::new();

    for gene_idx in 0..n_genes {
        let mut gene_modified = false;

        for sample_idx in 0..n_samples {
            let c = cooks[[gene_idx, sample_idx]];
            if c.is_finite() && c > cutoff && replaceable[sample_idx] {
                // DESeq2: replacement = as.integer(trimBaseMean * sizeFactor)
                // R's as.integer() truncates toward zero (equivalent to floor for positive values)
                let replacement = trim_base_means[gene_idx] * size_factors[sample_idx];
                dds.replace_count(gene_idx, sample_idx, replacement.floor())?;
                gene_modified = true;
            }
        }

        if gene_modified {
            modified_genes.push(gene_idx);
        }
    }

    // Identify ALL genes with ANY Cook's outlier > cutoff (R flags these for refit)
    let mut flagged_genes = Vec::new();
    for gene_idx in 0..n_genes {
        let has_outlier = (0..n_samples).any(|sample_idx| {
            let c = cooks[[gene_idx, sample_idx]];
            c.is_finite() && c > cutoff
        });
        if has_outlier {
            flagged_genes.push(gene_idx);
        }
    }

    Ok((modified_genes, flagged_genes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cutoff() {
        let cutoff = default_cooks_cutoff(10, 2);
        assert!(cutoff > 0.0);
        assert!(cutoff < 100.0);
    }

    #[test]
    fn test_trimmed_mean() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tm = trimmed_mean(&mut values, 1.0 / 8.0);
        // With 1/8 trim and 8 values, trim 1 from each end: mean of [2,3,4,5,6,7] = 4.5
        assert!((tm - 4.5).abs() < 1e-10);
    }
}
