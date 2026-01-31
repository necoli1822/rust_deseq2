//! Size factor estimation using the median of ratios method

use ndarray::{Array1, ArrayView2, Axis};

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};

/// Method for size factor estimation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeFactorMethod {
    /// Standard median of ratios (DESeq2 default)
    Ratio,
    /// Modified method using positive counts only for geometric mean
    PosCounts,
    /// Iterative method for data with many zeros
    Iterate,
}

/// Estimate size factors using the median of ratios method
///
/// This is the standard DESeq2 normalization method that accounts for
/// both sequencing depth and RNA composition bias.
pub fn estimate_size_factors(dds: &mut DESeqDataSet, method: SizeFactorMethod) -> Result<()> {
    let counts = dds.counts().counts();

    let size_factors = match method {
        SizeFactorMethod::Ratio => estimate_size_factors_ratio(counts)?,
        SizeFactorMethod::PosCounts => estimate_size_factors_poscounts(counts)?,
        SizeFactorMethod::Iterate => estimate_size_factors_iterate(counts)?,
    };

    dds.set_size_factors(size_factors)?;
    Ok(())
}

/// Standard median of ratios method
fn estimate_size_factors_ratio(counts: ArrayView2<f64>) -> Result<Array1<f64>> {
    let (n_genes, n_samples) = counts.dim();

    if n_genes == 0 || n_samples == 0 {
        return Err(DeseqError::EmptyData {
            reason: "Count matrix is empty".to_string(),
        });
    }

    // Step 1: Calculate geometric mean for each gene across all samples
    let mut geo_means = Vec::with_capacity(n_genes);
    let mut valid_genes = Vec::new();

    for (i, row) in counts.axis_iter(Axis(0)).enumerate() {
        // Check if all counts are positive (skip genes with zeros)
        if row.iter().all(|&x| x > 0.0) {
            let log_sum: f64 = row.iter().map(|&x| x.ln()).sum();
            let geo_mean = (log_sum / n_samples as f64).exp();
            geo_means.push(geo_mean);
            valid_genes.push(i);
        }
    }

    if valid_genes.is_empty() {
        return Err(DeseqError::SizeFactorFailed {
            reason: "No genes with all non-zero counts found".to_string(),
        });
    }

    // Step 2: For each sample, calculate ratios and take median
    let mut size_factors = Array1::zeros(n_samples);

    for j in 0..n_samples {
        let mut ratios: Vec<f64> = valid_genes
            .iter()
            .zip(geo_means.iter())
            .filter_map(|(&i, &geo_mean)| {
                let count = counts[[i, j]];
                if count > 0.0 && geo_mean > 0.0 {
                    Some(count / geo_mean)
                } else {
                    None
                }
            })
            .collect();

        if ratios.is_empty() {
            return Err(DeseqError::SizeFactorFailed {
                reason: format!("No valid ratios for sample {}", j),
            });
        }

        // Calculate median
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if ratios.len() % 2 == 0 {
            (ratios[ratios.len() / 2 - 1] + ratios[ratios.len() / 2]) / 2.0
        } else {
            ratios[ratios.len() / 2]
        };

        size_factors[j] = median;
    }

    // Validate size factors
    if size_factors.iter().any(|&x| x <= 0.0 || !x.is_finite()) {
        return Err(DeseqError::SizeFactorFailed {
            reason: "Invalid size factors computed".to_string(),
        });
    }

    Ok(size_factors)
}

/// Modified method using positive counts for geometric mean calculation
fn estimate_size_factors_poscounts(counts: ArrayView2<f64>) -> Result<Array1<f64>> {
    let (n_genes, n_samples) = counts.dim();

    if n_genes == 0 || n_samples == 0 {
        return Err(DeseqError::EmptyData {
            reason: "Count matrix is empty".to_string(),
        });
    }

    // Calculate geometric mean using only positive counts
    let mut geo_means = Vec::with_capacity(n_genes);
    let mut valid_genes = Vec::new();

    for (i, row) in counts.axis_iter(Axis(0)).enumerate() {
        let positive_counts: Vec<f64> = row.iter().filter(|&&x| x > 0.0).copied().collect();

        if !positive_counts.is_empty() {
            // R: exp(sum(log(x[x > 0])) / length(x)) — divide by TOTAL samples, not just positive
            let log_sum: f64 = positive_counts.iter().map(|&x| x.ln()).sum();
            let geo_mean = (log_sum / n_samples as f64).exp();
            geo_means.push(geo_mean);
            valid_genes.push(i);
        }
    }

    if valid_genes.is_empty() {
        return Err(DeseqError::SizeFactorFailed {
            reason: "No genes with positive counts found".to_string(),
        });
    }

    // Calculate size factors
    let mut size_factors = Array1::zeros(n_samples);

    for j in 0..n_samples {
        let mut ratios: Vec<f64> = valid_genes
            .iter()
            .zip(geo_means.iter())
            .filter_map(|(&i, &geo_mean)| {
                let count = counts[[i, j]];
                if count > 0.0 && geo_mean > 0.0 {
                    Some(count / geo_mean)
                } else {
                    None
                }
            })
            .collect();

        if ratios.is_empty() {
            // Use 1.0 as default for samples with no valid ratios
            size_factors[j] = 1.0;
            continue;
        }

        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if ratios.len() % 2 == 0 {
            (ratios[ratios.len() / 2 - 1] + ratios[ratios.len() / 2]) / 2.0
        } else {
            ratios[ratios.len() / 2]
        };

        size_factors[j] = median;
    }

    Ok(size_factors)
}

/// Iterative method for sparse data
///
/// NOTE: This implementation differs from R's estimateSizeFactorsIterate (core.R:2609-2640).
/// R uses NB-likelihood optimization via IRLS fitting per gene to get fitted values,
/// then computes new size factors from geometric means of (count/mu) ratios.
/// This implementation uses iterative poscounts recalculation as an approximation.
fn estimate_size_factors_iterate(counts: ArrayView2<f64>) -> Result<Array1<f64>> {
    // Start with poscounts method
    let mut size_factors = estimate_size_factors_poscounts(counts)?;
    let n_samples = counts.ncols();

    // Iterate to refine
    for _ in 0..10 {
        // Normalize counts
        let mut normalized = counts.to_owned();
        for j in 0..n_samples {
            if size_factors[j] > 0.0 {
                for i in 0..counts.nrows() {
                    normalized[[i, j]] /= size_factors[j];
                }
            }
        }

        // Recalculate size factors
        let new_sf = estimate_size_factors_poscounts(normalized.view())?;

        // Update and check convergence
        let mut max_diff = 0.0f64;
        for j in 0..n_samples {
            let adjustment = new_sf[j];
            let new_val = size_factors[j] * adjustment;
            max_diff = max_diff.max((new_val - size_factors[j]).abs());
            size_factors[j] = new_val;
        }

        if max_diff < 1e-6 {
            break;
        }
    }

    // Re-center (geometric mean of size factors = 1)
    let log_mean: f64 = size_factors.iter().map(|&x| x.ln()).sum::<f64>() / n_samples as f64;
    let center = log_mean.exp();
    size_factors.mapv_inplace(|x| x / center);

    Ok(size_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, SampleMetadata};
    use ndarray::array;

    fn create_test_dds() -> DESeqDataSet {
        let counts = CountMatrix::new(
            array![
                [100.0, 200.0, 80.0, 160.0],
                [500.0, 1000.0, 400.0, 800.0],
                [50.0, 100.0, 40.0, 80.0],
                [200.0, 400.0, 160.0, 320.0]
            ],
            vec![
                "gene1".to_string(),
                "gene2".to_string(),
                "gene3".to_string(),
                "gene4".to_string(),
            ],
            vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()],
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
                "condition",
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                ],
            )
            .unwrap();

        DESeqDataSet::new(counts, metadata, "condition").unwrap()
    }

    #[test]
    fn test_size_factor_estimation() {
        let mut dds = create_test_dds();
        estimate_size_factors(&mut dds, SizeFactorMethod::Ratio).unwrap();

        let sf = dds.size_factors().unwrap();
        assert_eq!(sf.len(), 4);

        // All size factors should be positive
        assert!(sf.iter().all(|&x| x > 0.0));

        // Size factors should reflect the 2x difference in "sequencing depth"
        // s2 has ~2x counts of s1, s4 has ~2x counts of s3
        let ratio = sf[1] / sf[0];
        assert!((ratio - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_normalized_counts() {
        let mut dds = create_test_dds();
        estimate_size_factors(&mut dds, SizeFactorMethod::Ratio).unwrap();

        let norm_counts = dds.normalized_counts().unwrap();

        // After normalization, the expression levels should be similar across samples
        // for genes that aren't differentially expressed
        let gene1_norm: Vec<f64> = norm_counts.row(0).to_vec();
        let mean = gene1_norm.iter().sum::<f64>() / 4.0;

        for val in gene1_norm {
            assert!((val - mean).abs() / mean < 0.1);
        }
    }
}
