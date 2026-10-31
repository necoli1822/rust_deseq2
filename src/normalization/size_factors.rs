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
    let size_factors = match method {
        SizeFactorMethod::Ratio => estimate_size_factors_ratio(dds.counts().counts())?,
        SizeFactorMethod::PosCounts => estimate_size_factors_poscounts(dds.counts().counts())?,
        SizeFactorMethod::Iterate => estimate_size_factors_iterate(dds)?,
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

    // R: exp(median((log(cnts) - loggeomeans)[is.finite(loggeomeans) & cnts > 0]))
    // The median is taken on the log scale, which differs from the linear-scale
    // median when the number of ratios is even.
    for j in 0..n_samples {
        let mut log_ratios: Vec<f64> = valid_genes
            .iter()
            .zip(geo_means.iter())
            .filter_map(|(&i, &geo_mean)| {
                let count = counts[[i, j]];
                if count > 0.0 && geo_mean > 0.0 {
                    Some(count.ln() - geo_mean.ln())
                } else {
                    None
                }
            })
            .collect();

        if log_ratios.is_empty() {
            return Err(DeseqError::SizeFactorFailed {
                reason: format!("Sample {} has no positive counts in genes with a positive geometric mean", j),
            });
        }

        log_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = log_ratios.len();
        let median = if m % 2 == 0 {
            (log_ratios[m / 2 - 1] + log_ratios[m / 2]) / 2.0
        } else {
            log_ratios[m / 2]
        };

        size_factors[j] = median.exp();
    }

    // R passes the poscounts geometric means in as `geoMeans`, and incoming geoMeans
    // trigger: sf <- sf / exp(mean(log(sf))), i.e. size factors with geometric mean 1.
    let log_geo_mean_sf = size_factors.iter().map(|s: &f64| s.ln()).sum::<f64>() / n_samples as f64;
    size_factors.mapv_inplace(|s| s / log_geo_mean_sf.exp());

    Ok(size_factors)
}

/// Iterative size factor estimation for data where every gene has a zero
/// R equivalent: estimateSizeFactorsIterate(object, niter = 10, Q = 0.05) in core.R
///
/// With design ~1, alternately (1) estimate dispersions with fitType = "mean" given
/// the current size factors, and (2) choose the size factors maximising the summed
/// negative binomial log-likelihood of all genes above the Q = 5% quantile of
/// per-gene log-likelihood, by optim(log(sf), fn, method = "L-BFGS-B",
/// control = list(fnscale = -1)) with sf = exp(p - mean(p)). Stops when the squared
/// change in log size factors is below 1e-4 (after at least two iterations).
fn estimate_size_factors_iterate(dds: &DESeqDataSet) -> Result<Array1<f64>> {
    use crate::dispersion::{estimate_dispersions, DispersionParams, TrendFitMethod};
    use crate::optim::nmath::{dnbinom_mu_log, r_sum};
    use crate::optim::{optim_lbfgsb, LbfgsbControl};

    const NITER: usize = 10;
    const Q: f64 = 0.05;
    let n_samples = dds.n_samples();
    let counts = dds.counts().counts();
    // idx <- rowSums(counts(object)) > 0
    let idx: Vec<usize> = (0..dds.n_genes())
        .filter(|&i| (0..n_samples).any(|j| counts[[i, j]] > 0.0))
        .collect();
    if idx.is_empty() {
        return Err(DeseqError::SizeFactorFailed { reason: "all genes have zero counts".to_string() });
    }

    // design(object) <- ~ 1
    let mut work = dds.clone();
    work.set_design_matrix(ndarray::Array2::from_elem((n_samples, 1), 1.0))?;

    let r_mean = |x: &[f64]| -> f64 {
        // R's mean(): long-double sum / n, then one refinement pass
        let n = x.len() as f64;
        let s = r_sum(x.iter().copied()) / n;
        s + r_sum(x.iter().map(|v| v - s)) / n
    };

    let mut sf = vec![1.0; n_samples];
    for iter in 1..=NITER {
        work.set_size_factors(Array1::from(sf.clone()))?;
        estimate_dispersions(&mut work, TrendFitMethod::Mean, &DispersionParams::default())?;
        let mu = work.mu().ok_or_else(|| DeseqError::SizeFactorFailed { reason: "mu not available".to_string() })?;
        let disps = work.dispersions().ok_or_else(|| DeseqError::SizeFactorFailed {
            reason: "dispersions not available".to_string(),
        })?;
        // q <- t(t(assays(object)[["mu"]]) / sf)[idx,]; disps <- dispersions(object)[idx]
        let q: Vec<Vec<f64>> = idx.iter().map(|&i| (0..n_samples).map(|j| mu[[i, j]] / sf[j]).collect()).collect();
        let size: Vec<f64> = idx.iter().map(|&i| 1.0 / disps[i]).collect();
        let y: Vec<Vec<f64>> = idx.iter().map(|&i| (0..n_samples).map(|j| counts[[i, j]]).collect()).collect();

        let objective = |p: &[f64]| -> f64 {
            let m = r_mean(p);
            let sf_new: Vec<f64> = p.iter().map(|v| (v - m).exp()).collect();
            let gene_ll: Vec<f64> = (0..idx.len())
                .map(|g| r_sum((0..n_samples).map(|j| dnbinom_mu_log(y[g][j], size[g], q[g][j] * sf_new[j]))))
                .collect();
            let threshold = quantile_type7(&gene_ll, Q);
            r_sum(gene_ll.iter().copied().filter(|&ll| ll > threshold))
        };

        let sf_old = sf.clone();
        let start: Vec<f64> = sf_old.iter().map(|v| v.ln()).collect();
        let control = LbfgsbControl { fnscale: -1.0, ..Default::default() };
        let res = optim_lbfgsb(&start, objective, None, &[f64::NEG_INFINITY], &[f64::INFINITY], &control)
            .map_err(|e| DeseqError::SizeFactorFailed { reason: format!("iterative size factors: {:?}", e) })?;
        if res.convergence != 0 {
            return Err(DeseqError::SizeFactorFailed {
                reason: "iterative size factor normalization did not converge within an iteration".to_string(),
            });
        }
        let m = r_mean(&res.par);
        sf = res.par.iter().map(|v| (v - m).exp()).collect();

        let change: f64 = sf_old.iter().zip(&sf).map(|(a, b)| (a.ln() - b.ln()).powi(2)).sum();
        if iter > 1 && change < 1e-4 {
            break;
        } else if iter == NITER {
            return Err(DeseqError::SizeFactorFailed {
                reason: "iterative size factor normalization did not converge".to_string(),
            });
        }
    }
    Ok(Array1::from(sf))
}

/// R's quantile(x, p) with the default type 7
fn quantile_type7(x: &[f64], p: f64) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = (v.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
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
