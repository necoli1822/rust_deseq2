//! MAP (Maximum A Posteriori) dispersion estimation
//!
//! This implements DESeq2's empirical Bayes shrinkage for dispersions.
//! Uses line search optimization to find the posterior mode, matching
//! DESeq2's C++ fitDisp function exactly.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::data::DESeqDataSet;
use crate::dispersion::DispersionParams;
use crate::error::{DeseqError, Result};

/// Estimate MAP dispersions by shrinking gene-wise estimates toward the trend
/// R equivalent: estimateDispersionsMAP() in core.R
///
/// IMPORTANT: This function reuses the mu values computed during gene-wise dispersion
/// estimation, matching DESeq2's behavior exactly. DESeq2 stores mu in assays[["mu"]]
/// and reuses it for MAP estimation.
pub fn estimate_map_dispersions(dds: &mut DESeqDataSet, params: &DispersionParams) -> Result<()> {
    // Clone data upfront to avoid borrow conflicts with the mutable set_dispersion_prior_var call
    let gene_dispersions = dds.gene_dispersions().ok_or_else(|| {
        DeseqError::DispersionEstimationFailed {
            gene_id: "N/A".to_string(),
            reason: "Gene-wise dispersions required".to_string(),
        }
    })?.clone();

    let trended_dispersions = dds.trended_dispersions().ok_or_else(|| {
        DeseqError::DispersionEstimationFailed {
            gene_id: "N/A".to_string(),
            reason: "Trended dispersions required".to_string(),
        }
    })?.clone();

    // Get mu matrix from gene-wise dispersion estimation (DESeq2 compatibility)
    let mu_matrix_owned = dds.mu().ok_or_else(|| {
        DeseqError::DispersionEstimationFailed {
            gene_id: "N/A".to_string(),
            reason: "mu matrix required (computed during gene-wise dispersion estimation)".to_string(),
        }
    })?.clone();

    let n_samples = dds.n_samples();
    let n_genes = dds.n_genes();
    let counts = dds.counts().counts().to_owned();

    // Get design matrix from gene-wise dispersion estimation
    // This ensures we use the same design matrix for prior variance calculation
    let design = dds.design_matrix().ok_or_else(|| {
        DeseqError::DispersionEstimationFailed {
            gene_id: "N/A".to_string(),
            reason: "Design matrix required (computed during gene-wise dispersion estimation)".to_string(),
        }
    })?.clone();

    let n_coef = design.ncols();

    // Estimate prior variance from the data (also get varLogDispEsts for outlier detection)
    let (prior_var, var_log_disp_ests) = estimate_prior_variance_with_var(
        gene_dispersions.as_slice().unwrap(),
        trended_dispersions.as_slice().unwrap(),
        n_samples,
        n_coef,
    );

    log::debug!("MAP shrinkage prior variance: {:.6}, varLogDispEsts: {:.6}", prior_var, var_log_disp_ests);

    // Cache prior variance and varLogDispEsts for refit and downstream use
    dds.set_dispersion_prior_var(prior_var);
    dds.set_var_log_disp_ests(var_log_disp_ests);

    // Calculate max_disp = max(n_samples, 10) matching R's DESeq2
    let max_disp = (n_samples as f64).max(10.0);

    // Calculate MAP estimates in parallel using DESeq2's exact optimization
    // Use mu from gene-wise estimation (stored in dds.mu()) instead of recalculating
    let map_dispersions: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let gene_counts: Vec<f64> = (0..n_samples).map(|j| counts[[i, j]]).collect();
            let gene_disp = gene_dispersions[i];
            let trend_disp = trended_dispersions[i];

            // Use stored mu from gene-wise estimation (DESeq2 compatibility)
            let mu: Vec<f64> = (0..n_samples).map(|j| mu_matrix_owned[[i, j]]).collect();

            fit_map_dispersion_with_params(
                &gene_counts,
                &design,
                &mu,
                gene_disp,
                trend_disp,
                prior_var,
                max_disp,
                params,
            )
        })
        .collect();

    // Apply DESeq2's dispersion outlier detection (fitDispOutliers)
    // Outlier detection uses: outlierSD = params.outlier_sd, threshold = outlierSD * sqrt(varLogDispEsts)
    let outlier_sd = params.outlier_sd;
    let outlier_threshold = outlier_sd * var_log_disp_ests.sqrt();

    let final_dispersions: Vec<f64> = (0..n_genes)
        .map(|i| {
            let gene_disp = gene_dispersions[i];
            let trend_disp = trended_dispersions[i];
            let map_disp = map_dispersions[i];

            // Skip outlier detection for boundary genes or invalid values
            if !gene_disp.is_finite() || !trend_disp.is_finite() || gene_disp <= 0.0 || trend_disp <= 0.0 {
                return map_disp;
            }

            let log_diff = gene_disp.ln() - trend_disp.ln();

            // Upper outlier: gene-wise dispersion is much higher than trend
            // Use gene-wise dispersion instead of MAP (no shrinkage)
            // NOTE: DESeq2 only handles upper outliers specially. Lower outliers
            // go through normal MAP optimization (they don't use trended directly).
            if log_diff > outlier_threshold {
                log::debug!(
                    "Gene {}: Upper outlier detected (log_diff={:.4} > threshold={:.4}), using gene-wise disp={:.6}",
                    i, log_diff, outlier_threshold, gene_disp
                );
                return gene_disp;
            }

            map_disp
        })
        .collect();

    // Track which genes are dispersion outliers (R's dispOutlier)
    let disp_outliers: Vec<bool> = (0..n_genes)
        .map(|i| {
            let gene_disp = gene_dispersions[i];
            let trend_disp = trended_dispersions[i];
            gene_disp.is_finite() && trend_disp.is_finite() && gene_disp > 0.0 && trend_disp > 0.0
                && (gene_disp.ln() - trend_disp.ln()) > outlier_threshold
        })
        .collect();

    let n_upper_outliers = disp_outliers.iter().filter(|&&x| x).count();

    log::debug!(
        "Dispersion upper outliers: {} (threshold={:.4})",
        n_upper_outliers, outlier_threshold
    );

    dds.set_dispersion_outliers(disp_outliers);
    dds.set_map_dispersions(Array1::from_vec(final_dispersions))?;
    Ok(())
}

/// Estimate the prior variance for dispersion shrinkage
/// R equivalent: estimateDispersionsPriorVar() in core.R
///
/// DESeq2's exact formula (from methods.R line 180):
/// 1. Filter genes with dispGeneEst >= minDisp * 100 (exclude boundary genes)
/// 2. varLogDispEsts = mad(log(dispGeneEst) - log(dispFit))^2
///    where mad = median(|X - median(X)|) * 1.4826
/// 3. expVarLogDisp = trigamma((n_samples - n_coef) / 2)
/// 4. dispPriorVar = max(varLogDispEsts - expVarLogDisp, 0.25)
///
/// SPECIAL CASE: When (n_samples - n_coef) <= 3, DESeq2 uses KL divergence
/// based estimation instead of the MAD^2 method.
///
/// Returns (dispPriorVar, varLogDispEsts) for use in outlier detection
pub fn estimate_prior_variance_with_var(
    gene_dispersions: &[f64],
    trended_dispersions: &[f64],
    n_samples: usize,
    n_coef: usize,
) -> (f64, f64) {
    const MIN_DISP: f64 = 1e-8;
    const MIN_DISP_THRESHOLD: f64 = MIN_DISP * 100.0;  // 1e-6

    // Compute log residuals for genes ABOVE the minimum dispersion threshold
    let mut log_residuals: Vec<f64> = gene_dispersions
        .iter()
        .zip(trended_dispersions.iter())
        .filter(|(&g, &t)| {
            g >= MIN_DISP_THRESHOLD && t > 0.0 && g.is_finite() && t.is_finite()
        })
        .map(|(&g, &t)| g.ln() - t.ln())
        .collect();

    if log_residuals.len() < 3 {
        return (0.25, 0.25); // Minimum default for both
    }

    // Calculate MAD^2 (Median Absolute Deviation squared)
    // This matches DESeq2's exact formula: mad(dispResiduals)^2
    let var_log_disp_ests = mad_squared(&mut log_residuals);

    // When m == p, degrees of freedom is 0 and trigamma is undefined.
    // R handles this by returning the minimum prior variance.
    if n_samples <= n_coef {
        return (0.25, var_log_disp_ests);
    }

    let df = n_samples as f64 - n_coef as f64;

    // DESeq2 special case: use KL divergence method for low sample sizes
    // Condition: (m - p) <= 3 && (m > p)
    let prior_var = if df <= 3.0 && n_samples > n_coef {
        estimate_prior_var_kl_divergence(&log_residuals, df)
    } else {
        // Standard MAD^2 method
        let exp_var_log_disp = trigamma(df / 2.0);
        (var_log_disp_ests - exp_var_log_disp).max(0.25)
    };

    log::debug!(
        "Prior var: n_samples={}, df={}, varLogDispEsts={:.4}, dispPriorVar={:.4}",
        n_samples, df, var_log_disp_ests, prior_var
    );

    (prior_var, var_log_disp_ests)
}

/// Estimate prior variance using KL divergence method
/// This matches DESeq2's estimateDispersionsPriorVar for low sample sizes (df <= 3)
///
/// Algorithm:
/// 1. Create histogram of observed residuals
/// 2. For each candidate prior variance, simulate from chi-squared + normal
/// 3. Find prior variance that minimizes KL divergence
fn estimate_prior_var_kl_divergence(log_residuals: &[f64], df: f64) -> f64 {
    use crate::rng::RMersenneTwister;

    // Use R-compatible RNG with set.seed(2) for exact reproducibility
    let mut rng = RMersenneTwister::new(2);

    // Histogram breaks: -10 to 10 with step 0.5 (matching R's -20:20/2)
    let breaks: Vec<f64> = (-20..=20).map(|i| i as f64 / 2.0).collect();

    // Filter observed residuals to histogram range
    let obs_dist: Vec<f64> = log_residuals
        .iter()
        .copied()
        .filter(|&x| x > breaks[0] && x < breaks[breaks.len() - 1])
        .collect();

    if obs_dist.is_empty() {
        return 0.25;
    }

    // Create observed histogram
    let obs_hist = create_histogram(&obs_dist, &breaks);
    let obs_density = normalize_histogram(&obs_hist, obs_dist.len());

    // Grid of candidate prior variances (0 to 8, 200 points)
    let obs_var_grid: Vec<f64> = (0..200).map(|i| i as f64 * 8.0 / 199.0).collect();

    // Calculate KL divergence for each candidate using R-compatible RNG
    let n_sim = 10000;

    let kl_divs: Vec<f64> = obs_var_grid
        .iter()
        .map(|&prior_var| {
            // Simulate: log(chi-squared(df)) + Normal(0, sqrt(prior_var)) - log(df)
            let prior_sd = prior_var.sqrt().max(1e-10);

            let rand_dist: Vec<f64> = (0..n_sim)
                .map(|_| {
                    let chi_val = rng.rchisq(df);
                    let norm_val = rng.rnorm() * prior_sd;
                    chi_val.ln() + norm_val - df.ln()
                })
                .filter(|&x| x > breaks[0] && x < breaks[breaks.len() - 1])
                .collect();

            if rand_dist.is_empty() {
                return f64::MAX;
            }

            let rand_hist = create_histogram(&rand_dist, &breaks);
            let rand_density = normalize_histogram(&rand_hist, rand_dist.len());

            // KL divergence with small value to avoid log(0)
            let small = obs_density
                .iter()
                .chain(rand_density.iter())
                .filter(|&&x| x > 0.0)
                .fold(f64::MAX, |a, &b| a.min(b));

            obs_density
                .iter()
                .zip(rand_density.iter())
                .map(|(&o, &r)| {
                    if o > 0.0 {
                        o * ((o + small).ln() - (r + small).ln())
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect();

    // Loess smoothing with span=0.2 (matches DESeq2)
    let _smoothed = loess_smooth(&obs_var_grid, &kl_divs, 0.2);

    // Find argmin on fine grid (predict using loess on fine grid)
    let fine_grid: Vec<f64> = (0..1000).map(|i| i as f64 * 8.0 / 999.0).collect();
    let fine_smoothed = loess_predict(&obs_var_grid, &kl_divs, &fine_grid, 0.2);

    let (argmin_idx, _) = fine_smoothed
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &f64::MAX));

    let argmin_kl = fine_grid[argmin_idx];

    log::debug!(
        "KL divergence method: df={:.1}, argminKL={:.4}",
        df, argmin_kl
    );

    argmin_kl.max(0.25)
}

/// Create histogram counts for given breaks
fn create_histogram(values: &[f64], breaks: &[f64]) -> Vec<usize> {
    let n_bins = breaks.len() - 1;
    let mut counts = vec![0usize; n_bins];

    for &v in values {
        for i in 0..n_bins {
            if v >= breaks[i] && v < breaks[i + 1] {
                counts[i] += 1;
                break;
            }
        }
    }
    counts
}

/// Normalize histogram to density
fn normalize_histogram(counts: &[usize], total: usize) -> Vec<f64> {
    if total == 0 {
        return vec![0.0; counts.len()];
    }
    let bin_width = 0.5; // breaks are 0.5 apart
    counts
        .iter()
        .map(|&c| c as f64 / (total as f64 * bin_width))
        .collect()
}

/// Loess smoothing (Locally Estimated Scatterplot Smoothing)
/// Matches R's loess with span=0.2 and degree=2 (quadratic)
fn loess_smooth(x: &[f64], y: &[f64], span: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Number of points to use in each local fit
    let k = ((n as f64 * span).round() as usize).max(3).min(n);

    (0..n)
        .map(|i| {
            let x_i = x[i];

            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| y[j].is_finite() && y[j] < f64::MAX)
                .map(|j| (j, (x[j] - x_i).abs()))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<(usize, f64)> = distances.into_iter().take(k).collect();

            if neighbors.is_empty() {
                return y[i];
            }

            // Maximum distance among neighbors
            let max_dist = neighbors.iter().map(|(_, d)| *d).fold(0.0_f64, |a, b| a.max(b));
            let max_dist = if max_dist < 1e-10 { 1.0 } else { max_dist };

            // Tricube weights: w(u) = (1 - |u|^3)^3 for |u| < 1
            let weights: Vec<f64> = neighbors.iter()
                .map(|(_, d)| {
                    let u = d / max_dist;
                    if u < 1.0 {
                        let t = 1.0 - u.powi(3);
                        t.powi(3)
                    } else {
                        0.0
                    }
                })
                .collect();

            // Weighted local quadratic regression: y = a + b*x + c*x^2
            // Using weighted least squares
            let x_vals: Vec<f64> = neighbors.iter().map(|(j, _)| x[*j]).collect();
            let y_vals: Vec<f64> = neighbors.iter().map(|(j, _)| y[*j]).collect();

            // Center x for numerical stability
            let x_center = x_vals.iter().sum::<f64>() / x_vals.len() as f64;
            let x_centered: Vec<f64> = x_vals.iter().map(|&xj| xj - x_center).collect();
            let x_i_centered = x_i - x_center;

            // Weighted sums for normal equations
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wx2 = 0.0;
            let mut sum_wx3 = 0.0;
            let mut sum_wx4 = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wxy = 0.0;
            let mut sum_wx2y = 0.0;

            for (idx, &w) in weights.iter().enumerate() {
                let xc = x_centered[idx];
                let yv = y_vals[idx];
                sum_w += w;
                sum_wx += w * xc;
                sum_wx2 += w * xc * xc;
                sum_wx3 += w * xc * xc * xc;
                sum_wx4 += w * xc * xc * xc * xc;
                sum_wy += w * yv;
                sum_wxy += w * xc * yv;
                sum_wx2y += w * xc * xc * yv;
            }

            // Solve 3x3 system for quadratic fit: [a, b, c]
            // |sum_w    sum_wx   sum_wx2 | |a|   |sum_wy  |
            // |sum_wx   sum_wx2  sum_wx3 | |b| = |sum_wxy |
            // |sum_wx2  sum_wx3  sum_wx4 | |c|   |sum_wx2y|

            let det = sum_w * (sum_wx2 * sum_wx4 - sum_wx3 * sum_wx3)
                    - sum_wx * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
                    + sum_wx2 * (sum_wx * sum_wx3 - sum_wx2 * sum_wx2);

            if det.abs() < 1e-10 {
                // Fall back to weighted mean
                return sum_wy / sum_w.max(1e-10);
            }

            // Cramer's rule for a
            let det_a = sum_wy * (sum_wx2 * sum_wx4 - sum_wx3 * sum_wx3)
                      - sum_wx * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
                      + sum_wx2 * (sum_wxy * sum_wx3 - sum_wx2 * sum_wx2y);

            // Cramer's rule for b
            let det_b = sum_w * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
                      - sum_wy * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
                      + sum_wx2 * (sum_wx * sum_wx2y - sum_wxy * sum_wx2);

            // Cramer's rule for c
            let det_c = sum_w * (sum_wx2 * sum_wx2y - sum_wxy * sum_wx3)
                      - sum_wx * (sum_wx * sum_wx2y - sum_wxy * sum_wx2)
                      + sum_wy * (sum_wx * sum_wx3 - sum_wx2 * sum_wx2);

            let a = det_a / det;
            let b = det_b / det;
            let c = det_c / det;

            // Predict at x_i
            a + b * x_i_centered + c * x_i_centered * x_i_centered
        })
        .collect()
}

/// Loess prediction at new x values
/// Predicts y values at new_x using loess fitted on (x, y)
fn loess_predict(x: &[f64], y: &[f64], new_x: &[f64], span: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![f64::NAN; new_x.len()];
    }

    // Number of points to use in each local fit
    let k = ((n as f64 * span).round() as usize).max(3).min(n);

    new_x.iter()
        .map(|&x_i| {
            // Find k nearest neighbors from original data
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| y[j].is_finite() && y[j] < f64::MAX)
                .map(|j| (j, (x[j] - x_i).abs()))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<(usize, f64)> = distances.into_iter().take(k).collect();

            if neighbors.is_empty() {
                return f64::NAN;
            }

            // Maximum distance among neighbors
            let max_dist = neighbors.iter().map(|(_, d)| *d).fold(0.0_f64, |a, b| a.max(b));
            let max_dist = if max_dist < 1e-10 { 1.0 } else { max_dist };

            // Tricube weights
            let weights: Vec<f64> = neighbors.iter()
                .map(|(_, d)| {
                    let u = d / max_dist;
                    if u < 1.0 {
                        let t = 1.0 - u.powi(3);
                        t.powi(3)
                    } else {
                        0.0
                    }
                })
                .collect();

            // Weighted local quadratic regression
            let x_vals: Vec<f64> = neighbors.iter().map(|(j, _)| x[*j]).collect();
            let y_vals: Vec<f64> = neighbors.iter().map(|(j, _)| y[*j]).collect();

            // Center x for numerical stability
            let x_center = x_vals.iter().sum::<f64>() / x_vals.len() as f64;
            let x_centered: Vec<f64> = x_vals.iter().map(|&xj| xj - x_center).collect();
            let x_i_centered = x_i - x_center;

            // Weighted sums
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wx2 = 0.0;
            let mut sum_wx3 = 0.0;
            let mut sum_wx4 = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wxy = 0.0;
            let mut sum_wx2y = 0.0;

            for (idx, &w) in weights.iter().enumerate() {
                let xc = x_centered[idx];
                let yv = y_vals[idx];
                sum_w += w;
                sum_wx += w * xc;
                sum_wx2 += w * xc * xc;
                sum_wx3 += w * xc * xc * xc;
                sum_wx4 += w * xc * xc * xc * xc;
                sum_wy += w * yv;
                sum_wxy += w * xc * yv;
                sum_wx2y += w * xc * xc * yv;
            }

            // Solve 3x3 system
            let det = sum_w * (sum_wx2 * sum_wx4 - sum_wx3 * sum_wx3)
                    - sum_wx * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
                    + sum_wx2 * (sum_wx * sum_wx3 - sum_wx2 * sum_wx2);

            if det.abs() < 1e-10 {
                return sum_wy / sum_w.max(1e-10);
            }

            let det_a = sum_wy * (sum_wx2 * sum_wx4 - sum_wx3 * sum_wx3)
                      - sum_wx * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
                      + sum_wx2 * (sum_wxy * sum_wx3 - sum_wx2 * sum_wx2y);
            let det_b = sum_w * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
                      - sum_wy * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
                      + sum_wx2 * (sum_wx * sum_wx2y - sum_wxy * sum_wx2);
            let det_c = sum_w * (sum_wx2 * sum_wx2y - sum_wxy * sum_wx3)
                      - sum_wx * (sum_wx * sum_wx2y - sum_wxy * sum_wx2)
                      + sum_wy * (sum_wx * sum_wx3 - sum_wx2 * sum_wx2);

            let a = det_a / det;
            let b = det_b / det;
            let c = det_c / det;

            a + b * x_i_centered + c * x_i_centered * x_i_centered
        })
        .collect()
}

/// Compute MAD^2 (Median Absolute Deviation squared)
/// MAD = median(|X - median(X)|) * 1.4826
/// Returns MAD^2 to match DESeq2's varLogDispEsts calculation
fn mad_squared(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    // Calculate median
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let median = if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };

    // Calculate absolute deviations from median
    let mut abs_devs: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();

    // Calculate median of absolute deviations
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_abs_dev = if n % 2 == 0 {
        (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
    } else {
        abs_devs[n / 2]
    };

    // MAD with scale factor for consistency with normal distribution
    // R's mad() uses constant = 1.4826
    let mad = med_abs_dev * 1.4826;

    mad * mad  // Return MAD^2
}

/// Trigamma function (derivative of digamma)
fn trigamma(x: f64) -> f64 {
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).powi(2) - trigamma(1.0 - x);
    }

    if x >= 8.0 {
        let mut result = 1.0 / x + 0.5 / (x * x);
        let x2 = x * x;
        result += 1.0 / (6.0 * x2 * x);
        result -= 1.0 / (30.0 * x2 * x2 * x);
        return result;
    }

    let mut result = 0.0;
    let mut z = x;
    while z < 8.0 {
        result += 1.0 / (z * z);
        z += 1.0;
    }
    result + trigamma(z)
}

/// Fit MAP dispersion using DESeq2's exact line search algorithm (fitDisp in C++)
/// R equivalent: fitDisp() (MAP mode) in core.R
///
/// This optimizes log_posterior = ll_part + prior_part + cr_term
/// using line search with Armijo rule
///
/// IMPORTANT: DESeq2 performs MAP shrinkage even for genes at the dispersion
/// boundary (1e-8). The prior pulls these genes toward the trended value.
pub fn fit_map_dispersion(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    gene_disp: f64,
    trend_disp: f64,
    prior_var: f64,
    max_disp: f64,
) -> f64 {
    fit_map_dispersion_with_params(counts, design, mu, gene_disp, trend_disp, prior_var, max_disp, &DispersionParams::default())
}

/// Fit MAP dispersion with configurable parameters
fn fit_map_dispersion_with_params(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    gene_disp: f64,
    trend_disp: f64,
    prior_var: f64,
    max_disp: f64,
    params: &DispersionParams,
) -> f64 {
    // Handle edge cases
    if trend_disp <= 0.0 || !trend_disp.is_finite() {
        return 0.1;
    }

    // For genes with invalid gene-wise dispersion, use trended
    if !gene_disp.is_finite() {
        return trend_disp;
    }

    // NOTE: DESeq2 still performs MAP optimization for genes at the boundary
    // The prior will pull them toward the trended value

    // DESeq2 parameters (from C++ source and R wrapper), now configurable via params
    // min_log_alpha in C++ is -30.0 for hard bounds, but R wrapper uses log(minDisp/10)
    // for the convergence break condition
    let min_disp = params.min_disp;
    let min_log_alpha_hard = -30.0;  // C++ hard bound
    let min_log_alpha = (min_disp / 10.0).ln();  // ≈ -20.72, used for convergence break
    let max_log_alpha = 10.0_f64;  // R's C++ DESeq2.cpp uses 10.0 as upper bound
    let epsilon = 1.0e-4;
    let maxit = params.maxit;
    let tol = params.disp_tol;
    let kappa_0 = params.kappa_0;

    // Prior parameters
    let log_alpha_prior_mean = trend_disp.ln();
    let log_alpha_prior_sigmasq = prior_var;

    // Initialize: DESeq2 uses dispGeneEst unless it's < 0.1 * dispFit (trended)
    // If dispGeneEst < 0.1 * dispFit, start at dispFit instead (core.R line 1020-1022)
    // This helps convergence for genes with very low gene-wise estimates
    let disp_init = if gene_disp > 0.1 * trend_disp {
        gene_disp
    } else {
        trend_disp
    };
    let mut log_alpha = disp_init.ln().max(min_log_alpha).min(max_log_alpha);

    // Calculate initial log-posterior and gradient
    let mut lp = log_posterior_with_prior(
        counts, design, mu, log_alpha,
        log_alpha_prior_mean, log_alpha_prior_sigmasq
    );
    let mut dlp = d_log_posterior_with_prior(
        counts, design, mu, log_alpha,
        log_alpha_prior_mean, log_alpha_prior_sigmasq
    );

    let mut kappa = kappa_0;
    let mut iter_accept = 0;
    let mut converged = false;

    for _iter in 0..maxit {
        // Propose new log_alpha along gradient direction
        let mut a_propose = log_alpha + kappa * dlp;

        // Apply hard bounds (DESeq2 C++ uses -30 and 10)
        if a_propose < min_log_alpha_hard {
            if dlp.abs() > 1e-10 {
                kappa = (min_log_alpha_hard - log_alpha) / dlp;
            }
            a_propose = min_log_alpha_hard;
        }
        if a_propose > max_log_alpha {
            if dlp.abs() > 1e-10 {
                kappa = (max_log_alpha - log_alpha) / dlp;
            }
            a_propose = max_log_alpha;
        }

        // Evaluate at proposed point (we maximize log_posterior, so theta = -log_posterior)
        let lp_new = log_posterior_with_prior(
            counts, design, mu, a_propose,
            log_alpha_prior_mean, log_alpha_prior_sigmasq
        );

        // Armijo condition: -lp_new <= -lp - kappa * epsilon * dlp^2
        // Equivalently: lp_new >= lp + kappa * epsilon * dlp^2 (since we maximize)
        let theta_kappa = -lp_new;
        let theta_hat_kappa = -lp - kappa * epsilon * dlp * dlp;

        if theta_kappa <= theta_hat_kappa {
            // Accept step
            iter_accept += 1;
            log_alpha = a_propose;
            let lpnew = log_posterior_with_prior(
                counts, design, mu, log_alpha,
                log_alpha_prior_mean, log_alpha_prior_sigmasq
            );
            let change = lpnew - lp;

            // DESeq2 convergence check: change < tol (not abs!)
            // Since we're maximizing, positive change means improvement
            if change < tol {
                converged = true;
                break;
            }

            // Check if at minimum bound (DESeq2 uses min_log_alpha from R wrapper)
            if log_alpha < min_log_alpha {
                converged = true;
                break;
            }

            lp = lpnew;

            // Update gradient
            dlp = d_log_posterior_with_prior(
                counts, design, mu, log_alpha,
                log_alpha_prior_mean, log_alpha_prior_sigmasq
            );

            // Adjust step size (DESeq2 style)
            kappa = (kappa * 1.1).min(kappa_0);
            if iter_accept % 5 == 0 {
                kappa /= 2.0;
            }
        } else {
            // Reject step, reduce kappa
            kappa /= 2.0;
        }
    }

    // If MAP optimization didn't converge, fall back to grid search over log-posterior
    // This matches R's behavior of refitting non-converged MAP genes
    if !converged {
        let grid_result = grid_search_map_dispersion(
            counts, design, mu, min_disp, max_disp,
            log_alpha_prior_mean, log_alpha_prior_sigmasq,
        );
        return grid_result.max(min_disp).min(max_disp);
    }

    log_alpha.exp().max(min_disp).min(max_disp)
}

/// Grid search for MAP dispersion when line search doesn't converge
/// Searches over dispersion values and picks the one with highest log posterior
fn grid_search_map_dispersion(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    min_disp: f64,
    max_disp: f64,
    log_alpha_prior_mean: f64,
    log_alpha_prior_sigmasq: f64,
) -> f64 {
    let min_log_alpha = min_disp.ln();
    let max_log_alpha = max_disp.ln();
    let n_grid = 20;

    // Coarse grid
    let delta = (max_log_alpha - min_log_alpha) / (n_grid - 1) as f64;
    let coarse_grid: Vec<f64> = (0..n_grid)
        .map(|i| min_log_alpha + i as f64 * delta)
        .collect();

    let coarse_lp: Vec<f64> = coarse_grid.iter()
        .map(|&la| log_posterior_with_prior(
            counts, design, mu, la,
            log_alpha_prior_mean, log_alpha_prior_sigmasq,
        ))
        .collect();

    let best_idx = coarse_lp.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_coarse = coarse_grid[best_idx];

    // Fine grid around best coarse value
    let fine_min = best_coarse - delta;
    let fine_max = best_coarse + delta;
    let fine_delta = (fine_max - fine_min) / (n_grid - 1) as f64;
    let fine_grid: Vec<f64> = (0..n_grid)
        .map(|i| fine_min + i as f64 * fine_delta)
        .collect();

    let fine_lp: Vec<f64> = fine_grid.iter()
        .map(|&la| log_posterior_with_prior(
            counts, design, mu, la,
            log_alpha_prior_mean, log_alpha_prior_sigmasq,
        ))
        .collect();

    let best_fine_idx = fine_lp.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    fine_grid[best_fine_idx].exp()
}

/// Log-posterior with prior (for MAP estimation)
/// Matches DESeq2's log_posterior function with usePrior=true
fn log_posterior_with_prior(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    log_alpha: f64,
    log_alpha_prior_mean: f64,
    log_alpha_prior_sigmasq: f64,
) -> f64 {
    let n = counts.len();
    let p = design.ncols();
    let alpha = log_alpha.exp();
    let alpha_inv = 1.0 / alpha;

    // Log-likelihood part
    let mut ll_part = 0.0;
    let mut weights = vec![0.0; n];

    for i in 0..n {
        let y = counts[i];
        let mu_i = mu[i].max(1e-10);

        // DESeq2's exact log-likelihood formula:
        // ll = lgamma(y + 1/alpha) - lgamma(1/alpha) - y*log(mu + 1/alpha) - (1/alpha)*log(1 + mu*alpha)
        ll_part += lgamma(y + alpha_inv) - lgamma(alpha_inv);
        ll_part -= y * (mu_i + alpha_inv).ln();
        ll_part -= alpha_inv * (1.0 + mu_i * alpha).ln();

        // Weight for Cox-Reid adjustment: w = 1 / (1/mu + alpha)
        weights[i] = 1.0 / (1.0 / mu_i + alpha);
    }

    // Cox-Reid adjustment: -0.5 * log|X'WX|
    let mut xtwx = vec![vec![0.0; p]; p];
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                xtwx[j][k] += weights[i] * design[[i, j]] * design[[i, k]];
            }
        }
    }

    let log_det = log_determinant(&xtwx);
    let cr_term = if log_det.is_finite() { -0.5 * log_det } else { 0.0 };

    // Prior part: -0.5 * (log_alpha - log_alpha_prior_mean)^2 / log_alpha_prior_sigmasq
    let prior_part = -0.5 * (log_alpha - log_alpha_prior_mean).powi(2) / log_alpha_prior_sigmasq;

    ll_part + prior_part + cr_term
}

/// Derivative of log-posterior w.r.t. log(alpha) with prior
///
/// DESeq2 exact formula (from DESeq2.cpp dlog_posterior):
/// ll_part = alpha^{-2} * sum(digamma(1/alpha) + log(1+mu*alpha) - mu*alpha/(1+mu*alpha)
///           - digamma(y+1/alpha) + y/(mu+1/alpha))
/// cr_term = -0.5 * trace(B^{-1} * dB)  where B = X'WX, dB = X'(dW/dalpha)X
/// result = (ll_part + cr_term) * alpha + prior_part
fn d_log_posterior_with_prior(
    counts: &[f64],
    design: &Array2<f64>,
    mu: &[f64],
    log_alpha: f64,
    log_alpha_prior_mean: f64,
    log_alpha_prior_sigmasq: f64,
) -> f64 {
    let alpha = log_alpha.exp();
    let alpha_neg1 = 1.0 / alpha;
    let alpha_neg2 = alpha_neg1 * alpha_neg1;
    let n = counts.len();
    let p = design.ncols();

    // DESeq2 exact ll_part formula (line 96):
    // ll_part = alpha^{-2} * sum(digamma(1/alpha) + log(1+mu*alpha) - mu*alpha/(1+mu*alpha)
    //                           - digamma(y+1/alpha) + y/(mu+1/alpha))
    let mut ll_sum = 0.0;
    for i in 0..n {
        let y = counts[i];
        let mu_i = mu[i].max(1e-10);

        ll_sum += digamma(alpha_neg1);                              // digamma(1/alpha)
        ll_sum += (1.0 + mu_i * alpha).ln();                        // log(1 + mu*alpha)
        ll_sum -= mu_i * alpha / (1.0 + mu_i * alpha);              // -mu*alpha/(1+mu*alpha)
        ll_sum -= digamma(y + alpha_neg1);                          // -digamma(y + 1/alpha)
        ll_sum += y / (mu_i + alpha_neg1);                          // y/(mu + 1/alpha)
    }
    let ll_part = alpha_neg2 * ll_sum;

    // Cox-Reid derivative (analytical, matching DESeq2 exactly)
    // w_i = 1/(1/mu_i + alpha)
    // dw_i/dalpha = -1/(1/mu_i + alpha)^2
    // B = X'WX, dB = X'(dW)X
    // d/dalpha[-0.5*log|B|] = -0.5 * trace(B^{-1} * dB)

    let mut w_diag = vec![0.0; n];
    let mut dw_diag = vec![0.0; n];
    for i in 0..n {
        let mu_i = mu[i].max(1e-10);
        let inv_mu_plus_alpha = 1.0 / mu_i + alpha;
        w_diag[i] = 1.0 / inv_mu_plus_alpha;
        dw_diag[i] = -1.0 / (inv_mu_plus_alpha * inv_mu_plus_alpha);
    }

    // Compute B = X'WX and dB = X'(dW)X
    let mut b = vec![vec![0.0; p]; p];
    let mut db = vec![vec![0.0; p]; p];
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                b[j][k] += w_diag[i] * design[[i, j]] * design[[i, k]];
                db[j][k] += dw_diag[i] * design[[i, j]] * design[[i, k]];
            }
        }
    }

    // Compute B^{-1} and trace(B^{-1} * dB)
    let cr_term = if let Some(b_inv) = matrix_inverse(&b) {
        // trace(B^{-1} * dB) = sum_ij B^{-1}[i,j] * dB[j,i]
        let mut trace_b_inv_db = 0.0;
        for i in 0..p {
            for j in 0..p {
                trace_b_inv_db += b_inv[i][j] * db[j][i];
            }
        }

        // d/dalpha[-0.5*log|B|] = -0.5 * trace(B^{-1} * dB)
        -0.5 * trace_b_inv_db
    } else {
        0.0
    };

    // Prior derivative: d/d(log_alpha) of -0.5 * (log_alpha - mean)^2 / sigmasq
    // = -(log_alpha - mean) / sigmasq
    let prior_part = -(log_alpha - log_alpha_prior_mean) / log_alpha_prior_sigmasq;

    // DESeq2 line 105: return (ll_part + cr_term) * alpha + prior_part
    // (ll_part and cr_term are w.r.t. alpha, multiply by alpha for chain rule)
    (ll_part + cr_term) * alpha + prior_part
}

/// Cox-Reid adjustment term: -0.5 * log|X'WX|
/// Compute log determinant using LU decomposition
fn log_determinant(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    // Clone the matrix for LU decomposition
    let mut lu: Vec<Vec<f64>> = m.to_vec();
    let mut sign = 1.0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[i][k].abs() > max_val {
                max_val = lu[i][k].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return f64::NEG_INFINITY; // Singular matrix
        }

        // Swap rows if needed
        if max_row != k {
            lu.swap(k, max_row);
            sign = -sign;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = lu[i][k] / lu[k][k];
            for j in k..n {
                lu[i][j] -= factor * lu[k][j];
            }
        }
    }

    // Log determinant is sum of log of diagonal elements
    // For positive definite matrices (X'WX should be), sign should always be positive
    if sign < 0.0 {
        return f64::NAN; // Negative determinant (shouldn't happen for X'WX)
    }

    let mut log_det = 0.0_f64;
    for i in 0..n {
        if lu[i][i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        log_det += lu[i][i].ln();
    }

    log_det
}

/// Compute matrix inverse using Gauss-Jordan elimination
fn matrix_inverse(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 {
        return None;
    }

    // Create augmented matrix [M | I]
    let mut aug: Vec<Vec<f64>> = m.iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.extend(std::iter::repeat(0.0).take(n));
            new_row[n + i] = 1.0;
            new_row
        })
        .collect();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = aug[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular matrix
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
        }

        // Scale pivot row
        let pivot = aug[k][k];
        for j in 0..(2 * n) {
            aug[k][j] /= pivot;
        }

        // Eliminate column
        for i in 0..n {
            if i != k {
                let factor = aug[i][k];
                for j in 0..(2 * n) {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }
    }

    // Extract inverse from right half
    Some(aug.iter().map(|row| row[n..].to_vec()).collect())
}

/// Digamma function
fn digamma(x: f64) -> f64 {
    statrs::function::gamma::digamma(x)
}

/// Log-gamma function
fn lgamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_shrinkage() {
        // Simple test case
        let counts = vec![100.0, 110.0, 90.0, 200.0, 210.0, 190.0];
        let design = Array2::from_shape_vec((6, 2), vec![
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
        ]).unwrap();
        let mu = vec![100.0, 100.0, 100.0, 200.0, 200.0, 200.0];

        let gene_disp = 0.5;
        let trend_disp = 0.1;
        let prior_var = 0.5;
        let max_disp = 10.0;

        let map = fit_map_dispersion(&counts, &design, &mu, gene_disp, trend_disp, prior_var, max_disp);

        // MAP should be reasonable (positive and bounded)
        assert!(map > 0.0, "MAP dispersion should be positive");
        assert!(map < 10.0, "MAP dispersion should be bounded");
    }

    #[test]
    fn test_prior_variance_estimation() {
        let gene = vec![0.1, 0.2, 0.15, 0.12, 0.18];
        let trend = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        let (prior_var, _) = estimate_prior_variance_with_var(&gene, &trend, 6, 2);

        // Should be positive
        assert!(prior_var > 0.0);
    }

    #[test]
    fn test_trigamma() {
        // trigamma(1) = pi^2/6 ≈ 1.6449
        let t1 = trigamma(1.0);
        assert!((t1 - std::f64::consts::PI.powi(2) / 6.0).abs() < 0.001);

        // trigamma(2) ≈ 0.6449
        let t2 = trigamma(2.0);
        assert!((t2 - 0.6449).abs() < 0.01);
    }
}
