//! Variance Stabilizing Transformation (VST)
//!
//! Transforms count data to approximately homoskedastic values,
//! where variance is independent of mean. Useful for visualization,
//! clustering, and machine learning.
//!
//! Implementation follows DESeq2's getVarianceStabilizedData() function.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};

/// VST computation method
/// R equivalent: fitType parameter in varianceStabilizingTransformation()
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VstMethod {
    /// Parametric dispersion fit - uses closed-form expression
    /// vst(q) = log2((1 + extraPois + 2*a*q + 2*sqrt(a*q*(1 + extraPois + a*q))) / (4*a))
    Parametric, // R: fitType="parametric"
    /// Local dispersion fit - uses numerical integration
    Local, // R: fitType="local"
    /// Mean dispersion - uses fixed dispersion formula
    /// vst(q) = (2*asinh(sqrt(a*q)) - log(a) - log(4)) / log(2)
    Mean, // R: fitType="mean"
}

impl Default for VstMethod {
    fn default() -> Self {
        VstMethod::Parametric
    }
}

/// Result of VST transformation
/// R equivalent: DESeqTransform S4 class in AllClasses.R
#[derive(Debug)]
pub struct VstResult {
    /// Transformed data matrix (genes x samples)
    pub data: Array2<f64>,
    /// Gene IDs
    pub gene_ids: Vec<String>,
    /// Sample IDs
    pub sample_ids: Vec<String>,
    /// Method used for transformation
    pub method: VstMethod,
    /// Dispersion intercept (asymptotic dispersion)
    pub disp_intercept: f64,
    /// Extra Poisson variance term
    pub extra_pois: f64,
}

/// Apply Variance Stabilizing Transformation
/// R equivalent: varianceStabilizingTransformation() / vst() in vst.R
///
/// This function transforms count data to the log2 scale in a way that
/// normalizes with respect to library size and yields approximately
/// homoskedastic data (constant variance across the range of mean values).
///
/// # Arguments
/// * `dds` - DESeqDataSet with size factors estimated
/// * `method` - VST method to use (Parametric, Local, or Mean)
/// * `blind` - If true, ignore design and use ~1 for dispersion estimation
///
/// # Returns
/// * `VstResult` containing transformed data
///
/// # Example
/// ```ignore
/// let vst_data = vst(&dds, VstMethod::Parametric, true)?;
/// ```
pub fn vst(dds: &DESeqDataSet, method: VstMethod, blind: bool) -> Result<VstResult> {
    // Check prerequisites
    if !dds.has_size_factors() {
        return Err(DeseqError::InvalidInput {
            reason: "Size factors must be estimated before VST".to_string(),
        });
    }

    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();
    let counts = dds.counts().counts();
    let size_factors = dds.size_factors().unwrap();

    // Calculate normalized counts
    let mut norm_counts = Array2::zeros((n_genes, n_samples));
    for i in 0..n_genes {
        for j in 0..n_samples {
            norm_counts[[i, j]] = counts[[i, j]] / size_factors[j];
        }
    }

    // R's vst() uses nsub=1000: subsample genes for dispersion estimation when n_genes > 1000
    let nsub: usize = 1000;
    let sub_norm_counts = if n_genes > nsub {
        // Subsample genes for faster dispersion estimation (R default: nsub=1000)
        // Use deterministic selection: pick genes with highest row variance for robustness
        let mut gene_vars: Vec<(usize, f64)> = (0..n_genes)
            .map(|i| {
                let mean = (0..n_samples).map(|j| norm_counts[[i, j]]).sum::<f64>() / n_samples as f64;
                let var = (0..n_samples)
                    .map(|j| (norm_counts[[i, j]] - mean).powi(2))
                    .sum::<f64>() / (n_samples as f64 - 1.0).max(1.0);
                (i, var)
            })
            .collect();
        gene_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected: Vec<usize> = gene_vars.iter().take(nsub).map(|&(i, _)| i).collect();
        let mut sub = Array2::zeros((nsub, n_samples));
        for (new_i, &orig_i) in selected.iter().enumerate() {
            for j in 0..n_samples {
                sub[[new_i, j]] = norm_counts[[orig_i, j]];
            }
        }
        Some(sub)
    } else {
        None
    };
    let estimation_counts = sub_norm_counts.as_ref().unwrap_or(&norm_counts);

    // Get dispersion function coefficients
    // R behavior:
    //   blind=TRUE  -> re-estimate dispersions using intercept-only design (ignoring existing)
    //   blind=FALSE -> use existing dispersions from the fitted model
    let (disp_intercept, extra_pois) = if blind {
        // blind=TRUE: estimate dispersions from data directly, ignoring any model-based estimates.
        // When the caller has already re-estimated dispersions with intercept-only design
        // (as main.rs does), use those. Otherwise fall back to data-driven estimation.
        if let Some((a0, a1)) = dds.dispersion_function() {
            (a0, a1)
        } else if let Some(_trend_disp) = dds.trended_dispersions() {
            estimate_dispersion_params(dds, estimation_counts)
        } else {
            // No dispersions at all - estimate mean dispersion from subsampled data
            let mean_disp = estimate_mean_dispersion(estimation_counts);
            (mean_disp, 0.01)
        }
    } else {
        // blind=FALSE: use existing dispersions from the fitted model.
        // The caller should have already run the full DESeq pipeline.
        if !dds.has_dispersions() {
            return Err(DeseqError::InvalidInput {
                reason: "blind=false requires dispersions to be estimated first (run DESeq)".to_string(),
            });
        }
        if let Some((a0, a1)) = dds.dispersion_function() {
            (a0, a1)
        } else if let Some(_trend_disp) = dds.trended_dispersions() {
            estimate_dispersion_params(dds, estimation_counts)
        } else {
            let mean_disp = estimate_mean_dispersion(estimation_counts);
            (mean_disp, 0.01)
        }
    };

    // Apply transformation based on method
    let transformed = match method {
        VstMethod::Parametric => {
            vst_parametric(&norm_counts, disp_intercept, extra_pois)
        }
        VstMethod::Mean => {
            vst_mean(&norm_counts, disp_intercept)
        }
        VstMethod::Local => {
            // Local fit uses numerical integration with the trended dispersions
            if let Some(trend_disp) = dds.trended_dispersions() {
                vst_local(&norm_counts, trend_disp.as_slice().unwrap(), size_factors)
            } else {
                // No trended dispersions available, fall back to parametric
                log::warn!("VST local: no trended dispersions available, falling back to parametric");
                vst_parametric(&norm_counts, disp_intercept, extra_pois)
            }
        }
    };

    Ok(VstResult {
        data: transformed,
        gene_ids: dds.counts().gene_ids().to_vec(),
        sample_ids: dds.counts().sample_ids().to_vec(),
        method,
        disp_intercept,
        extra_pois,
    })
}

/// Parametric VST transformation
/// DESeq2 formula: log2((1 + e + 2*a*q + 2*sqrt(a*q*(1 + e + a*q))) / (4*a))
/// where a = asymptotic dispersion, e = extra Poisson term, q = normalized count
fn vst_parametric(norm_counts: &Array2<f64>, asympt_disp: f64, extra_pois: f64) -> Array2<f64> {
    let n_genes = norm_counts.nrows();
    let n_samples = norm_counts.ncols();

    let mut result = Array2::zeros((n_genes, n_samples));

    // Parallel transformation
    let flat_result: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .flat_map(|i| {
            (0..n_samples)
                .map(|j| {
                    let q = norm_counts[[i, j]].max(0.0);
                    vst_parametric_single(q, asympt_disp, extra_pois)
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    for i in 0..n_genes {
        for j in 0..n_samples {
            result[[i, j]] = flat_result[i * n_samples + j];
        }
    }

    result
}

/// Single value parametric VST
/// R DESeq2 applies this formula to ALL values including zero (no early return).
fn vst_parametric_single(q: f64, asympt_disp: f64, extra_pois: f64) -> f64 {
    let a = asympt_disp;
    let e = extra_pois;
    let q = q.max(0.0);

    // DESeq2 formula - applies to ALL values including zero
    let numerator = 1.0 + e + 2.0 * a * q + 2.0 * (a * q * (1.0 + e + a * q)).sqrt();
    let denominator = 4.0 * a;

    if numerator > 0.0 && denominator > 0.0 {
        (numerator / denominator).ln() / 2.0_f64.ln()
    } else {
        0.0
    }
}

/// Mean-dispersion VST transformation
/// DESeq2 formula for negative binomial: (2*asinh(sqrt(a*q)) - log(a) - log(4)) / log(2)
fn vst_mean(norm_counts: &Array2<f64>, alpha: f64) -> Array2<f64> {
    let n_genes = norm_counts.nrows();
    let n_samples = norm_counts.ncols();

    let mut result = Array2::zeros((n_genes, n_samples));

    let flat_result: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .flat_map(|i| {
            (0..n_samples)
                .map(|j| {
                    let q = norm_counts[[i, j]].max(0.0);
                    vst_mean_single(q, alpha)
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    for i in 0..n_genes {
        for j in 0..n_samples {
            result[[i, j]] = flat_result[i * n_samples + j];
        }
    }

    result
}

/// Single value mean-dispersion VST
/// R DESeq2 applies this formula to ALL values including zero (no early return for q=0).
fn vst_mean_single(q: f64, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let q = q.max(0.0);

    // DESeq2 formula: (2*asinh(sqrt(alpha*q)) - log(alpha) - log(4)) / log(2)
    let asinh_term = 2.0 * (alpha * q).sqrt().asinh();
    (asinh_term - alpha.ln() - 4.0_f64.ln()) / 2.0_f64.ln()
}

/// Local-fit VST transformation using numerical integration
/// R DESeq2: when fitType="local", uses the trended dispersion function
/// to compute the variance stabilizing transformation via numerical integration.
/// For each normalized count q, integrates 1/sqrt(variance(q)) from 0 to q
/// where variance(q) = q + alpha(q) * q^2, and alpha(q) is the trended dispersion
/// looked up by the gene's mean expression.
fn vst_local(
    norm_counts: &Array2<f64>,
    trended_dispersions: &[f64],
    _size_factors: &Array1<f64>,
) -> Array2<f64> {
    let n_genes = norm_counts.nrows();
    let n_samples = norm_counts.ncols();

    // For each gene, apply VST via numerical integration using its trended dispersion
    let mut result = Array2::zeros((n_genes, n_samples));

    let flat_result: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .flat_map(|i| {
            let alpha = trended_dispersions[i].max(1e-8);
            (0..n_samples)
                .map(|j| {
                    let q = norm_counts[[i, j]].max(0.0);
                    // Numerical integration of 1/sqrt(variance(mu)) dmu from 0 to q
                    // variance(mu) = mu + alpha * mu^2 (negative binomial)
                    vst_numerical_integrate(q, alpha)
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    for i in 0..n_genes {
        for j in 0..n_samples {
            result[[i, j]] = flat_result[i * n_samples + j];
        }
    }

    result
}

/// Numerically integrate the VST transform: integral of 1/sqrt(mu + alpha*mu^2) dmu
/// from 0 to q, then convert to log2 scale.
/// Uses Simpson's rule for numerical integration.
fn vst_numerical_integrate(q: f64, alpha: f64) -> f64 {
    if q <= 0.0 {
        // For zero counts, use the formula value at a small epsilon
        let eps = 0.5;
        let val = 1.0 / (eps + alpha * eps * eps).sqrt();
        return (val * eps) / 2.0_f64.ln();
    }

    // Number of integration steps (even number for Simpson's rule)
    let n_steps: usize = 100;
    let h = q / n_steps as f64;

    // Simpson's rule: integral ~= h/3 * [f(0) + 4*f(h) + 2*f(2h) + 4*f(3h) + ... + f(q)]
    let integrand = |mu: f64| -> f64 {
        let mu = mu.max(1e-10);
        let var = mu + alpha * mu * mu;
        1.0 / var.sqrt()
    };

    let mut sum = integrand(1e-10) + integrand(q); // avoid exact 0 at lower bound
    for k in 1..n_steps {
        let mu_k = k as f64 * h;
        if k % 2 == 0 {
            sum += 2.0 * integrand(mu_k);
        } else {
            sum += 4.0 * integrand(mu_k);
        }
    }

    let integral = sum * h / 3.0;
    // Convert to log2 scale
    integral / 2.0_f64.ln()
}

/// Estimate dispersion parameters from data
/// Returns (asymptotic dispersion, extra Poisson variance)
fn estimate_dispersion_params(dds: &DESeqDataSet, norm_counts: &Array2<f64>) -> (f64, f64) {
    // If we have dispersion function parameters, use those
    // Otherwise estimate from trended dispersions

    if let Some(trend_disp) = dds.trended_dispersions() {
        // Calculate mean expression for each gene
        let n_genes = norm_counts.nrows();
        let n_samples = norm_counts.ncols();

        let base_means: Vec<f64> = (0..n_genes)
            .map(|i| {
                (0..n_samples).map(|j| norm_counts[[i, j]]).sum::<f64>() / n_samples as f64
            })
            .collect();

        // DESeq2: asymptotic dispersion is the dispersion at high expression
        // Select genes with baseMean > 75th percentile
        let mut sorted_means: Vec<f64> = base_means.iter().filter(|&&m| m > 0.0).copied().collect();
        sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q75_idx = (sorted_means.len() as f64 * 0.75) as usize;
        let q75_mean = if q75_idx < sorted_means.len() {
            sorted_means[q75_idx]
        } else {
            sorted_means.last().copied().unwrap_or(100.0)
        };

        // Get dispersions for high-expression genes
        let high_expr_disps: Vec<f64> = base_means
            .iter()
            .zip(trend_disp.iter())
            .filter(|(&m, _)| m >= q75_mean)
            .map(|(_, &d)| d)
            .collect();

        let asympt_disp = if high_expr_disps.is_empty() {
            // Fallback: use median of all dispersions
            let mut all_disps = trend_disp.to_vec();
            all_disps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_disps[all_disps.len() / 2]
        } else {
            // Median of high-expression dispersions
            let mut sorted = high_expr_disps.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        // Extra Poisson term - estimate from low expression genes
        // For simplicity, use a small default
        let extra_pois = 0.01;

        (asympt_disp.max(0.001), extra_pois)
    } else {
        // No trended dispersions - estimate mean dispersion
        (estimate_mean_dispersion(norm_counts), 0.01)
    }
}

/// Estimate mean dispersion using method of moments
fn estimate_mean_dispersion(norm_counts: &Array2<f64>) -> f64 {
    let n_genes = norm_counts.nrows();
    let n_samples = norm_counts.ncols();

    // For each gene: estimate dispersion = (var - mean) / mean^2
    let mut dispersions = Vec::new();

    for i in 0..n_genes {
        let row: Vec<f64> = (0..n_samples).map(|j| norm_counts[[i, j]]).collect();
        let mean = row.iter().sum::<f64>() / n_samples as f64;

        if mean > 1.0 {
            // Only use genes with sufficient expression
            let var = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64;
            let disp = (var - mean) / (mean * mean);
            if disp > 0.0 && disp.is_finite() {
                dispersions.push(disp);
            }
        }
    }

    if dispersions.is_empty() {
        return 0.1; // Default dispersion
    }

    // Use median
    dispersions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dispersions[dispersions.len() / 2].max(0.001)
}

impl VstResult {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vst_parametric_single() {
        // Test basic transformation
        let q = 100.0;
        let a = 0.1;
        let e = 0.01;

        let result = vst_parametric_single(q, a, e);
        // Should be approximately log2(100) = 6.64 for large counts
        assert!(result > 5.0 && result < 8.0);
    }

    #[test]
    fn test_vst_mean_single() {
        let q = 100.0;
        let a = 0.1;

        let result = vst_mean_single(q, a);
        // Should be similar to log2 transform
        assert!(result > 5.0 && result < 8.0);
    }

    #[test]
    fn test_vst_zero_handling() {
        // R DESeq2 applies VST formula even to zero counts, producing a non-zero floor value
        let vst_zero = vst_parametric_single(0.0, 0.1, 0.01);
        assert!(vst_zero.is_finite(), "VST of zero should be finite");
        assert!(vst_zero != 0.0, "VST of zero should not be exactly 0.0 (R DESeq2 gives a floor value)");

        let vst_mean_zero = vst_mean_single(0.0, 0.1);
        assert!(vst_mean_zero.is_finite(), "VST mean of zero should be finite");
        // asinh(0) = 0, so formula gives (0 - log(alpha) - log(4)) / log(2)
        // which is a finite negative number
        assert!(vst_mean_zero != 0.0, "VST mean of zero should not be exactly 0.0");
    }
}
