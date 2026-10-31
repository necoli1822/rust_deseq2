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
use crate::dispersion::DispersionTrend;
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
/// R equivalent: varianceStabilizingTransformation() / getVarianceStabilizedData() in vst.R
///
/// This function transforms count data to the log2 scale in a way that
/// normalizes with respect to library size and yields approximately
/// homoskedastic data (constant variance across the range of mean values).
///
/// The transformation is determined by the fitted dispersion trend stored in `dds`
/// (R: the "fitType" attribute of dispersionFunction(object)), so dispersions must
/// be estimated first, with an intercept-only design for a blind transformation.
/// As in R, a parametric fit that fell back to a local fit gives the local VST.
///
/// This is the full-data VST; R's faster vst() estimates the trend on a subset of
/// `nsub` genes and is not implemented here.
///
/// # Arguments
/// * `dds` - DESeqDataSet with size factors and dispersions estimated
/// * `method` - the fitType the dispersions were estimated with (reported back; a
///   mismatch with the stored trend is logged and the stored trend is used)
/// * `_blind` - kept for API compatibility; blindness is decided by the design the
///   dispersions were estimated with
///
/// # Returns
/// * `VstResult` containing transformed data
///
/// # Example
/// ```ignore
/// let vst_data = vst(&dds, VstMethod::Parametric, true)?;
/// ```
pub fn vst(dds: &DESeqDataSet, method: VstMethod, _blind: bool) -> Result<VstResult> {
    // Check prerequisites
    if !dds.has_size_factors() {
        return Err(DeseqError::InvalidInput {
            reason: "Size factors must be estimated before VST".to_string(),
        });
    }
    let trend = dds.dispersion_trend().ok_or_else(|| DeseqError::InvalidInput {
        reason: "Dispersions must be estimated before VST (R: call estimateDispersions)".to_string(),
    })?;

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

    let (used, transformed, disp_intercept, extra_pois) = match trend {
        DispersionTrend::Parametric { asympt_disp, extra_pois } => (
            VstMethod::Parametric,
            vst_parametric(&norm_counts, *asympt_disp, *extra_pois),
            *asympt_disp,
            *extra_pois,
        ),
        DispersionTrend::Mean(alpha) => (VstMethod::Mean, vst_mean(&norm_counts, *alpha), *alpha, 0.0),
        DispersionTrend::Local(_) => (
            VstMethod::Local,
            vst_local(&norm_counts, trend, size_factors),
            f64::NAN,
            f64::NAN,
        ),
    };
    if used != method {
        log::info!("VST follows the fitted dispersion trend ({:?}) rather than the requested {:?}", used, method);
    }

    Ok(VstResult {
        data: transformed,
        gene_ids: dds.counts().gene_ids().to_vec(),
        sample_ids: dds.counts().sample_ids().to_vec(),
        method: used,
        disp_intercept,
        extra_pois,
    })
}

/// Variance stabilizing transformation with the dispersion trend fitted on a
/// subset of genes
/// R equivalent: vst(object, blind, nsub, fitType) in vst.R
///
/// R keeps the genes with mean normalized count > 5, orders them by that mean and
/// takes the `nsub` at positions round(seq(1, n, length.out = nsub)). Gene-wise
/// dispersions and the trend (no MAP step) are estimated on that subset with the
/// full data's size factors and the design already set on `dds` (an intercept-only
/// design for a blind transformation), and the trend is then applied to all genes.
pub fn vst_subsampled(dds: &DESeqDataSet, method: VstMethod, nsub: usize) -> Result<VstResult> {
    use crate::dispersion::{estimate_gene_dispersions, fit_dispersion_trend, DispersionParams, TrendFitMethod};

    if nsub < 2 {
        return Err(DeseqError::InvalidInput { reason: "nsub must be at least 2".to_string() });
    }
    if dds.n_genes() < nsub {
        return Err(DeseqError::InvalidInput {
            reason: "less than 'nsub' rows, it is recommended to use the full VST directly".to_string(),
        });
    }
    let size_factors = dds.size_factors().ok_or_else(|| DeseqError::InvalidInput {
        reason: "Size factors must be estimated before VST".to_string(),
    })?;
    let counts = dds.counts().counts();
    let base_means: Vec<f64> = (0..dds.n_genes())
        .map(|i| (0..dds.n_samples()).map(|j| counts[[i, j]] / size_factors[j]).sum::<f64>() / dds.n_samples() as f64)
        .collect();

    // object.sub <- object[baseMean > 5,]; o <- order(baseMean)  (order() is stable)
    let mut kept: Vec<usize> = (0..dds.n_genes()).filter(|&i| base_means[i] > 5.0).collect();
    if kept.len() < nsub {
        return Err(DeseqError::InvalidInput {
            reason: "less than 'nsub' rows with mean normalized count > 5, it is recommended to use the full VST directly"
                .to_string(),
        });
    }
    kept.sort_by(|&a, &b| base_means[a].partial_cmp(&base_means[b]).unwrap());

    // idx <- o[round(seq(from = 1, to = length(o), length.out = nsub))]  (1-based, round half to even)
    let n = kept.len();
    let by = (n as f64 - 1.0) / (nsub as f64 - 1.0);
    let selected: Vec<usize> = (0..nsub)
        .map(|k| {
            let pos = if k == nsub - 1 { n as f64 } else { 1.0 + k as f64 * by };
            // R's round() rounds halves to even (f64::round_ties_even needs Rust 1.77)
            let floor = pos.floor();
            let frac = pos - floor;
            let rounded = if frac > 0.5 || (frac == 0.5 && floor % 2.0 != 0.0) { floor + 1.0 } else { floor };
            kept[rounded as usize - 1]
        })
        .collect();

    let sub_counts = dds.counts().subset_genes(&selected)?;
    let mut sub = DESeqDataSet::new(sub_counts, dds.sample_metadata().clone(), dds.design_variable())?;
    sub.set_size_factors(size_factors.clone())?;
    if let Some(design) = dds.design_matrix() {
        sub.set_design_matrix(design.clone())?;
    }
    let trend_method = match method {
        VstMethod::Parametric => TrendFitMethod::Parametric,
        VstMethod::Mean => TrendFitMethod::Mean,
        VstMethod::Local => TrendFitMethod::Local,
    };
    estimate_gene_dispersions(&mut sub, &DispersionParams::default())?;
    fit_dispersion_trend(&mut sub, trend_method)?;

    // dispersionFunction(object) <- dispersionFunction(object.sub)
    let trend = sub.dispersion_trend().cloned().ok_or_else(|| DeseqError::TrendFittingFailed {
        reason: "no dispersion trend fitted on the gene subset".to_string(),
    })?;
    let mut full = dds.clone();
    if let DispersionTrend::Parametric { asympt_disp, extra_pois } = trend {
        full.set_dispersion_function(asympt_disp, extra_pois);
    }
    full.set_dispersion_trend(trend);
    vst(&full, method, false)
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

/// Local-fit VST transformation by numerical integration of the dispersion trend
/// R equivalent: getVarianceStabilizedData() for fitType = "local":
///   xg <- sinh(seq(asinh(0), asinh(max(ncounts)), length.out = 1000))[-1]
///   baseVarsAtGrid <- dispersionFunction(object)(xg) * xg^2 + mean(1/sf) * xg
///   integrand <- 1 / sqrt(baseVarsAtGrid)
///   splf <- splinefun(asinh(midpoints of xg), cumsum(trapezoids of integrand))
///   h1, h2 <- quantile(rowMeans(ncounts), c(.95, .999))
///   eta <- (log2(h2) - log2(h1)) / (splf(asinh(h2)) - splf(asinh(h1)))
///   xi  <- log2(h1) - eta * splf(asinh(h1))
///   result = eta * splf(asinh(ncounts)) + xi
fn vst_local(norm_counts: &Array2<f64>, trend: &DispersionTrend, size_factors: &Array1<f64>) -> Array2<f64> {
    let max_count = norm_counts.iter().cloned().fold(0.0_f64, f64::max);
    let ngrid = 1000;
    let hi = max_count.asinh();
    let xg: Vec<f64> = (1..ngrid)
        .map(|k| (hi * k as f64 / (ngrid - 1) as f64).sinh())
        .collect();
    let xim = size_factors.iter().map(|s| 1.0 / s).sum::<f64>() / size_factors.len() as f64;
    let integrand: Vec<f64> = xg.iter().map(|&x| 1.0 / (trend.eval(x) * x * x + xim * x).sqrt()).collect();

    let knots: Vec<f64> = (0..xg.len() - 1).map(|k| ((xg[k + 1] + xg[k]) / 2.0).asinh()).collect();
    let mut acc = 0.0;
    let values: Vec<f64> = (0..xg.len() - 1)
        .map(|k| {
            acc += (xg[k + 1] - xg[k]) * (integrand[k + 1] + integrand[k]) / 2.0;
            acc
        })
        .collect();
    let spline = FmmSpline::new(&knots, &values);

    let row_means: Vec<f64> = norm_counts
        .rows()
        .into_iter()
        .map(|r| r.sum() / r.len() as f64)
        .collect();
    let h1 = quantile_type7(&row_means, 0.95);
    let h2 = quantile_type7(&row_means, 0.999);
    let eta = (h2.log2() - h1.log2()) / (spline.eval(h2.asinh()) - spline.eval(h1.asinh()));
    let xi = h1.log2() - eta * spline.eval(h1.asinh());

    norm_counts.mapv(|q| eta * spline.eval(q.asinh()) + xi)
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

/// Cubic interpolating spline with Forsythe-Malcolm-Moler end conditions
/// R equivalent: splinefun(x, y) with the default method = "fmm"
/// (fmm_spline / spline_eval in R's src/library/stats/src/splines.c)
struct FmmSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl FmmSpline {
    fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        if n < 3 {
            if n == 2 {
                b[0] = (y[1] - y[0]) / (x[1] - x[0]);
                b[1] = b[0];
            }
            return Self { x: x.to_vec(), y: y.to_vec(), b, c, d };
        }
        let nm1 = n - 1;
        // Tridiagonal system: b = diagonal, d = off-diagonal, c = right-hand side
        d[0] = x[1] - x[0];
        c[1] = (y[1] - y[0]) / d[0];
        for i in 1..nm1 {
            d[i] = x[i + 1] - x[i];
            b[i] = 2.0 * (d[i - 1] + d[i]);
            c[i + 1] = (y[i + 1] - y[i]) / d[i];
            c[i] = c[i + 1] - c[i];
        }
        // End conditions: third derivatives from divided differences
        b[0] = -d[0];
        b[nm1] = -d[n - 2];
        c[0] = 0.0;
        c[nm1] = 0.0;
        if n > 3 {
            c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0]);
            c[nm1] = c[n - 2] / (x[nm1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4]);
            c[0] = c[0] * d[0] * d[0] / (x[3] - x[0]);
            c[nm1] = -c[nm1] * d[n - 2] * d[n - 2] / (x[nm1] - x[n - 4]);
        }
        // Gaussian elimination
        for i in 1..n {
            let t = d[i - 1] / b[i - 1];
            b[i] -= t * d[i - 1];
            c[i] -= t * c[i - 1];
        }
        // Backward substitution
        c[nm1] /= b[nm1];
        for i in (0..nm1).rev() {
            c[i] = (c[i] - d[i] * c[i + 1]) / b[i];
        }
        // Polynomial coefficients
        b[nm1] = (y[nm1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[nm1]);
        for i in 0..nm1 {
            b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i]);
            d[i] = (c[i + 1] - c[i]) / d[i];
            c[i] *= 3.0;
        }
        c[nm1] *= 3.0;
        d[nm1] = d[n - 2];
        Self { x: x.to_vec(), y: y.to_vec(), b, c, d }
    }

    fn eval(&self, u: f64) -> f64 {
        let n = self.x.len();
        // interval i with x[i] <= u < x[i+1]; the end cubics extrapolate
        let i = if u < self.x[0] {
            0
        } else {
            let (mut lo, mut hi) = (0usize, n);
            while hi > lo + 1 {
                let k = (lo + hi) / 2;
                if u < self.x[k] {
                    hi = k;
                } else {
                    lo = k;
                }
            }
            lo
        };
        let dx = u - self.x[i];
        self.y[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))
    }
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
