//! Dispersion trend fitting

use ndarray::Array1;

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};

/// Method for fitting the dispersion-mean trend
/// R equivalent: fitType parameter in estimateDispersionsFit() in core.R
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendFitMethod {
    /// Parametric fit: dispersion = a0 + a1/mean
    Parametric, // R: fitType="parametric"
    /// Local regression (loess-like)
    Local, // R: fitType="local"
    /// Use mean of dispersions (no trend)
    Mean, // R: fitType="mean"
}

/// Fit a trend to the gene-wise dispersions as a function of mean expression
/// R equivalent: estimateDispersionsFit() in core.R
pub fn fit_dispersion_trend(dds: &mut DESeqDataSet, method: TrendFitMethod) -> Result<()> {
    let gene_dispersions = dds.gene_dispersions().ok_or_else(|| DeseqError::TrendFittingFailed {
        reason: "Gene-wise dispersions must be estimated first".to_string(),
    })?;

    let normalized = dds.normalized_counts().ok_or_else(|| DeseqError::TrendFittingFailed {
        reason: "Normalized counts required for trend fitting".to_string(),
    })?;

    // Calculate mean expression for each gene
    let n_samples = dds.n_samples() as f64;
    let means: Vec<f64> = normalized
        .rows()
        .into_iter()
        .map(|row| row.sum() / n_samples)
        .collect();

    let trended = match method {
        TrendFitMethod::Parametric => {
            // DESeq2 behavior: try parametric first, fall back to local if:
            // 1. Gamma GLM fails to converge
            // 2. Coefficient a1 < 0 (dispersion would increase with mean - invalid)
            match fit_parametric_trend(&means, gene_dispersions.as_slice().unwrap()) {
                Ok((parametric_fit, coefs)) => {
                    // Store dispersion function coefficients for VST
                    dds.set_dispersion_function(coefs.0, coefs.1);
                    parametric_fit
                }
                Err(e) => {
                    log::info!("Parametric fit failed ({}), using local regression", e);
                    fit_local_trend(&means, gene_dispersions.as_slice().unwrap())?
                }
            }
        }
        TrendFitMethod::Local => fit_local_trend(&means, gene_dispersions.as_slice().unwrap())?,
        TrendFitMethod::Mean => {
            // R equivalent: estimateDispersionsFit() fitType="mean" in core.R
            // useForMean <- mcols(objectNZ)$dispGeneEst > 10 * minDisp
            // meanDisp <- mean(mcols(objectNZ)$dispGeneEst[useForMean], na.rm=TRUE, trim=0.001)
            let min_disp = 1e-8;
            let mut valid_disps: Vec<f64> = gene_dispersions
                .iter()
                .filter(|&&d| d > 10.0 * min_disp && d.is_finite())
                .copied()
                .collect();

            let mean_disp = if valid_disps.is_empty() {
                gene_dispersions.mean().unwrap_or(0.1)
            } else {
                // Trimmed mean with trim=0.001 (matching R's mean(..., trim=0.001))
                valid_disps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = valid_disps.len();
                let trim_count = ((n as f64) * 0.001).floor() as usize;
                let trimmed = &valid_disps[trim_count..n - trim_count.max(0)];
                if trimmed.is_empty() {
                    valid_disps.iter().sum::<f64>() / valid_disps.len() as f64
                } else {
                    trimmed.iter().sum::<f64>() / trimmed.len() as f64
                }
            };
            Array1::from_elem(gene_dispersions.len(), mean_disp)
        }
    };

    dds.set_trended_dispersions(trended)?;
    Ok(())
}

/// Fit parametric trend: dispersion = a0 + a1/mean
/// Matches DESeq2's parametricDispersionFit function exactly
/// Returns (trended_dispersions, (asymptDisp, extraPois))
fn fit_parametric_trend(means: &[f64], dispersions: &[f64]) -> Result<(Array1<f64>, (f64, f64))> {
    // Filter out invalid values (match DESeq2's filter: dispersion > 100 * minDisp = 1e-6)
    let valid: Vec<(f64, f64)> = means
        .iter()
        .zip(dispersions.iter())
        .filter(|(&m, &d)| m > 0.0 && d > 1e-6 && d.is_finite())
        .map(|(&m, &d)| (m, d))
        .collect();

    if valid.len() < 3 {
        return Err(DeseqError::TrendFittingFailed {
            reason: "Not enough valid data points for trend fitting".to_string(),
        });
    }

    // DESeq2's iterative Gamma GLM with residual filtering
    let (a0, a1) = fit_parametric_deseq2_style(&valid)?;

    // Calculate trended dispersions
    // R does not floor parametric predictions - they are used as-is
    let trended: Vec<f64> = means
        .iter()
        .map(|&m| {
            if m > 0.0 {
                a0 + a1 / m
            } else {
                a0
            }
        })
        .collect();

    Ok((Array1::from_vec(trended), (a0, a1)))
}

/// DESeq2-style parametric dispersion fit with iterative residual filtering
/// Matches R DESeq2's parametricDispersionFit function in core.R:2186-2209
fn fit_parametric_deseq2_style(data: &[(f64, f64)]) -> Result<(f64, f64)> {
    let mut coefs = (0.1_f64, 1.0_f64);  // (asymptDisp, extraPois)
    let max_iter = 11;  // R uses niter=11 in parametricDispersionFit
    let tol = 1e-6;

    for iter in 0..max_iter {
        let old_coefs = coefs;

        // Filter data points by residuals (DESeq2: 1e-4 < residual < 15)
        let mut n_fail_low = 0;
        let mut n_fail_high = 0;
        let good_data: Vec<(f64, f64)> = data
            .iter()
            .filter(|&&(mean, disp)| {
                let fitted = coefs.0 + coefs.1 / mean;
                if fitted <= 0.0 {
                    return false;
                }
                let residual = disp / fitted;
                if residual <= 1e-4 {
                    n_fail_low += 1;
                    return false;
                }
                if residual >= 15.0 {
                    n_fail_high += 1;
                    return false;
                }
                true
            })
            .copied()
            .collect();

        log::debug!(
            "Parametric filter: {} total, {} fail low, {} fail high, {} good",
            data.len(), n_fail_low, n_fail_high, good_data.len()
        );

        if good_data.len() < 3 {
            return Err(DeseqError::TrendFittingFailed {
                reason: "Not enough good residuals for parametric fit".to_string(),
            });
        }

        log::debug!(
            "Parametric fit iter {}: {} good genes, starting from a0={:.6}, a1={:.6}",
            iter + 1, good_data.len(), coefs.0, coefs.1
        );

        // Fit Gamma GLM with identity link: disp ~ I(1/mean)
        // Pass current coefficients as starting values (like R's glm start=coefs)
        let (new_coefs, glm_converged) = fit_gamma_glm_identity_with_start(&good_data, coefs)?;

        log::debug!(
            "  After GLM: a0={:.6}, a1={:.6}, converged={}",
            new_coefs.0, new_coefs.1, glm_converged
        );

        coefs = new_coefs;

        // DESeq2 check: ALL coefficients must be > 0
        if coefs.0 <= 0.0 || coefs.1 <= 0.0 {
            return Err(DeseqError::TrendFittingFailed {
                reason: format!(
                    "parametric dispersion fit failed: coefficients not positive (a0={:.4}, a1={:.4})",
                    coefs.0, coefs.1
                ),
            });
        }

        // Check convergence: sum(log(coefs/oldcoefs)^2) < 1e-6
        // AND glm must have converged (R checks glm$converged)
        let log_change = (coefs.0 / old_coefs.0).ln().powi(2)
                       + (coefs.1 / old_coefs.1).ln().powi(2);

        if log_change < tol && glm_converged {
            log::debug!(
                "Parametric fit converged at iter {}: a0={:.6}, a1={:.4}",
                iter + 1, coefs.0, coefs.1
            );
            return Ok(coefs);
        }
    }

    Err(DeseqError::TrendFittingFailed {
        reason: "dispersion fit did not converge".to_string(),
    })
}

/// Fit Gamma GLM with identity link: y ~ 1 + 1/x
/// Returns ((intercept, slope), converged) = ((asymptDisp, extraPois), bool)
fn fit_gamma_glm_identity_with_start(data: &[(f64, f64)], start: (f64, f64)) -> Result<((f64, f64), bool)> {
    let mut a0 = start.0;
    let mut a1 = start.1;
    let irls_max_iter = 25;
    let irls_tol = 1e-8;
    let mut converged = false;

    // Initialize deviance with starting coefficients
    let mut dev_old = {
        let mut dev = 0.0;
        for &(mean, disp) in data {
            let x = 1.0 / mean;
            let mu = (a0 + a1 * x).max(1e-8);
            // Gamma deviance: 2 * sum(-log(y/mu) + (y - mu)/mu)
            dev += 2.0 * (-(disp / mu).ln() + (disp - mu) / mu);
        }
        dev
    };

    for _iter in 0..irls_max_iter {
        let mut sum_w = 0.0_f64;
        let mut sum_wx = 0.0_f64;
        let mut sum_wz = 0.0_f64;
        let mut sum_wxx = 0.0_f64;
        let mut sum_wxz = 0.0_f64;

        for &(mean, disp) in data {
            let x = 1.0 / mean;
            let mu = (a0 + a1 * x).max(1e-8);
            let w = 1.0 / (mu * mu);  // Gamma weight
            let z = disp;  // Working response for identity link

            sum_w += w;
            sum_wx += w * x;
            sum_wz += w * z;
            sum_wxx += w * x * x;
            sum_wxz += w * x * z;
        }

        let det = sum_w * sum_wxx - sum_wx * sum_wx;
        if det.abs() < 1e-10 {
            break;
        }

        a0 = (sum_wxx * sum_wz - sum_wx * sum_wxz) / det;
        a1 = (sum_w * sum_wxz - sum_wx * sum_wz) / det;

        // Compute deviance with updated coefficients
        let mut dev = 0.0;
        for &(mean, disp) in data {
            let x = 1.0 / mean;
            let mu = (a0 + a1 * x).max(1e-8);
            // Gamma deviance: 2 * sum(-log(y/mu) + (y - mu)/mu)
            dev += 2.0 * (-(disp / mu).ln() + (disp - mu) / mu);
        }

        // R's glm.fit convergence: abs(dev - devold)/(0.1 + abs(dev)) < epsilon
        let dev_change = (dev_old - dev).abs() / (0.1 + dev.abs());
        if dev_change < irls_tol {
            converged = true;
            break;
        }

        dev_old = dev;
    }

    Ok(((a0, a1), converged))
}


/// Locfit-style local regression with tree-based evaluation and Hermite interpolation
///
/// This implementation exactly matches R locfit's behavior:
/// - DESeq2 uses: locfit(logDisps ~ logMeans, weights=means)
/// - Default parameters: deg=2 (local quadratic), nn=0.7, cut=0.8
/// - Tree-based adaptive evaluation
/// - Hermite cubic interpolation between evaluation points
fn fit_local_trend(means: &[f64], dispersions: &[f64]) -> Result<Array1<f64>> {
    let n = means.len();
    let min_disp = 1e-8;

    if n < 5 {
        return Err(DeseqError::TrendFittingFailed {
            reason: "Not enough data points for local fitting".to_string(),
        });
    }

    // Filter valid values and work in log scale
    // R uses: useForFit <- dispGeneEst > 100 * minDisp (i.e. > 1e-6)
    let mut valid_data: Vec<(f64, f64, f64)> = means
        .iter()
        .zip(dispersions.iter())
        .filter(|(&m, &d)| m > 0.0 && d > min_disp * 100.0 && d.is_finite())
        .map(|(&m, &d)| (m.ln(), d.ln(), m))  // (log_mean, log_disp, weight=mean)
        .collect();

    if valid_data.len() < 5 {
        let mean_disp = dispersions.iter().filter(|d| d.is_finite()).sum::<f64>()
            / dispersions.iter().filter(|d| d.is_finite()).count() as f64;
        return Ok(Array1::from_elem(n, mean_disp));
    }

    // Sort by log_mean
    valid_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let xd: Vec<f64> = valid_data.iter().map(|(m, _, _)| *m).collect();
    let yd: Vec<f64> = valid_data.iter().map(|(_, d, _)| *d).collect();
    let obs_weights: Vec<f64> = valid_data.iter().map(|(_, _, w)| *w).collect();

    // Create locfit tree structure with deg=2 (local quadratic)
    let locfit = LocfitTree::build(&xd, &yd, &obs_weights, 0.7, 0.8);

    // Predict for all genes
    let mut trended = Vec::with_capacity(n);

    for i in 0..n {
        let mean = means[i];
        if mean <= 0.0 {
            trended.push(0.1);
            continue;
        }

        let x = mean.ln();
        let log_disp_pred = locfit.predict(x);
        trended.push(log_disp_pred.exp().max(1e-8));
    }

    Ok(Array1::from_vec(trended))
}

/// Evaluation point in the locfit tree
/// Stores fitted value and derivative for Hermite interpolation
#[derive(Debug, Clone)]
struct EvalPoint {
    x: f64,           // evaluation x location
    value: f64,       // fitted value (β₀)
    deriv: f64,       // derivative (β₁)
    h: f64,           // local bandwidth
}

/// Locfit tree structure for 1D local regression with interpolation
struct LocfitTree {
    eval_points: Vec<EvalPoint>,
    xd: Vec<f64>,
    yd: Vec<f64>,
    obs_weights: Vec<f64>,
    k: usize,
}

impl LocfitTree {
    /// Build locfit tree with adaptive evaluation points
    fn build(xd: &[f64], yd: &[f64], obs_weights: &[f64], nn: f64, cut: f64) -> Self {
        let n = xd.len();
        let k = ((n as f64 * nn + 1e-12) as usize).max(3).min(n);

        let xd = xd.to_vec();
        let yd = yd.to_vec();
        let obs_weights = obs_weights.to_vec();

        let mut tree = LocfitTree {
            eval_points: Vec::new(),
            xd,
            yd,
            obs_weights,
            k,
        };

        // Get data range
        let x_min = tree.xd[0];
        let x_max = tree.xd[tree.xd.len() - 1];

        // Fit at boundary points first (locfit atree_start)
        let ep0 = tree.fit_at_point(x_min);
        let ep1 = tree.fit_at_point(x_max);

        tree.eval_points.push(ep0);
        tree.eval_points.push(ep1);

        // Recursively split cells that are too large relative to bandwidth
        tree.split_cell(0, 1, cut);

        // Sort evaluation points by x for binary search
        tree.eval_points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

        tree
    }

    /// Fit local QUADRATIC regression (deg=2) at point x
    /// Returns (value, derivative, bandwidth)
    ///
    /// Uses locfit's basis: [1, dx, dx²/2]
    fn fit_at_point(&self, x: f64) -> EvalPoint {
        let n = self.xd.len();

        // === locfit nbhd1() algorithm: find k nearest neighbors in sorted 1D data ===
        let z = match self.xd.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx == 0 { 0 }
                else if idx >= n { n - 1 }
                else if (x - self.xd[idx - 1]).abs() <= (self.xd[idx] - x).abs() { idx - 1 }
                else { idx }
            }
        };

        let mut l = z;
        let mut r = z;

        if l == 0 { r = (self.k - 1).min(n - 1); }
        if r == n - 1 { l = n.saturating_sub(self.k); }

        while r - l < self.k - 1 {
            let can_go_left = l > 0;
            let can_go_right = r < n - 1;

            if can_go_left && can_go_right {
                if (x - self.xd[l - 1]).abs() < (self.xd[r + 1] - x).abs() {
                    l -= 1;
                } else {
                    r += 1;
                }
            } else if can_go_left {
                l -= 1;
            } else if can_go_right {
                r += 1;
            } else {
                break;
            }

            if l == 0 { r = (self.k - 1).min(n - 1); }
            if r == n - 1 { l = n.saturating_sub(self.k); }
        }

        let h_left = (x - self.xd[l]).abs();
        let h_right = (self.xd[r] - x).abs();
        let h = h_left.max(h_right).max(1e-10);

        // === Local QUADRATIC regression (deg=2): basis [1, dx, dx²/2] ===
        // Solves weighted least squares: y = β₀ + β₁*dx + β₂*(dx²/2)
        let mut sum_w = 0.0_f64;
        let mut sum_wx1 = 0.0_f64;
        let mut sum_wx2 = 0.0_f64;
        let mut sum_wy = 0.0_f64;
        let mut sum_wx1x1 = 0.0_f64;
        let mut sum_wx1x2 = 0.0_f64;
        let mut sum_wx2x2 = 0.0_f64;
        let mut sum_wx1y = 0.0_f64;
        let mut sum_wx2y = 0.0_f64;

        for j in 0..n {
            let dx = self.xd[j] - x;
            let u = dx.abs() / h;

            if u >= 1.0 {
                continue;
            }

            // Tricube kernel
            let u3 = u * u * u;
            let kernel_w = {
                let t = 1.0 - u3;
                t * t * t
            };

            let w = kernel_w * self.obs_weights[j];

            // Basis functions (locfit style)
            let f1 = dx;              // dx
            let f2 = dx * dx / 2.0;   // dx²/2

            sum_w += w;
            sum_wx1 += w * f1;
            sum_wx2 += w * f2;
            sum_wy += w * self.yd[j];
            sum_wx1x1 += w * f1 * f1;
            sum_wx1x2 += w * f1 * f2;
            sum_wx2x2 += w * f2 * f2;
            sum_wx1y += w * f1 * self.yd[j];
            sum_wx2y += w * f2 * self.yd[j];
        }

        // Solve 3x3 system for [β₀, β₁, β₂]
        let (value, deriv) = solve_for_value_and_deriv(
            sum_w, sum_wx1, sum_wx2,
            sum_wx1x1, sum_wx1x2, sum_wx2x2,
            sum_wy, sum_wx1y, sum_wx2y
        );

        EvalPoint { x, value, deriv, h }
    }

    /// Recursively split cells that are too large (locfit atree_grow)
    fn split_cell(&mut self, left_idx: usize, right_idx: usize, cut: f64) {
        let left = &self.eval_points[left_idx];
        let right = &self.eval_points[right_idx];

        let cell_width = right.x - left.x;
        let min_h = left.h.min(right.h);

        // Check if cell needs splitting: cell_width / min_h > cut
        if cell_width / min_h <= cut || cell_width < 1e-10 {
            return;
        }

        // Split at midpoint
        let mid_x = (left.x + right.x) / 2.0;
        let mid_ep = self.fit_at_point(mid_x);

        let mid_idx = self.eval_points.len();
        self.eval_points.push(mid_ep);

        // Recursively split sub-cells
        self.split_cell(left_idx, mid_idx, cut);
        self.split_cell(mid_idx, right_idx, cut);
    }

    /// Predict at point x using Hermite interpolation
    fn predict(&self, x: f64) -> f64 {
        let n_eval = self.eval_points.len();

        if n_eval == 0 {
            return 0.0;
        }

        if n_eval == 1 {
            let ep = &self.eval_points[0];
            return ep.value + ep.deriv * (x - ep.x);
        }

        // Find containing cell (binary search for left endpoint)
        let mut left_idx = 0;
        let mut right_idx = n_eval - 1;

        // Handle extrapolation
        if x <= self.eval_points[0].x {
            // Extrapolate from leftmost point using its local polynomial
            let ep = &self.eval_points[0];
            return ep.value + ep.deriv * (x - ep.x);
        }
        if x >= self.eval_points[n_eval - 1].x {
            // Extrapolate from rightmost point
            let ep = &self.eval_points[n_eval - 1];
            return ep.value + ep.deriv * (x - ep.x);
        }

        // Binary search for containing cell
        while right_idx - left_idx > 1 {
            let mid = (left_idx + right_idx) / 2;
            if x < self.eval_points[mid].x {
                right_idx = mid;
            } else {
                left_idx = mid;
            }
        }

        // Hermite cubic interpolation (locfit hermite2 + rectcell_interp)
        let left = &self.eval_points[left_idx];
        let right = &self.eval_points[right_idx];

        let d = right.x - left.x;  // cell width
        if d.abs() < 1e-15 {
            return left.value;
        }

        let t = (x - left.x) / d;  // normalized position [0, 1]

        // Hermite basis functions (from locfit ev_interp.c:hermite2)
        // phi[0] = weight for left value
        // phi[1] = weight for right value
        // phi[2] = weight for left derivative * d
        // phi[3] = weight for right derivative * d
        let phi1 = t * t * (3.0 - 2.0 * t);      // right value weight
        let phi0 = 1.0 - phi1;                    // left value weight
        let phi2 = t * (1.0 - t) * (1.0 - t);    // left deriv weight
        let phi3 = t * t * (t - 1.0);            // right deriv weight

        // Interpolate: f(x) = phi0*f_L + phi1*f_R + d*(phi2*d_L + phi3*d_R)
        phi0 * left.value + phi1 * right.value + d * (phi2 * left.deriv + phi3 * right.deriv)
    }
}

/// Solve 3x3 symmetric positive definite system for value (β₀) and derivative (β₁)
fn solve_for_value_and_deriv(
    m00: f64, m01: f64, m02: f64,
    m11: f64, m12: f64, m22: f64,
    b0: f64, b1: f64, b2: f64,
) -> (f64, f64) {
    // Check for near-singular matrix
    let det = m00 * (m11 * m22 - m12 * m12)
            - m01 * (m01 * m22 - m12 * m02)
            + m02 * (m01 * m12 - m11 * m02);

    if det.abs() < 1e-15 * m00.max(1.0) * m11.max(1.0) * m22.max(1.0) {
        // Matrix is singular, fall back to weighted mean
        if m00 > 1e-10 {
            return (b0 / m00, 0.0);
        } else {
            return (0.0, 0.0);
        }
    }

    // Cramer's rule for β₀ and β₁
    // β₀ = det(M with first column replaced by b) / det(M)
    let det_a0 = b0 * (m11 * m22 - m12 * m12)
               - m01 * (b1 * m22 - m12 * b2)
               + m02 * (b1 * m12 - m11 * b2);

    // β₁ = det(M with second column replaced by b) / det(M)
    let det_a1 = m00 * (b1 * m22 - m12 * b2)
               - b0 * (m01 * m22 - m12 * m02)
               + m02 * (m01 * b2 - b1 * m02);

    (det_a0 / det, det_a1 / det)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_glm_identity() {
        let data: Vec<(f64, f64)> = (1..50)
            .map(|i| {
                let x = i as f64 * 20.0 + 100.0;
                let y = 0.1 + 10.0 / x;
                (x, y)
            })
            .collect();

        let result = fit_gamma_glm_identity_with_start(&data, (0.1, 1.0));
        assert!(result.is_ok(), "Gamma GLM fit should succeed");

        let ((a0, a1), converged) = result.unwrap();
        assert!(a0 > 0.0, "a0 should be positive");
        assert!(a1 > 0.0, "a1 should be positive");
        assert!(converged, "GLM should converge on well-behaved data");
    }

    #[test]
    fn test_local_trend_fallback() {
        let means: Vec<f64> = (1..100).map(|i| i as f64 * 10.0).collect();
        let disps: Vec<f64> = means.iter().map(|&m| 0.1 + 5.0 / m).collect();

        let result = fit_local_trend(&means, &disps);
        assert!(result.is_ok(), "Local trend should succeed");

        let trended = result.unwrap();
        assert_eq!(trended.len(), means.len());
        assert!(trended.iter().all(|&v| v > 0.0), "All trended values should be positive");
    }

    #[test]
    fn test_hermite_interpolation() {
        // Test that Hermite interpolation preserves endpoint values
        let xd: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let yd: Vec<f64> = xd.iter().map(|x| x.sin()).collect();
        let weights: Vec<f64> = vec![1.0; 10];

        let tree = LocfitTree::build(&xd, &yd, &weights, 0.7, 0.8);

        // At evaluation points, prediction should be close to fitted value
        for ep in &tree.eval_points {
            let pred = tree.predict(ep.x);
            assert!((pred - ep.value).abs() < 0.01,
                "Prediction at eval point should match: {} vs {}", pred, ep.value);
        }
    }
}
