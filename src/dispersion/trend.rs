//! Dispersion trend fitting

use ndarray::Array1;

use crate::data::DESeqDataSet;
use crate::error::{DeseqError, Result};

use super::locfit::Locfit;

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

/// The fitted dispersion-mean trend, kept so trended dispersions can be evaluated
/// at new means after the fit (e.g. when refitting genes without outliers).
/// R equivalent: dispersionFunction(object)
#[derive(Debug, Clone)]
pub enum DispersionTrend {
    /// asymptDisp + extraPois / mean
    Parametric { asympt_disp: f64, extra_pois: f64 },
    /// Local regression of log dispersion on log mean
    Local(LocalTrend),
    /// A single dispersion for every gene
    Mean(f64),
}

impl DispersionTrend {
    /// Trended dispersion at a given mean of normalized counts
    pub fn eval(&self, mean: f64) -> f64 {
        match self {
            DispersionTrend::Parametric { asympt_disp, extra_pois } => {
                if mean > 0.0 {
                    asympt_disp + extra_pois / mean
                } else {
                    *asympt_disp
                }
            }
            DispersionTrend::Local(trend) => trend.eval(mean),
            DispersionTrend::Mean(disp) => *disp,
        }
    }
}

/// A fitted local dispersion trend (see [`DispersionTrend::Local`])
#[derive(Debug, Clone)]
pub struct LocalTrend(Locfit);

impl LocalTrend {
    /// R: dispFunction <- function(means) exp(predict(fit, data.frame(logMeans=log(means))))
    /// (no lower bound: far outside the data the prediction follows locfit's
    /// global quadratic and can be very small, as in R)
    fn eval(&self, mean: f64) -> f64 {
        if mean <= 0.0 {
            return 0.1;
        }
        self.0.predict(mean.ln()).exp()
    }
}

/// Fit a trend to the gene-wise dispersions as a function of mean expression
/// R equivalent: estimateDispersionsFit() in core.R
pub fn fit_dispersion_trend(dds: &mut DESeqDataSet, method: TrendFitMethod) -> Result<()> {
    let gene_dispersions = dds.gene_dispersions().ok_or_else(|| DeseqError::TrendFittingFailed {
        reason: "Gene-wise dispersions must be estimated first".to_string(),
    })?;

    // R: the trend is fitted against mcols(object)$baseMean, which with
    // observation weights is the mean of weights * normalized counts
    let normalized = dds.base_mean_counts().ok_or_else(|| DeseqError::TrendFittingFailed {
        reason: "Normalized counts required for trend fitting".to_string(),
    })?;

    // Calculate mean expression for each gene
    let n_samples = dds.n_samples() as f64;
    let means: Vec<f64> = normalized
        .rows()
        .into_iter()
        .map(|row| row.sum() / n_samples)
        .collect();

    let (trended, trend) = match method {
        TrendFitMethod::Parametric => {
            // DESeq2 behavior: try parametric first, fall back to local if:
            // 1. Gamma GLM fails to converge
            // 2. Coefficient a1 < 0 (dispersion would increase with mean - invalid)
            match fit_parametric_trend(&means, gene_dispersions.as_slice().unwrap()) {
                Ok((parametric_fit, coefs)) => {
                    // Store dispersion function coefficients for VST
                    dds.set_dispersion_function(coefs.0, coefs.1);
                    let trend = DispersionTrend::Parametric { asympt_disp: coefs.0, extra_pois: coefs.1 };
                    (parametric_fit, trend)
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
            (Array1::from_elem(gene_dispersions.len(), mean_disp), DispersionTrend::Mean(mean_disp))
        }
    };

    dds.set_trended_dispersions(trended)?;
    dds.set_dispersion_trend(trend);
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
        .filter(|&(&m, &d)| m > 0.0 && d > 1e-6 && d.is_finite())
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


/// Local regression of log dispersion on log mean
/// R equivalent: localDispersionFit() in core.R,
///   locfit(logDisps ~ logMeans, data = d[disps >= minDisp*10, ], weights = means)
/// with locfit's defaults (see [`super::locfit`]). The genes are kept in their
/// original order, as locfit's sums run in data order.
fn fit_local_trend(means: &[f64], dispersions: &[f64]) -> Result<(Array1<f64>, DispersionTrend)> {
    let n = means.len();
    let min_disp = 1e-8;

    if n < 5 {
        return Err(DeseqError::TrendFittingFailed {
            reason: "Not enough data points for local fitting".to_string(),
        });
    }

    // R uses: useForFit <- dispGeneEst > 100 * minDisp (i.e. > 1e-6), then
    // localDispersionFit keeps disps >= 10 * minDisp, which is implied
    let valid: Vec<(f64, f64)> = means
        .iter()
        .zip(dispersions.iter())
        .filter(|&(&m, &d)| m > 0.0 && d > min_disp * 100.0 && d.is_finite())
        .map(|(&m, &d)| (m, d))
        .collect();

    if valid.len() < 5 {
        let mean_disp = dispersions.iter().filter(|d| d.is_finite()).sum::<f64>()
            / dispersions.iter().filter(|d| d.is_finite()).count() as f64;
        return Ok((Array1::from_elem(n, mean_disp), DispersionTrend::Mean(mean_disp)));
    }

    let xd: Vec<f64> = valid.iter().map(|(m, _)| m.ln()).collect();
    let yd: Vec<f64> = valid.iter().map(|(_, d)| d.ln()).collect();
    let obs_weights: Vec<f64> = valid.iter().map(|(m, _)| *m).collect();

    let fit = Locfit::fit(&xd, &yd, &obs_weights, 0.7, 0.8)
        .map_err(|e| DeseqError::TrendFittingFailed { reason: e.to_string() })?;

    let trend = LocalTrend(fit);
    let trended: Vec<f64> = means.iter().map(|&mean| trend.eval(mean)).collect();

    Ok((Array1::from_vec(trended), DispersionTrend::Local(trend)))
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

        let (trended, _) = result.unwrap();
        assert_eq!(trended.len(), means.len());
        assert!(trended.iter().all(|&v| v > 0.0), "All trended values should be positive");
    }

    /// Unsorted (mean, dispersion) pairs; the smallest mean is 0.22313 and the
    /// largest 43477.6
    const LOCFIT_MEANS: [f64; 30] = [
        0.22313, 4.2207, 79.838, 1510.2, 28566.8, 1.82212, 34.4669, 651.971, 12332.6, 0.786628,
        14.8797, 281.463, 5324.11, 0.339596, 6.42374, 121.51, 2298.47, 43477.6, 2.77319, 52.4573,
        992.275, 18769.7, 1.19722, 22.6464, 428.375, 8103.08, 0.516851, 9.77668, 184.934, 3498.19,
    ];
    const LOCFIT_DISPS: [f64; 30] = [
        8.10705, 0.374204, 0.0548772, 0.0408124, 0.0400529, 1.30394, 0.110499, 0.0435081, 0.040157,
        1.58678, 0.120265, 0.0450843, 0.0403429, 7.1801, 0.404472, 0.0566752, 0.0407024, 0.0400295,
        0.530078, 0.0730089, 0.0421499, 0.0401275, 1.80349, 0.115338, 0.0431307, 0.0401447,
        3.21114, 0.249677, 0.0527286, 0.0406921,
    ];
    /// Below, at and above the data range
    const LOCFIT_PROBES: [f64; 12] =
        [1e-3, 0.05, 0.2, 0.22313, 0.5, 1.0, 7.5, 100.0, 1000.0, 50000.0, 66000.0, 1e6];

    fn check_local_trend(means: &[f64], disps: &[f64], expected: &[f64]) {
        let (_, trend) = fit_local_trend(means, disps).unwrap();
        for (&m, &want) in LOCFIT_PROBES.iter().zip(expected) {
            let got = trend.eval(m);
            let rel = ((got - want) / want).abs();
            assert!(rel < 1e-13, "mean {}: {} vs R {} (rel {:e})", m, got, want, rel);
        }
    }

    /// R (locfit 1.5-9.12):
    ///   fit <- locfit(log(disps) ~ log(means), weights = means)
    ///   exp(predict(fit, data.frame(logMeans = log(probes))))
    /// The values below the smallest mean follow the linear extrapolation of
    /// the residual from locfit's parametric component plus that global
    /// quadratic.
    #[test]
    fn test_local_trend_matches_r_locfit() {
        let expected = [
            49028.635345649353, 113.4253285836247, 15.100658316098947, 12.917082631097028,
            4.3499753925336897, 1.8776104360250319, 0.25872949116560312, 0.061380883701299058,
            0.04279795708644063, 0.040207535306937665, 0.04044807273371042, 0.049761942049015635,
        ];
        check_local_trend(&LOCFIT_MEANS, &LOCFIT_DISPS, &expected);
    }

    /// Sorted input takes locfit's ordered-data neighbourhood search (nbhd1),
    /// which also changes the summation order; R's values differ from the
    /// unsorted fit around 1e-12.
    #[test]
    fn test_local_trend_matches_r_locfit_sorted() {
        let mut pairs: Vec<(f64, f64)> = LOCFIT_MEANS.iter().copied().zip(LOCFIT_DISPS.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let means: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let disps: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        let expected = [
            49028.635345848008, 113.42532858367638, 15.100658316100665, 12.91708263109841,
            4.3499753925339917, 1.8776104360250976, 0.25872949116560251, 0.061380883701299134,
            0.04279795708644063, 0.040207535306937756, 0.040448072733711302, 0.049761942049077759,
        ];
        check_local_trend(&means, &disps, &expected);
    }
}
