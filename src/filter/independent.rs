//! Independent filtering to improve power
//! Matches DESeq2's independent filtering implementation in results.R

use crate::io::DESeqResults;
use crate::testing::benjamini_hochberg;

/// Apply independent filtering based on mean expression
/// R equivalent: pvalueAdjustment() (independent filtering) in results.R
///
/// Filters out genes with low mean expression before multiple testing correction.
/// This can improve power by reducing the multiple testing burden.
///
/// Matches DESeq2's pvalueAdjustment function in results.R:670-749
///
/// The `alpha` parameter controls the significance level used to optimize the
/// filtering threshold. R DESeq2 defaults to alpha=0.1 in results().
pub fn independent_filtering(results: &mut DESeqResults, alpha: f64) {
    let n = results.gene_ids.len();
    if n == 0 {
        return;
    }

    let base_means = &results.base_means;

    // Calculate lowerQuantile = proportion of zeros (DESeq2: mean(filter == 0))
    let zero_count = base_means.iter().filter(|&&m| m == 0.0 || !m.is_finite()).count();
    let lower_quantile = zero_count as f64 / n as f64;

    // upperQuantile = 0.95 (or 1 if lowerQuantile >= 0.95)
    let upper_quantile = if lower_quantile < 0.95 { 0.95 } else { 1.0 };

    // Generate 50 theta values (DESeq2: seq(lowerQuantile, upperQuantile, length.out=50))
    let n_theta = 50;
    let thetas: Vec<f64> = (0..n_theta)
        .map(|i| lower_quantile + (upper_quantile - lower_quantile) * (i as f64) / (n_theta as f64 - 1.0))
        .collect();

    // Get sorted base means for quantile calculation
    let mut sorted_means: Vec<f64> = base_means
        .iter()
        .filter(|m| m.is_finite())
        .copied()
        .collect();
    sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted_means.is_empty() {
        return;
    }

    // Calculate cutoffs for each theta (DESeq2: quantile(filter, theta))
    // Uses R's quantile type=7 (default): linear interpolation
    // Formula: h = (n-1)*p + 1, result = x[floor(h)] + (h - floor(h)) * (x[ceil(h)] - x[floor(h)])
    let cutoffs: Vec<f64> = thetas
        .iter()
        .map(|&theta| quantile_type7(&sorted_means, theta))
        .collect();

    // For each theta, calculate adjusted p-values and count rejections
    let mut all_padj: Vec<Vec<f64>> = Vec::with_capacity(n_theta);
    let mut num_rej: Vec<usize> = Vec::with_capacity(n_theta);

    for &cutoff in &cutoffs {
        // Filter p-values: keep only genes with baseMean >= cutoff
        let filtered_pvalues: Vec<f64> = results
            .pvalues
            .iter()
            .zip(base_means.iter())
            .map(|(&p, &m)| {
                if m >= cutoff && p.is_finite() {
                    p
                } else {
                    f64::NAN
                }
            })
            .collect();

        let padj = benjamini_hochberg(&filtered_pvalues);
        let rejections = padj.iter().filter(|&&p| p.is_finite() && p < alpha).count();

        all_padj.push(padj);
        num_rej.push(rejections);
    }

    // DESeq2: If max rejections <= 10, don't filter (use j=0)
    let max_rej = *num_rej.iter().max().unwrap_or(&0);
    let best_j = if max_rej <= 10 {
        0
    } else {
        // DESeq2 uses lowess(numRej ~ theta, f=1/5) for smoothing
        // f=1/5 means use 20% of points for local regression
        // iter=3 matches R's lowess() default robustness iterations
        let lowess_fit = lowess_smooth(&thetas, &num_rej, 0.2, 3);

        // Calculate residuals and RMSE (only for non-zero rejections)
        let residuals: Vec<f64> = num_rej
            .iter()
            .zip(lowess_fit.iter())
            .filter(|(&n, _)| n > 0)
            .map(|(&n, &f)| n as f64 - f)
            .collect();

        let max_fit = lowess_fit.iter().cloned().fold(f64::MIN, f64::max);
        let rmse = if !residuals.is_empty() {
            (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt()
        } else {
            0.0
        };

        let thresh = max_fit - rmse;

        // Find first theta where numRej > thresh
        let mut selected = 0;
        for (j, &rej) in num_rej.iter().enumerate() {
            if (rej as f64) > thresh {
                selected = j;
                break;
            }
        }

        // Fallback: if no theta found, try 90% of maxFit, then 80%
        if selected == 0 && (num_rej[0] as f64) <= thresh {
            for (j, &rej) in num_rej.iter().enumerate() {
                if (rej as f64) > 0.9 * max_fit {
                    selected = j;
                    break;
                }
            }
        }
        if selected == 0 && (num_rej[0] as f64) <= 0.9 * max_fit {
            for (j, &rej) in num_rej.iter().enumerate() {
                if (rej as f64) > 0.8 * max_fit {
                    selected = j;
                    break;
                }
            }
        }

        selected
    };

    // Apply the selected theta
    results.padj = all_padj.into_iter().nth(best_j).unwrap_or_else(|| {
        benjamini_hochberg(&results.pvalues)
    });

    // Debug: print numRej curve
    log::debug!("numRej curve:");
    for j in (0..n_theta).step_by(5) {
        log::debug!("  theta={:.3}, cutoff={:.2}, numRej={}", thetas[j], cutoffs[j], num_rej[j]);
    }

    log::info!(
        "Independent filtering: theta={:.3}, cutoff={:.2}, rejections={}, max_rej={}",
        thetas[best_j],
        cutoffs[best_j],
        num_rej[best_j],
        max_rej
    );
}

/// R's quantile type=7 (default) implementation
/// Formula: h = (n-1)*p + 1
///          result = x[floor(h)-1] + (h - floor(h)) * (x[ceil(h)-1] - x[floor(h)-1])
/// Note: R uses 1-based indexing, so we adjust for 0-based Rust indexing
fn quantile_type7(sorted_x: &[f64], p: f64) -> f64 {
    let n = sorted_x.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted_x[0];
    }

    // R's type=7: h = (n-1)*p + 1 (1-based index)
    // Convert to 0-based: h0 = (n-1)*p
    let h = (n as f64 - 1.0) * p;
    let h_floor = h.floor() as usize;
    let h_ceil = h.ceil() as usize;

    // Clamp indices to valid range
    let lo = h_floor.min(n - 1);
    let hi = h_ceil.min(n - 1);

    if lo == hi {
        sorted_x[lo]
    } else {
        // Linear interpolation
        let frac = h - h_floor as f64;
        sorted_x[lo] + frac * (sorted_x[hi] - sorted_x[lo])
    }
}

/// LOWESS (Locally Weighted Scatterplot Smoothing) implementation
/// Faithful port of R's clowess() C implementation from lowess.c
///
/// * `x` - x values (must be sorted in ascending order)
/// * `y` - y values (num_rej as usize)
/// * `f` - smoother span fraction (R default: 2/3, DESeq2 uses 1/5)
/// * `nsteps` - number of robustness iterations (R default: 3)
fn lowess_smooth(x: &[f64], y: &[usize], f: f64, nsteps: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Convert y to f64
    let y_f64: Vec<f64> = y.iter().map(|&yi| yi as f64).collect();

    if n == 1 {
        return vec![y_f64[0]];
    }

    // Delta: R's lowess() wrapper uses delta = 0.01 * diff(range(x))
    let delta = 0.01 * (x[n - 1] - x[0]);

    let mut ys = vec![0.0; n];
    let mut rw = vec![0.0; n];
    let mut res = vec![0.0; n];

    clowess(x, &y_f64, f, nsteps, delta, &mut ys, &mut rw, &mut res);

    ys
}

/// Port of R's lowest() function from lowess.c
/// Computes the weighted local regression at a single point.
///
/// All indices are 0-based (unlike the R C code which uses 1-based via pointer decrement).
///
/// * `x` - sorted x values (0-based, length n)
/// * `y` - y values (0-based, length n)
/// * `n` - number of points
/// * `xs` - the x value at which to evaluate
/// * `nleft` - left boundary of window (0-based index)
/// * `nright` - right boundary of window (0-based index)
/// * `w` - workspace for weights (0-based, length n)
/// * `userw` - whether to use robustness weights
/// * `rw` - robustness weights (0-based, length n)
///
/// Returns `Some(ys)` if successful, `None` if total weight is zero.
fn lowest(
    x: &[f64],
    y: &[f64],
    n: usize,
    xs: f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    userw: bool,
    rw: &[f64],
) -> Option<f64> {
    let range = x[n - 1] - x[0];
    let h = f64::max(xs - x[nleft], x[nright] - xs);
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;

    let mut a = 0.0_f64;
    let mut j = nleft;
    while j < n {
        w[j] = 0.0;
        let r = (x[j] - xs).abs();
        if r <= h9 {
            if r <= h1 {
                w[j] = 1.0;
            } else {
                // tricube: (1 - (r/h)^3)^3
                let u = r / h;
                w[j] = (1.0 - u * u * u) * (1.0 - u * u * u) * (1.0 - u * u * u);
            }
            if userw {
                w[j] *= rw[j];
            }
            a += w[j];
        } else if x[j] > xs {
            break;
        }
        j += 1;
    }

    // nrt: last index that got a weight (0-based)
    let nrt = j - 1;

    if a <= 0.0 {
        return None;
    }

    // Normalize weights
    for j in nleft..=nrt {
        w[j] /= a;
    }

    if h > 0.0 {
        // Weighted mean of x
        a = 0.0;
        for j in nleft..=nrt {
            a += w[j] * x[j];
        }
        let b = xs - a;

        // Weighted sum of squared deviations from weighted mean
        let mut c = 0.0;
        for j in nleft..=nrt {
            c += w[j] * (x[j] - a) * (x[j] - a);
        }

        if c.sqrt() > 0.001 * range {
            let b_over_c = b / c;
            for j in nleft..=nrt {
                w[j] *= b_over_c * (x[j] - a) + 1.0;
            }
        }
    }

    // Compute fitted value
    let mut ys = 0.0;
    for j in nleft..=nrt {
        ys += w[j] * y[j];
    }
    Some(ys)
}

/// Port of R's clowess() function from lowess.c
/// Main LOWESS algorithm with robustness iterations and delta optimization.
///
/// * `x` - sorted x values (0-based)
/// * `y` - y values (0-based)
/// * `f` - smoother span fraction
/// * `nsteps` - number of robustness iterations
/// * `delta` - skip distance for optimization
/// * `ys` - output: fitted values
/// * `rw` - workspace: robustness weights
/// * `res` - workspace: residuals (also used as weight workspace in lowest())
fn clowess(
    x: &[f64],
    y: &[f64],
    f: f64,
    nsteps: usize,
    delta: f64,
    ys: &mut [f64],
    rw: &mut [f64],
    res: &mut [f64],
) {
    let n = x.len();
    debug_assert!(n >= 2);

    // R: ns = imax2(2, imin2(n, (int)(f*n + 1e-7)))
    let ns = 2.max((n as isize).min((f * n as f64 + 1e-7) as isize)) as usize;

    let mut iter = 1usize;
    while iter <= nsteps + 1 {
        // 0-based: nleft=0, nright=ns-1
        let mut nleft: usize = 0;
        let mut nright: usize = ns - 1;
        // last: tracks the last point where lowest() was evaluated
        // Using isize -1 to represent "no point yet" (R uses last=0 with 1-based)
        let mut last: isize = -1;
        // i: current point to evaluate (0-based)
        let mut i: usize = 0;

        loop {
            // Slide window: if right boundary can expand and left point is farther
            // than next right point, slide the window right
            if nright < n - 1 {
                // R (1-based): d1 = x[i] - x[nleft], d2 = x[nright+1] - x[i]
                let d1 = x[i] - x[nleft];
                let d2 = x[nright + 1] - x[i];
                if d1 > d2 {
                    nleft += 1;
                    nright += 1;
                    continue;
                }
            }

            // Evaluate lowest() at point i
            let fit = lowest(x, y, n, x[i], nleft, nright, res, iter > 1, rw);
            match fit {
                Some(val) => ys[i] = val,
                None => ys[i] = y[i],
            }

            // Interpolate skipped points between last and i
            // R (1-based): if (last < i-1), 0-based equivalent:
            if last >= 0 && (last as usize) + 1 < i {
                let last_u = last as usize;
                let denom = x[i] - x[last_u];
                for j in (last_u + 1)..i {
                    let alpha = (x[j] - x[last_u]) / denom;
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last_u];
                }
            }

            // Update last
            last = i as isize;
            // R: cut = x[last] + delta (1-based last, but same value)
            let cut = x[i] + delta;

            // Advance i: skip points within delta that have the same x value
            // R (1-based): for (i = last+1; i <= n; i++)
            let last_u = i;
            i = last_u + 1;
            while i < n {
                if x[i] > cut {
                    break;
                }
                if x[i] == x[last_u] {
                    ys[i] = ys[last_u];
                    last = i as isize;
                }
                i += 1;
            }

            // R: i = imax2(last+1, i-1)
            // 0-based equivalent: same logic
            let last_val = last as usize;
            i = (last_val + 1).max(if i > 0 { i - 1 } else { 0 });

            // R: if (last >= n) break (1-based), 0-based: last >= n-1
            if last_val >= n - 1 {
                break;
            }
        }

        // Compute residuals: res[i] = y[i] - ys[i]
        // R: res[i] = y[i+1] - ys[i+1] (because y/ys are 1-based there, res is 0-based)
        for i in 0..n {
            res[i] = y[i] - ys[i];
        }

        // sc = mean(|residuals|)
        let mut sc = 0.0_f64;
        for i in 0..n {
            sc += res[i].abs();
        }
        sc /= n as f64;

        // If we've done all robustness iterations, stop
        if iter > nsteps {
            break;
        }

        // Compute absolute residuals into rw for partial sorting
        for i in 0..n {
            rw[i] = res[i].abs();
        }

        // Compute cmad (6 * median of |residuals|) using partial sort
        // R uses rPsort for partial sorting; Rust equivalent is select_nth_unstable
        let m1 = n / 2; // 0-based index for median
        let cmad = if n % 2 == 0 {
            // Even n: R uses m1 = n/2, m2 = n-m1-1, cmad = 3*(rw[m1]+rw[m2])
            // rPsort(rw, n, m1) then rPsort(rw, n, m2)
            // After first partial sort, rw[m1] is correct.
            // m2 = n - m1 - 1 = n - n/2 - 1 = n/2 - 1
            rw.select_nth_unstable_by(m1, |a, b| a.partial_cmp(b).unwrap());
            let val_m1 = rw[m1];
            let m2 = n - m1 - 1; // = m1 - 1 for even n
            // Need to partial sort for m2 as well. Since m2 < m1, and after
            // select_nth_unstable(m1), elements 0..m1 are <= rw[m1].
            // We need to find the m2-th element among 0..m1.
            rw[..m1].select_nth_unstable_by(m2, |a, b| a.partial_cmp(b).unwrap());
            let val_m2 = rw[m2];
            3.0 * (val_m1 + val_m2)
        } else {
            // Odd n: cmad = 6 * rw[m1]
            rw.select_nth_unstable_by(m1, |a, b| a.partial_cmp(b).unwrap());
            6.0 * rw[m1]
        };

        // Early termination: if cmad is tiny relative to sc
        if cmad < 1e-7 * sc {
            break;
        }

        // Compute bisquare robustness weights
        let c9 = 0.999 * cmad;
        let c1 = 0.001 * cmad;
        for i in 0..n {
            let r = res[i].abs();
            if r <= c1 {
                rw[i] = 1.0;
            } else if r <= c9 {
                // bisquare: (1 - (r/cmad)^2)^2
                let u = r / cmad;
                rw[i] = (1.0 - u * u) * (1.0 - u * u);
            } else {
                rw[i] = 0.0;
            }
        }

        iter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Contrast;

    #[test]
    fn test_independent_filtering() {
        let mut results = DESeqResults {
            gene_ids: (0..100).map(|i| format!("gene_{}", i)).collect(),
            base_means: (0..100).map(|i| (i + 1) as f64 * 10.0).collect(),
            base_vars: vec![50.0; 100],
            log2_fold_changes: vec![1.0; 100],
            lfc_se: vec![0.5; 100],
            stat: vec![2.0; 100],
            pvalues: (0..100).map(|i| 0.001 + i as f64 * 0.01).collect(),
            padj: vec![0.0; 100],
            dispersions: vec![0.1; 100],
            gene_wise_dispersions: vec![0.1; 100],
            trended_dispersions: vec![0.1; 100],
            contrast: Contrast {
                variable: "treatment".to_string(),
                numerator: "treated".to_string(),
                denominator: "control".to_string(),
            },
        };

        independent_filtering(&mut results, 0.1);

        // Check that padj values are set
        assert!(results.padj.iter().any(|&p| p.is_finite()));
    }

    #[test]
    fn test_lowess_robustness_iterations() {
        // Test data with outliers - robustness iterations should handle these better
        let x = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let y_clean = vec![10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
        let y_with_outlier = vec![10, 12, 14, 50, 18, 20, 22, 24, 26, 28, 30]; // outlier at index 3

        // Fit without robustness iterations (nsteps=0)
        let fit_no_iter = lowess_smooth(&x, &y_with_outlier, 0.5, 0);

        // Fit with robustness iterations (nsteps=3, matching R default)
        let fit_with_iter = lowess_smooth(&x, &y_with_outlier, 0.5, 3);

        // Both fits should produce n values
        assert_eq!(fit_no_iter.len(), x.len());
        assert_eq!(fit_with_iter.len(), x.len());

        // The robust fit should be less influenced by the outlier at index 3
        let true_value = 16.0;
        let outlier_idx = 3;

        let error_no_iter = (fit_no_iter[outlier_idx] - true_value).abs();
        let error_with_iter = (fit_with_iter[outlier_idx] - true_value).abs();

        // Robustness iterations should produce a fit closer to the true value
        assert!(
            error_with_iter <= error_no_iter,
            "Robust LOWESS should be no worse than non-robust at outlier. \
             Error without iterations: {:.2}, with iterations: {:.2}",
            error_no_iter,
            error_with_iter
        );

        // Also check that clean data gives similar results with and without iterations
        let fit_clean_no_iter = lowess_smooth(&x, &y_clean, 0.5, 0);
        let fit_clean_with_iter = lowess_smooth(&x, &y_clean, 0.5, 3);

        // For clean linear data, the difference should be small
        let max_diff: f64 = fit_clean_no_iter
            .iter()
            .zip(fit_clean_with_iter.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 2.0,
            "For clean data, robust and non-robust fits should be similar. Max diff: {:.2}",
            max_diff
        );
    }

    #[test]
    fn test_lowess_basic_linear() {
        // Perfect linear data: lowess should reproduce it closely
        let x = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let y = vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        let fit = lowess_smooth(&x, &y, 0.5, 3);
        assert_eq!(fit.len(), x.len());

        // Each fitted value should be close to the true linear value
        for (i, &f_val) in fit.iter().enumerate() {
            let expected = i as f64 * 10.0;
            assert!(
                (f_val - expected).abs() < 5.0,
                "At index {}, expected ~{:.1}, got {:.1}",
                i, expected, f_val
            );
        }
    }

    #[test]
    fn test_lowess_single_point() {
        let x = vec![1.0];
        let y = vec![42];
        let fit = lowess_smooth(&x, &y, 0.5, 3);
        assert_eq!(fit.len(), 1);
        assert_eq!(fit[0], 42.0);
    }

    #[test]
    fn test_lowess_two_points() {
        let x = vec![0.0, 1.0];
        let y = vec![10, 20];
        let fit = lowess_smooth(&x, &y, 1.0, 0);
        assert_eq!(fit.len(), 2);
        // With f=1.0, all points are used: should interpolate linearly
        assert!((fit[0] - 10.0).abs() < 1.0);
        assert!((fit[1] - 20.0).abs() < 1.0);
    }
}
