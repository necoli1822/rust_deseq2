//! Statistical utility functions shared across modules
//!
//! Contains weighted quantile and variance estimation functions used by
//! both rlog transformation and LFC shrinkage.

/// qnorm(0.975) - the 97.5th percentile of the standard normal
const QNORM_0975: f64 = 1.959963984540054;

/// Weighted quantile matching R's Hmisc.wtd.quantile with type='quantile' and normwt=TRUE.
/// R equivalent: Hmisc::wtd.quantile() used in core.R
///
/// Algorithm (from DESeq2's core.R, lines 2782-2818):
/// 1. Remove NA/zero-weight entries
/// 2. Compute weighted table: sort x, aggregate weights for duplicate x values
/// 3. Normalize weights: weights *= n / sum(weights) (normwt=TRUE)
/// 4. n = sum(normalized weights)
/// 5. order = 1 + (n - 1) * prob
/// 6. low = max(floor(order), 1), high = min(low + 1, n)
/// 7. frac = order %% 1 (fractional part)
/// 8. Use step interpolation (approx f=1 = right-continuous) on cumsum(wts) -> x
/// 9. quantile = (1 - frac) * allq[low] + frac * allq[high]
pub fn weighted_quantile(x: &[f64], weights: &[f64], prob: f64) -> f64 {
    assert_eq!(x.len(), weights.len());

    // Step 1: Remove zero-weight and NaN entries
    let mut pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(weights.iter())
        .filter(|(&xi, &wi)| wi > 0.0 && !xi.is_nan() && !wi.is_nan())
        .map(|(&xi, &wi)| (xi, wi))
        .collect();

    if pairs.is_empty() {
        return 0.0;
    }

    // Step 2: Sort by x value
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Aggregate weights for duplicate x values (Hmisc.wtd.table)
    let mut unique_x: Vec<f64> = Vec::new();
    let mut agg_weights: Vec<f64> = Vec::new();
    let mut prev_x = pairs[0].0;
    let mut sum_w = pairs[0].1;
    for &(xi, wi) in &pairs[1..] {
        if xi == prev_x {
            sum_w += wi;
        } else {
            unique_x.push(prev_x);
            agg_weights.push(sum_w);
            prev_x = xi;
            sum_w = wi;
        }
    }
    unique_x.push(prev_x);
    agg_weights.push(sum_w);

    // Step 3: Normalize weights (normwt=TRUE)
    // R: weights <- weights * length(x) / sum(weights)
    // Normalization is on the raw (non-aggregated) weights, using raw count n.
    let n_raw = pairs.len() as f64;
    let raw_weight_sum: f64 = pairs.iter().map(|&(_, w)| w).sum();
    let norm_factor = n_raw / raw_weight_sum;

    // Apply normalization to the aggregated weights
    for w in agg_weights.iter_mut() {
        *w *= norm_factor;
    }

    // Step 4: n = sum(normalized weights)
    let n: f64 = agg_weights.iter().sum();

    // Step 5: order = 1 + (n - 1) * prob
    let order = 1.0 + (n - 1.0) * prob;

    // Step 6: low, high
    let low = order.floor().max(1.0);
    let high = (low + 1.0).min(n);

    // Step 7: fractional part
    let frac = order - order.floor();

    // Step 8: approx(cumsum(wts), x, xout=c(low, high), method='constant', f=1, rule=2)
    // Build cumulative weight array
    let mut cum_weights: Vec<f64> = Vec::with_capacity(agg_weights.len());
    let mut cumsum = 0.0;
    for &w in &agg_weights {
        cumsum += w;
        cum_weights.push(cumsum);
    }

    // Interpolate: step function with f=1 (right-continuous)
    let allq_low = step_interp_right(&cum_weights, &unique_x, low);
    let allq_high = step_interp_right(&cum_weights, &unique_x, high);

    // Step 9: quantile = (1 - frac) * allq_low + frac * allq_high
    (1.0 - frac) * allq_low + frac * allq_high
}

/// Step interpolation matching R's approx(method='constant', f=1, rule=2)
///
/// Given (xs, ys) pairs where xs is cumulative weights and ys is sorted unique x values,
/// find the y value at target xout.
///
/// f=1 means: for xout between xs[i] and xs[i+1], return ys[i+1] (right-continuous step).
/// rule=2: if xout < xs[0], return ys[0]; if xout > xs[n-1], return ys[n-1].
fn step_interp_right(xs: &[f64], ys: &[f64], xout: f64) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();

    if n == 0 {
        return 0.0;
    }

    // rule=2: clamp to boundary
    if xout <= xs[0] {
        return ys[0];
    }
    if xout >= xs[n - 1] {
        return ys[n - 1];
    }

    // f=1: for xout in (xs[i-1], xs[i]], return ys[i]
    // Find smallest index i where xs[i] >= xout
    for i in 0..n {
        if xs[i] >= xout {
            return ys[i];
        }
    }

    // Should not reach here
    ys[n - 1]
}

/// Match weighted upper quantile for variance estimation.
/// R equivalent: matchWeightedUpperQuantileForVariance() in core.R
///
/// R: matchWeightedUpperQuantileForVariance(x, weights, upperQuantile=0.05)
///   sdEst = Hmisc.wtd.quantile(abs(x), weights, probs=1-upperQuantile, normwt=TRUE)
///           / qnorm(1 - upperQuantile/2)
///   return sdEst^2
pub fn match_weighted_upper_quantile_for_variance(x: &[f64], weights: &[f64], upper_quantile: f64) -> f64 {

    // Take absolute values
    let abs_x: Vec<f64> = x.iter().map(|&v| v.abs()).collect();

    // probs = 1 - upperQuantile = 0.95
    let prob = 1.0 - upper_quantile;

    let wtd_q95 = weighted_quantile(&abs_x, weights, prob);
    log::debug!("abs_lfc_wtd_q95={:.15}", wtd_q95);
    log::debug!("qnorm_0975={:.15}", QNORM_0975);

    let sd_est = wtd_q95 / QNORM_0975;
    log::debug!("sdEst={:.15}", sd_est);

    let var_est = sd_est * sd_est;

    // Ensure a minimum prior variance to avoid degenerate fits
    if var_est <= 0.0 || !var_est.is_finite() {
        1e-6
    } else {
        var_est
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_quantile_simple() {
        // Simple uniform weights - should match regular quantile
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let q50 = weighted_quantile(&x, &w, 0.5);
        assert!(
            (q50 - 3.0).abs() < 1e-10,
            "median of 1..5 should be 3.0, got {}",
            q50
        );
    }

    #[test]
    fn test_weighted_quantile_skewed() {
        // Heavily weighted toward high values
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 100.0];
        let q50 = weighted_quantile(&x, &w, 0.5);
        // With most weight on 5.0, the median should be close to 5.0
        assert!(q50 >= 4.0, "weighted median should be >= 4.0, got {}", q50);
    }

    #[test]
    fn test_weighted_quantile_upper() {
        let x = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let q95 = weighted_quantile(&x, &w, 0.95);
        assert!(
            q95.is_finite(),
            "95th percentile should be finite, got {}",
            q95
        );
        assert!(q95 >= 2.0 && q95 <= 5.0);
    }

    #[test]
    fn test_match_weighted_upper_quantile_for_variance() {
        // Test with known values
        let x = vec![0.1, -0.2, 0.3, -0.1, 0.5, -0.3, 0.2, -0.4, 0.15, -0.25];
        let w = vec![1.0; 10];
        let var = match_weighted_upper_quantile_for_variance(&x, &w, 0.05);
        assert!(var > 0.0, "variance should be positive, got {}", var);
        assert!(var.is_finite(), "variance should be finite");
    }

    #[test]
    fn test_step_interp_right_basic() {
        let cum = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        assert_eq!(step_interp_right(&cum, &x, 2.5), 30.0);
        assert_eq!(step_interp_right(&cum, &x, 1.0), 10.0);
        assert_eq!(step_interp_right(&cum, &x, 0.5), 10.0);
        assert_eq!(step_interp_right(&cum, &x, 5.0), 50.0);
        assert_eq!(step_interp_right(&cum, &x, 6.0), 50.0);
    }
}
