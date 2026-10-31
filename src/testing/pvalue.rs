//! P-value calculation from test statistics

use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Calculate two-sided p-value from z-statistic
/// R equivalent: 2 * pnorm(abs(stat), lower.tail=FALSE) in nbinomWaldTest()
pub fn calculate_pvalue(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    2.0 * normal.cdf(-z.abs())
}

/// Calculate two-sided p-value from t-statistic with given degrees of freedom
/// R equivalent: 2 * pt(abs(stat), df=df, lower.tail=FALSE)
/// Used when useT=TRUE in nbinomWaldTest()
pub fn calculate_pvalue_t(stat: f64, df: f64) -> f64 {
    // a NaN df is R's NA df (df <- ifelse(df > 0, df, NA))
    if !stat.is_finite() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }

    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    2.0 * t_dist.cdf(-stat.abs())
}

/// P-value for H0: |LFC| <= threshold (altHypothesis = "greaterAbs")
/// R equivalent: results(..., lfcThreshold=T, altHypothesis="greaterAbs") in results.R
///   pnorm(-|lfc| + T, sd=se) + pnorm(-|lfc| - T, sd=se)
/// or, with `df` (useT=TRUE), pt((-|lfc| + T)/se, df) + pt((-|lfc| - T)/se, df).
/// This is the sum of both tails beyond +-T, not the older "greaterAbs2014" test.
pub fn calculate_pvalue_greater_abs(lfc: f64, se: f64, threshold: f64, df: Option<f64>) -> f64 {
    if !lfc.is_finite() || !se.is_finite() || se <= 0.0 {
        return f64::NAN;
    }
    let upper = (-lfc.abs() + threshold) / se;
    let lower = (-lfc.abs() - threshold) / se;
    match df {
        Some(df) if df > 0.0 => {
            let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
            t_dist.cdf(upper) + t_dist.cdf(lower)
        }
        Some(_) => f64::NAN,
        None => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.cdf(upper) + normal.cdf(lower)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pvalue_greater_abs_matches_r() {
        // R: pnorm(-1.5 + 1, sd = 0.3) + pnorm(-1.5 - 1, sd = 0.3) = 0.04779035
        let p = calculate_pvalue_greater_abs(1.5, 0.3, 1.0, None);
        assert!((p - 0.04779035).abs() < 1e-7, "p = {}", p);
        // symmetric in the sign of the LFC
        assert_eq!(p, calculate_pvalue_greater_abs(-1.5, 0.3, 1.0, None));
    }

    #[test]
    fn test_pvalue_symmetric() {
        let p1 = calculate_pvalue(2.0);
        let p2 = calculate_pvalue(-2.0);
        assert!((p1 - p2).abs() < 1e-10);
    }

    #[test]
    fn test_pvalue_range() {
        for z in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let p = calculate_pvalue(z);
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_pvalue_zero() {
        let p = calculate_pvalue(0.0);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pvalue_t_distribution() {
        // With large df, t-distribution approaches normal
        let p_normal = calculate_pvalue(2.0);
        let p_t_large = calculate_pvalue_t(2.0, 1000.0);
        assert!((p_normal - p_t_large).abs() < 0.001);

        // With small df, t-distribution gives larger p-values (more conservative)
        let p_t_small = calculate_pvalue_t(2.0, 3.0);
        assert!(p_t_small > p_normal);
    }
}
