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
    if !stat.is_finite() || df <= 0.0 {
        return f64::NAN;
    }

    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    2.0 * t_dist.cdf(-stat.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

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
