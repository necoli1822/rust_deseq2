//! Negative binomial distribution utilities

use statrs::function::gamma::ln_gamma;

/// Minimum mu value during GLM fitting (DESeq2's minmu parameter)
/// DESeq2 default is 0.5 - this is applied to mu before weight calculation
pub const MIN_MU: f64 = 0.5;

/// Maximum absolute value for LFC beta coefficient (not intercept)
/// DESeq2 uses 'large' parameter = 30.0 in C++ fitBeta function
/// If any |beta| > large, iteration stops and gene is flagged
/// 30.0 corresponds to ~log2(10^9) fold change
pub const MAX_LFC_BETA: f64 = 30.0;

/// Maximum eta value to prevent overflow (exp(700) ≈ 1e304)
pub const MAX_ETA: f64 = 700.0;

/// Calculate the mean of a negative binomial distribution
/// given the linear predictor eta and size factor
///
/// mu = size_factor * exp(eta)
pub fn nb_mean(eta: f64, size_factor: f64) -> f64 {
    // Only clamp to prevent overflow, not to limit LFC
    let eta_clamped = eta.clamp(-MAX_ETA, MAX_ETA);
    size_factor * eta_clamped.exp()
}

/// Calculate the variance of a negative binomial distribution
///
/// Var(Y) = mu + alpha * mu^2
pub fn nb_variance(mu: f64, alpha: f64) -> f64 {
    mu + alpha * mu * mu
}

/// Calculate the log-likelihood of a single observation
/// under the negative binomial distribution
///
/// P(Y = k | mu, alpha) = (k + r - 1 choose k) * (1/(1+alpha*mu))^r * (alpha*mu/(1+alpha*mu))^k
/// where r = 1/alpha
pub fn nb_log_likelihood(k: f64, mu: f64, alpha: f64) -> f64 {
    if mu <= 0.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let r = 1.0 / alpha;
    let p = alpha * mu / (1.0 + alpha * mu);

    // log P(Y = k | mu, alpha)
    ln_gamma(k + r) - ln_gamma(r) - ln_gamma(k + 1.0)
        + r * (1.0 - p).ln()
        + k * p.ln()
}

/// Calculate the weight for IRLS (Iteratively Reweighted Least Squares)
///
/// W = mu / (1 + alpha * mu)
/// Note: mu should already have minmu applied before calling this
pub fn nb_weight(mu: f64, alpha: f64) -> f64 {
    mu / (1.0 + alpha * mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nb_mean() {
        let mu = nb_mean(2.0, 1.0);
        assert!((mu - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_nb_variance() {
        let mu = 10.0;
        let alpha = 0.1;
        let var = nb_variance(mu, alpha);
        assert!((var - (10.0 + 0.1 * 100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_nb_log_likelihood() {
        // For Poisson limit (alpha -> 0), should approach Poisson log-likelihood
        let ll = nb_log_likelihood(5.0, 5.0, 0.001);
        assert!(ll.is_finite());
        assert!(ll < 0.0); // Log-likelihood should be negative
    }

    #[test]
    fn test_nb_weight() {
        let w = nb_weight(10.0, 0.1);
        assert!((w - 10.0 / 2.0).abs() < 1e-10);
    }
}
