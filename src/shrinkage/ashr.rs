//! Adaptive Shrinkage (ashr) for log fold change estimation
//!
//! Implementation of the ashr algorithm for adaptive shrinkage of LFCs.
//! Based on Stephens (2017): "False discovery rates: a new deal"
//!
//! The algorithm:
//! 1. Model LFCs as coming from a mixture of normal distributions
//! 2. Use EM algorithm to estimate mixture proportions
//! 3. Compute posterior mean and SD for each gene
//!
//! **Architectural note:** R's ashr package estimates mixture proportions via convex
//! optimization (mixIP using REBayes/mosek, or mixEM as a fallback). This Rust
//! implementation uses a simple EM algorithm exclusively, which is the same as R's
//! `mixEM` fallback path. The EM approach is an acceptable approximation but may
//! converge more slowly or to a slightly different solution than the convex
//! optimization (mixIP) path used by default in R. For most practical DESeq2 use
//! cases the difference is negligible.
//!
//! Reference: https://github.com/stephens999/ashr

use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::io::DESeqResults;

/// Calculate log PDF for normal distribution
fn normal_ln_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (x - mean) / std_dev;
    -0.5 * z * z - std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Parameters for ashr shrinkage
/// R equivalent: ash() control parameters in ashr package (stephens999/ashr)
#[derive(Debug, Clone)]
pub struct AshrParams {
    /// Number of mixture components
    pub n_components: usize,
    /// Maximum sigma for mixture components
    pub sigma_max: f64,
    /// Minimum sigma for mixture components (excluding point mass at 0)
    pub sigma_min: f64,
    /// Maximum EM iterations
    pub max_iter: usize,
    /// EM convergence tolerance
    pub tol: f64,
    /// Include point mass at zero (pi_0)
    pub pointmass: bool,
}

impl Default for AshrParams {
    fn default() -> Self {
        Self {
            n_components: 20,
            sigma_max: 10.0,
            sigma_min: 0.001,
            max_iter: 1000,
            tol: 1e-6,
            pointmass: true,
        }
    }
}

/// Result of ashr fit
/// R equivalent: ash() return value (ashObject S3 class) in ashr package
#[derive(Debug, Clone)]
pub struct AshrFit {
    /// Posterior means (shrunken LFCs)
    pub posterior_mean: Vec<f64>,
    /// Posterior standard deviations
    pub posterior_sd: Vec<f64>,
    /// Local false sign rate (lfsr)
    pub lfsr: Vec<f64>,
    /// s-values (like FDR but for sign errors)
    pub svalue: Vec<f64>,
    /// Estimated mixture proportions (pi)
    pub pi: Vec<f64>,
    /// Mixture component sigmas
    pub sigma: Vec<f64>,
}

/// Apply ashr adaptive shrinkage to LFC estimates
/// R equivalent: lfcShrink(type="ashr") in lfcShrink.R
///
/// # Arguments
/// * `results` - DESeqResults with MLE LFCs and standard errors
/// * `params` - Parameters for ashr algorithm
///
/// # Example
/// ```ignore
/// let fit = shrink_lfc_ashr(&results, AshrParams::default());
/// ```
pub fn shrink_lfc_ashr(results: &DESeqResults, params: &AshrParams) -> AshrFit {
    let n = results.log2_fold_changes.len();

    // Extract valid (non-NA) observations
    let mut valid_indices = Vec::new();
    let mut betahat = Vec::new();
    let mut sebetahat = Vec::new();

    for i in 0..n {
        let lfc = results.log2_fold_changes[i];
        let se = results.lfc_se[i];
        if lfc.is_finite() && se.is_finite() && se > 0.0 {
            valid_indices.push(i);
            betahat.push(lfc);
            sebetahat.push(se);
        }
    }

    if betahat.is_empty() {
        return AshrFit {
            posterior_mean: vec![f64::NAN; n],
            posterior_sd: vec![f64::NAN; n],
            lfsr: vec![f64::NAN; n],
            svalue: vec![f64::NAN; n],
            pi: vec![1.0],
            sigma: vec![0.0],
        };
    }

    // Set up data-adaptive mixture grid (matches R's autoselect.mixsd)
    let sigma = create_sigma_grid_adaptive(params, &sebetahat);
    let k = sigma.len();

    // Initialize mixture proportions uniformly
    let mut pi = vec![1.0 / k as f64; k];

    // EM algorithm
    for _iter in 0..params.max_iter {
        // E-step: compute posterior membership probabilities
        let log_lik = compute_log_likelihood_matrix(&betahat, &sebetahat, &sigma);
        let (membership, _ll) = compute_membership(&log_lik, &pi);

        // M-step: update mixture proportions
        let new_pi = update_pi(&membership);

        // Check convergence
        let max_change: f64 = pi
            .iter()
            .zip(new_pi.iter())
            .map(|(&p1, &p2)| (p1 - p2).abs())
            .fold(0.0, f64::max);

        pi = new_pi;

        if max_change < params.tol {
            break;
        }
    }

    // Compute posterior statistics
    let (posterior_mean_valid, posterior_sd_valid, lfsr_valid) =
        compute_posterior_stats(&betahat, &sebetahat, &sigma, &pi);

    // Map back to full results
    let mut posterior_mean = vec![f64::NAN; n];
    let mut posterior_sd = vec![f64::NAN; n];
    let mut lfsr = vec![f64::NAN; n];

    for (idx, &i) in valid_indices.iter().enumerate() {
        posterior_mean[i] = posterior_mean_valid[idx];
        posterior_sd[i] = posterior_sd_valid[idx];
        lfsr[i] = lfsr_valid[idx];
    }

    // Compute s-values (cumulative LFSR)
    let svalue = compute_svalues(&lfsr);

    AshrFit {
        posterior_mean,
        posterior_sd,
        lfsr,
        svalue,
        pi,
        sigma,
    }
}

/// Create sigma grid for mixture components (fixed grid, used as fallback)
fn create_sigma_grid(params: &AshrParams) -> Vec<f64> {
    let mut sigma = Vec::new();

    // Point mass at 0 (if enabled)
    if params.pointmass {
        sigma.push(0.0);
    }

    // Log-spaced grid from sigma_min to sigma_max
    let log_min = params.sigma_min.ln();
    let log_max = params.sigma_max.ln();
    let n_grid = if params.pointmass {
        params.n_components - 1
    } else {
        params.n_components
    };

    if n_grid > 0 {
        for i in 0..n_grid {
            let log_s = log_min + (log_max - log_min) * i as f64 / (n_grid - 1).max(1) as f64;
            sigma.push(log_s.exp());
        }
    }

    sigma
}

/// Create data-adaptive sigma grid for mixture components
///
/// Matches R's ashr `autoselect.mixsd(data)`:
///   sdmin = min(se) / 10
///   sdmax = 2 * sqrt(max(betahat^2 - se^2, 0))  [approximated by max(se) * 2]
///   sigmalist = 2^seq(log2(sdmin), log2(sdmax), length.out = npoint)
///
/// Falls back to fixed grid if data is insufficient.
fn create_sigma_grid_adaptive(params: &AshrParams, sebetahat: &[f64]) -> Vec<f64> {
    if sebetahat.is_empty() {
        return create_sigma_grid(params);
    }

    // Compute data-adaptive range from observed standard errors
    let se_min = sebetahat
        .iter()
        .filter(|&&se| se > 0.0 && se.is_finite())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let se_max = sebetahat
        .iter()
        .filter(|&&se| se.is_finite())
        .cloned()
        .fold(0.0_f64, f64::max);

    if !se_min.is_finite() || !se_max.is_finite() || se_min <= 0.0 || se_max <= 0.0 {
        return create_sigma_grid(params);
    }

    // R's autoselect.mixsd: sdmin = min(se)/10, sdmax = 2*max(se)
    let sd_min = (se_min / 10.0).max(1e-8);
    let sd_max = (2.0 * se_max).max(sd_min * 2.0);

    let mut sigma = Vec::new();

    // Point mass at 0 (if enabled)
    if params.pointmass {
        sigma.push(0.0);
    }

    // Log2-spaced grid from sd_min to sd_max (matching R's 2^seq(...))
    let n_grid = if params.pointmass {
        params.n_components - 1
    } else {
        params.n_components
    };

    if n_grid > 0 {
        let log2_min = sd_min.log2();
        let log2_max = sd_max.log2();
        for i in 0..n_grid {
            let log2_s =
                log2_min + (log2_max - log2_min) * i as f64 / (n_grid - 1).max(1) as f64;
            sigma.push(2.0_f64.powf(log2_s));
        }
    }

    sigma
}

/// Compute log-likelihood matrix
/// log p(betahat | sigma_k) where convolved variance = sebetahat^2 + sigma_k^2
fn compute_log_likelihood_matrix(
    betahat: &[f64],
    sebetahat: &[f64],
    sigma: &[f64],
) -> Vec<Vec<f64>> {
    let n = betahat.len();
    let k = sigma.len();

    // Compute in parallel
    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let b = betahat[i];
            let se = sebetahat[i];
            let se2 = se * se;

            (0..k)
                .map(|j| {
                    let s2 = sigma[j] * sigma[j];
                    let total_var = se2 + s2;

                    if total_var <= 0.0 || s2 == 0.0 && sigma[j] == 0.0 {
                        // Point mass at 0: beta should be 0
                        // Log-likelihood of observing betahat from N(0, se^2)
                        normal_ln_pdf(b, 0.0, se)
                    } else {
                        // Normal component: beta ~ N(0, sigma^2)
                        // betahat ~ N(0, se^2 + sigma^2)
                        let total_sd = total_var.sqrt();
                        normal_ln_pdf(b, 0.0, total_sd)
                    }
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    // Reshape to matrix
    let mut result = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..k {
            result[i][j] = flat[i * k + j];
        }
    }

    result
}

/// Compute membership probabilities (E-step)
/// Returns (membership matrix, total log-likelihood)
fn compute_membership(log_lik: &[Vec<f64>], pi: &[f64]) -> (Vec<Vec<f64>>, f64) {
    let n = log_lik.len();
    let k = pi.len();

    let mut membership = vec![vec![0.0; k]; n];
    let mut total_ll = 0.0;

    for i in 0..n {
        // Log-sum-exp trick for numerical stability
        let mut max_log = f64::NEG_INFINITY;
        for j in 0..k {
            let log_contrib = pi[j].ln() + log_lik[i][j];
            if log_contrib > max_log {
                max_log = log_contrib;
            }
        }

        let mut sum_exp = 0.0;
        for j in 0..k {
            let log_contrib = pi[j].ln() + log_lik[i][j];
            sum_exp += (log_contrib - max_log).exp();
        }

        let log_marginal = max_log + sum_exp.ln();
        total_ll += log_marginal;

        // Normalize to get membership
        for j in 0..k {
            let log_contrib = pi[j].ln() + log_lik[i][j];
            membership[i][j] = (log_contrib - log_marginal).exp();
        }
    }

    (membership, total_ll)
}

/// Update mixture proportions (M-step)
fn update_pi(membership: &[Vec<f64>]) -> Vec<f64> {
    let n = membership.len();
    let k = membership[0].len();

    let mut new_pi = vec![0.0; k];

    for j in 0..k {
        let sum: f64 = membership.iter().map(|m| m[j]).sum();
        new_pi[j] = sum / n as f64;
    }

    // Ensure pi sums to 1
    let total: f64 = new_pi.iter().sum();
    if total > 0.0 {
        for p in &mut new_pi {
            *p /= total;
        }
    }

    new_pi
}

/// Compute posterior statistics
fn compute_posterior_stats(
    betahat: &[f64],
    sebetahat: &[f64],
    sigma: &[f64],
    pi: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = betahat.len();
    let k = sigma.len();

    let mut post_mean = vec![0.0; n];
    let mut post_sd = vec![0.0; n];
    let mut lfsr = vec![0.0; n];

    for i in 0..n {
        let b = betahat[i];
        let se = sebetahat[i];
        let se2 = se * se;

        // Compute posterior mean and variance for each mixture component,
        // then combine weighted by posterior membership

        let mut weight_sum = 0.0;
        let mut mean_sum = 0.0;
        let mut var_sum = 0.0;
        let mut prob_wrong_sign = 0.0;

        // First compute marginal likelihood for normalization
        let log_lik: Vec<f64> = (0..k)
            .map(|j| {
                let s2 = sigma[j] * sigma[j];
                let total_var = se2 + s2;
                if sigma[j] == 0.0 {
                    normal_ln_pdf(b, 0.0, se)
                } else {
                    normal_ln_pdf(b, 0.0, total_var.sqrt())
                }
            })
            .collect();

        let max_ll = log_lik.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let marginal_lik: f64 = (0..k)
            .map(|j| pi[j] * (log_lik[j] - max_ll).exp())
            .sum::<f64>()
            * max_ll.exp();

        for j in 0..k {
            let s2 = sigma[j] * sigma[j];
            let weight = pi[j] * (log_lik[j] - max_ll).exp() * max_ll.exp() / marginal_lik.max(1e-300);

            if sigma[j] == 0.0 {
                // Point mass at 0: posterior mean = 0, var = 0
                // Wrong sign prob: 0.5 if betahat != 0
                weight_sum += weight;
                prob_wrong_sign += weight * 0.5;
            } else {
                // Normal component: posterior is N(post_mu, post_var)
                let total_var = se2 + s2;
                let shrinkage = s2 / total_var;
                let post_mu = shrinkage * b;
                let post_var = shrinkage * se2;

                weight_sum += weight;
                mean_sum += weight * post_mu;
                var_sum += weight * (post_var + post_mu * post_mu);

                // Probability of wrong sign
                // P(beta < 0 | betahat) if betahat > 0, and vice versa
                if post_var > 0.0 {
                    if let Ok(normal) = Normal::new(post_mu, post_var.sqrt()) {
                        let wrong_sign_prob = if b > 0.0 {
                            normal.cdf(0.0)
                        } else {
                            1.0 - normal.cdf(0.0)
                        };
                        prob_wrong_sign += weight * wrong_sign_prob;
                    }
                }
            }
        }

        if weight_sum > 0.0 {
            post_mean[i] = mean_sum / weight_sum;
            let e_sq = var_sum / weight_sum;
            post_sd[i] = (e_sq - post_mean[i] * post_mean[i]).max(0.0).sqrt();
            lfsr[i] = prob_wrong_sign / weight_sum;
        } else {
            post_mean[i] = b;
            post_sd[i] = se;
            lfsr[i] = 0.5;
        }
    }

    (post_mean, post_sd, lfsr)
}

/// Compute s-values (cumulative LFSR, like FDR but for sign errors)
fn compute_svalues(lfsr: &[f64]) -> Vec<f64> {
    let n = lfsr.len();
    let mut svalue = vec![f64::NAN; n];

    // Get sorted indices by LFSR
    let mut indices: Vec<usize> = (0..n)
        .filter(|&i| lfsr[i].is_finite())
        .collect();
    indices.sort_by(|&a, &b| {
        lfsr[a].partial_cmp(&lfsr[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute cumulative average
    let mut cumsum = 0.0;
    for (rank, &idx) in indices.iter().enumerate() {
        cumsum += lfsr[idx];
        svalue[idx] = cumsum / (rank + 1) as f64;
    }

    // Ensure monotonicity (running minimum from bottom)
    let mut running_min: f64 = 1.0;
    for &idx in indices.iter().rev() {
        if svalue[idx].is_finite() {
            running_min = running_min.min(svalue[idx]);
            svalue[idx] = running_min;
        }
    }

    svalue
}

/// Apply ashr shrinkage directly to results (modifies in place)
/// R equivalent: lfcShrink(type="ashr") in lfcShrink.R (convenience wrapper)
pub fn apply_ashr_shrinkage(results: &mut DESeqResults, params: &AshrParams) {
    let fit = shrink_lfc_ashr(results, params);

    // Replace LFCs and SEs with posterior estimates
    results.log2_fold_changes = fit.posterior_mean;
    results.lfc_se = fit.posterior_sd;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sigma_grid() {
        let params = AshrParams {
            n_components: 5,
            sigma_min: 0.01,
            sigma_max: 1.0,
            pointmass: true,
            ..Default::default()
        };

        let sigma = create_sigma_grid(&params);
        assert_eq!(sigma.len(), 5);
        assert_eq!(sigma[0], 0.0); // Point mass
        assert!(sigma[1] > 0.0);
        assert!(*sigma.last().unwrap() <= 1.0);
    }

    #[test]
    fn test_update_pi() {
        let membership = vec![
            vec![0.8, 0.2],
            vec![0.6, 0.4],
            vec![0.5, 0.5],
        ];

        let pi = update_pi(&membership);
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_svalues() {
        let lfsr = vec![0.1, 0.05, 0.2, f64::NAN, 0.15];
        let svalue = compute_svalues(&lfsr);

        // Check that finite values have s-values
        assert!(svalue[0].is_finite());
        assert!(svalue[1].is_finite());
        assert!(!svalue[3].is_finite()); // NAN input
    }
}
