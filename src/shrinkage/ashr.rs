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
    /// Grid multiplier between adjacent mixture standard deviations
    /// R: ash(gridmult = sqrt(2))
    pub grid_mult: f64,
}

impl Default for AshrParams {
    fn default() -> Self {
        Self {
            n_components: 20,
            sigma_max: 10.0,
            sigma_min: 0.001,
            max_iter: 1000,
            tol: 1e-9,
            // lfcShrink calls ash(..., method = "shrink"), which sets
            // pointmass = FALSE and prior = "uniform" (ashr R/ash.R)
            pointmass: false,
            grid_mult: std::f64::consts::SQRT_2,
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
    let sigma = create_sigma_grid_adaptive(params, &betahat, &sebetahat);
    let k = sigma.len();

    // The component likelihoods do not depend on pi, so compute them once.
    let log_lik = compute_log_likelihood_matrix(&betahat, &sebetahat, &sigma);

    // Mixture weights under a uniform prior: plain maximum likelihood. R solves the
    // same concave problem with mixSQP to high precision, so iterate to a tight
    // tolerance (with SQUAREM acceleration, as ashr's own mixEM does).
    let pi = fit_mixture_weights(&log_lik, k, params.max_iter, params.tol);

    // Compute posterior statistics
    let (posterior_mean_valid, posterior_sd_valid, lfsr_valid) =
        compute_posterior_stats(&betahat, &sebetahat, &sigma, &pi);

    // Map back to full results. Observations ash excludes (NA estimate or SE, SE 0
    // or Inf, e.g. all-zero genes) get the prior instead of a posterior, as ashr's
    // calc_pm/calc_psd do: the mixture mean (0) and sd sqrt(sum(pi * sigma^2)).
    let mix_sd = pi.iter().zip(&sigma).map(|(p, s)| p * s * s).sum::<f64>().sqrt();
    let mut posterior_mean = vec![0.0; n];
    let mut posterior_sd = vec![mix_sd; n];
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
/// R equivalent: ashr's autoselect.mixsd(data, mult = gridmult) for mode 0:
///   sigmaamin = min(se) / 10
///   sigmaamax = 2 * sqrt(max(betahat^2 - se^2))   (8 * sigmaamin if all betahat^2 <= se^2)
///   npoint    = ceiling(log2(sigmaamax / sigmaamin) / log2(mult))
///   mixsd     = mult^((-npoint):0) * sigmaamax
/// with a 0 prepended when the point mass is included.
fn create_sigma_grid_adaptive(params: &AshrParams, betahat: &[f64], sebetahat: &[f64]) -> Vec<f64> {
    if sebetahat.is_empty() {
        return create_sigma_grid(params);
    }

    let se_min = sebetahat.iter().cloned().fold(f64::INFINITY, f64::min);
    let sigma_min = se_min / 10.0;
    let max_excess = betahat
        .iter()
        .zip(sebetahat)
        .map(|(&b, &se)| b * b - se * se)
        .fold(f64::NEG_INFINITY, f64::max);
    let sigma_max = if max_excess <= 0.0 {
        8.0 * sigma_min
    } else {
        2.0 * max_excess.sqrt()
    };

    let mult = params.grid_mult;
    let npoint = ((sigma_max / sigma_min).log2() / mult.log2()).ceil().max(0.0) as i32;

    let mut sigma = Vec::with_capacity(npoint as usize + 2);
    if params.pointmass {
        sigma.push(0.0);
    }
    for e in -npoint..=0 {
        sigma.push(mult.powi(e) * sigma_max);
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

/// Maximum-likelihood mixture weights on the probability simplex
/// R equivalent: ashr::ash() with prior = "uniform", which maximises
/// sum_i log(sum_k L_ik pi_k) with mixsqp (a sequential quadratic programming solver).
///
/// The problem is concave in pi with only a handful of components, so this uses the
/// same idea as mixSQP: Newton steps on the free components (those not at zero)
/// under the constraint sum(pi) = 1, a backtracking line search that keeps pi >= 0,
/// and an active set that releases a zero component when its gradient says the
/// likelihood would increase. Plain EM converges far too slowly to reach R's
/// solution (posterior means still moved at 1e-4 after 1000 iterations).
///
/// Optimality (KKT) for this objective: with g = gradient, g'pi = n always, and at
/// the optimum g_k = n where pi_k > 0 and g_k <= n where pi_k = 0.
fn fit_mixture_weights(log_lik: &[Vec<f64>], k: usize, max_iter: usize, tol: f64) -> Vec<f64> {
    // Row-scaled likelihoods: the scaling only shifts the objective by a constant.
    let lik: Vec<Vec<f64>> = log_lik
        .par_iter()
        .map(|row| {
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            row.iter().map(|&l| (l - m).exp()).collect()
        })
        .collect();
    let n = lik.len() as f64;

    let objective = |pi: &[f64]| -> f64 {
        lik.iter()
            .map(|row| row.iter().zip(pi).map(|(l, p)| l * p).sum::<f64>().max(1e-300).ln())
            .sum()
    };
    // Gradient and Hessian of the objective
    let derivatives = |pi: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut g = vec![0.0; k];
        let mut h = vec![vec![0.0; k]; k];
        for row in &lik {
            let d: f64 = row.iter().zip(pi).map(|(l, p)| l * p).sum::<f64>().max(1e-300);
            for a in 0..k {
                let ra = row[a] / d;
                g[a] += ra;
                for b in a..k {
                    h[a][b] -= ra * row[b] / d;
                }
            }
        }
        for a in 0..k {
            for b in 0..a {
                h[a][b] = h[b][a];
            }
        }
        (g, h)
    };

    let mut pi = vec![1.0 / k as f64; k];
    let mut obj = objective(&pi);
    let mut newton_stalled = false;
    for iter in 0..max_iter {
        let (g, h) = derivatives(&pi);

        // KKT check, scaled by n so it does not depend on the number of genes
        let free: Vec<usize> = (0..k).filter(|&j| pi[j] > 0.0).collect();
        let worst_free = free.iter().map(|&j| (g[j] / n - 1.0).abs()).fold(0.0, f64::max);
        let (release, worst_bound) = (0..k)
            .filter(|&j| pi[j] == 0.0)
            .map(|j| (j, g[j] / n - 1.0))
            .fold((None, 0.0), |acc, (j, v)| if v > acc.1 { (Some(j), v) } else { acc });
        if worst_free < tol && worst_bound < tol {
            log::debug!("ashr mixture weights converged after {} iterations", iter);
            break;
        }

        // Work on the larger KKT violation. A Newton step on the free set can stall
        // at rounding level just above `tol` while a zero component still wants to
        // be released; then (or when the last Newton step made no progress) take the
        // release step instead of repeating a step that cannot improve.
        let release_first = release.is_some() && (worst_bound > worst_free || newton_stalled);
        let mut d = vec![0.0; k];
        if worst_free >= tol && !release_first {
            // Newton step on the free components: maximise g'd + d'Hd/2 subject to
            // 1'd = 0, via the KKT system [-H 1; 1' 0] [d; mu] = [g; 0]. Every free
            // component is positive, so some positive step is always feasible.
            let m = free.len();
            let ridge = 1e-10 * (0..k).map(|j| -h[j][j]).sum::<f64>().max(1e-300) / k as f64;
            let mut a = vec![vec![0.0; m + 1]; m + 1];
            let mut rhs = vec![0.0; m + 1];
            for (r, &jr) in free.iter().enumerate() {
                for (c, &jc) in free.iter().enumerate() {
                    a[r][c] = -h[jr][jc];
                }
                a[r][r] += ridge;
                a[r][m] = 1.0;
                a[m][r] = 1.0;
                rhs[r] = g[jr];
            }
            match solve_dense(a, rhs) {
                Some(step) => {
                    for (r, &j) in free.iter().enumerate() {
                        d[j] = step[r];
                    }
                }
                None => break,
            }
        } else if let Some(j) = release {
            // The free components are optimal but a zero component would increase
            // the likelihood (g_j > n): move towards that vertex (a Frank-Wolfe step,
            // slope g_j - g'pi = g_j - n > 0), with the 1-D Newton length.
            for (l, dl) in d.iter_mut().enumerate() {
                *dl = if l == j { 1.0 } else { 0.0 } - pi[l];
            }
            let slope: f64 = (0..k).map(|l| g[l] * d[l]).sum();
            let curv: f64 = (0..k).map(|a| (0..k).map(|b| d[a] * h[a][b] * d[b]).sum::<f64>()).sum();
            let len = if curv < 0.0 { (slope / -curv).min(1.0) } else { 1.0 };
            d.iter_mut().for_each(|v| *v *= len);
        }

        // Longest step keeping pi >= 0; the component that blocks it is set to zero
        // exactly so it leaves the free set. Then backtrack (Armijo).
        let mut t_max: f64 = 1.0;
        let mut blocking = None;
        for j in 0..k {
            if d[j] < 0.0 && -pi[j] / d[j] < t_max {
                t_max = -pi[j] / d[j];
                blocking = Some(j);
            }
        }
        let slope: f64 = (0..k).map(|j| g[j] * d[j]).sum();
        let mut t = t_max;
        let mut accepted = None;
        for _ in 0..60 {
            let mut cand: Vec<f64> = (0..k).map(|j| (pi[j] + t * d[j]).max(0.0)).collect();
            if t == t_max {
                if let Some(j) = blocking {
                    cand[j] = 0.0;
                }
            }
            let total: f64 = cand.iter().sum();
            cand.iter_mut().for_each(|v| *v /= total);
            let cand_obj = objective(&cand);
            if cand_obj >= obj + 1e-4 * t * slope {
                accepted = Some((cand, cand_obj));
                break;
            }
            t *= 0.5;
        }
        let was_newton = worst_free >= tol && !release_first;
        match accepted {
            Some((cand, cand_obj)) => {
                newton_stalled = was_newton && cand_obj <= obj;
                pi = cand;
                obj = cand_obj;
            }
            // No ascent at floating-point resolution: if a Newton step failed but a
            // component can still be released, try that next; otherwise stop.
            None if was_newton && release.is_some() => newton_stalled = true,
            None => break,
        }
    }
    pi
}

/// Solve a small dense linear system by Gaussian elimination with partial pivoting
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            if f != 0.0 {
                for c in col..n {
                    a[row][c] -= f * a[col][c];
                }
                b[row] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = ((row + 1)..n).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
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
    fn test_fit_mixture_weights_recovers_mixture() {
        // Two components; data generated mostly from the second one.
        let sigma = vec![0.1, 2.0];
        let betahat: Vec<f64> = (0..200).map(|i| if i % 4 == 0 { 0.05 } else { 3.0 }).collect();
        let sebetahat = vec![0.1; 200];
        let log_lik = compute_log_likelihood_matrix(&betahat, &sebetahat, &sigma);
        let pi = fit_mixture_weights(&log_lik, 2, 10_000, 1e-12);
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-10);

        // Compare with a brute-force search of the (one-dimensional) likelihood
        let loglik = |w: f64| -> f64 {
            log_lik.iter().map(|row| ((1.0 - w) * row[0].exp() + w * row[1].exp()).ln()).sum()
        };
        let best = (0..=10_000)
            .map(|i| i as f64 / 10_000.0)
            .max_by(|a, b| loglik(*a).partial_cmp(&loglik(*b)).unwrap())
            .unwrap();
        assert!((pi[1] - best).abs() < 2e-4, "pi = {:?}, grid optimum = {}", pi, best);
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
