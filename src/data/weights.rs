//! Observation weights (R: `assays(dds)[["weights"]]`)
//!
//! DESeq2 accepts a genes x samples matrix of non-negative observation weights.
//! The raw weights enter `getBaseMeansAndVariances()`; every model fit uses the
//! weights normalised by their row maximum (`getAndCheckWeights()`).

use ndarray::Array2;

/// R: `weightThreshold = 1e-2` of `estimateDispersionsGeneEst()` /
/// `estimateDispersionsMAP()` / `getAndCheckWeights()`. Samples with a
/// normalised weight at or below it are left out of the Cox-Reid term.
pub const WEIGHT_THRESHOLD: f64 = 1e-2;

/// R: `weights <- pmax(weights, 1e-6)` in `estimateDispersionsGeneEst()`
/// (only the gene-wise dispersion fit floors the weights).
pub const GENE_EST_MIN_WEIGHT: f64 = 1e-6;

/// Weights normalised by their row maximum.
/// R: `weights <- weights / apply(weights, 1, max)` in `getAndCheckWeights()`.
/// A row whose weights are all zero becomes NaN (0/0), as in R; such rows fail
/// [`weights_ok`] and are never fitted.
pub fn normalize_weights(raw: &Array2<f64>) -> Array2<f64> {
    let mut w = raw.to_owned();
    for mut row in w.rows_mut() {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|x| x / max);
    }
    w
}

/// Rank of a matrix as R's `qr(x)$rank` (LINPACK `dqrdc2`, `tol = 1e-07`)
/// determines it: a column is dropped as negligible when the norm of its part
/// orthogonal to the columns already accepted falls below `tol` times its
/// original norm.
pub fn qr_rank(x: &Array2<f64>) -> usize {
    const TOL: f64 = 1e-7;
    let (n, p) = x.dim();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for j in 0..p {
        let mut v: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        let orig = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if !orig.is_finite() {
            continue;
        }
        // modified Gram-Schmidt, twice for stability
        for _ in 0..2 {
            for q in &basis {
                let d: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi -= d * qi;
                }
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if orig > 0.0 && norm >= TOL * orig {
            basis.push(v.iter().map(|a| a / norm).collect());
        }
    }
    basis.len()
}

/// Per-gene check that the weights still allow parameter estimation.
/// R: the `weights.ok` test of `getAndCheckWeights()` (core.R) for a full-rank
/// model matrix:
///
/// ```r
/// test1 <- qr(weights[i,] * modelMatrix)$rank == m
/// mm.sub <- modelMatrix[weights[i,] > weightThreshold,,drop=FALSE]
/// mm.sub <- mm.sub[,colSums(abs(mm.sub)) > 0,drop=FALSE]
/// test2 <- qr(mm.sub)$rank == ncol(mm.sub)
/// ```
///
/// `weights` are the normalised weights. Genes that fail are treated by R as
/// if all their counts were zero (`mcols(dds)$allZero`, flagged in
/// `mcols(dds)$weightsFail`), so every result for them is NA.
pub fn weights_ok(weights: &Array2<f64>, design: &Array2<f64>) -> Vec<bool> {
    let (n, m) = design.dim();
    (0..weights.nrows())
        .map(|i| {
            let w = weights.row(i);
            if w.iter().any(|x| !x.is_finite()) {
                return false;
            }
            let scaled = Array2::from_shape_fn((n, m), |(r, c)| w[r] * design[[r, c]]);
            let test1 = qr_rank(&scaled) == m;
            let sub = cr_design(design, &w.to_vec(), WEIGHT_THRESHOLD).0;
            let test2 = qr_rank(&sub) == sub.ncols();
            test1 && test2
        })
        .collect()
}

/// The model matrix of the Cox-Reid term under observation weights, with the
/// indices of the samples it keeps.
/// R (DESeq2.cpp, `log_posterior` / `dlog_posterior`):
///
/// ```cpp
/// x = x.rows(find(wts > weightThreshold));
/// x = x.cols(find(sum(abs(x)) > 0.0));
/// ```
pub fn cr_design(design: &Array2<f64>, weights: &[f64], threshold: f64) -> (Array2<f64>, Vec<usize>) {
    let rows: Vec<usize> = (0..design.nrows()).filter(|&i| weights[i] > threshold).collect();
    let cols: Vec<usize> = (0..design.ncols())
        .filter(|&j| rows.iter().map(|&i| design[[i, j]].abs()).sum::<f64>() > 0.0)
        .collect();
    let sub = Array2::from_shape_fn((rows.len(), cols.len()), |(r, c)| design[[rows[r], cols[c]]]);
    (sub, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rank_and_checks() {
        let x = array![[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]];
        assert_eq!(qr_rank(&x), 2);
        assert_eq!(qr_rank(&array![[1.0, 2.0], [1.0, 2.0]]), 1);
        let w = array![
            [1.0, 0.5, 0.5, 1.0],   // fine
            [1.0, 1.0, 0.0, 0.0],   // second group has zero weight: rank deficient
            [1.0, 1.0, 0.005, 0.5], // one low weight: still estimable
            [1.0, 1.0, 0.005, 0.005], // second group only below threshold: CR design loses a column but stays full rank
        ];
        assert_eq!(weights_ok(&w, &x), vec![true, false, true, true]);
        let (sub, rows) = cr_design(&x, &[1.0, 1.0, 0.005, 0.005], WEIGHT_THRESHOLD);
        assert_eq!(rows, vec![0, 1]);
        assert_eq!(sub.dim(), (2, 1));
    }

    #[test]
    fn normalize_by_row_max() {
        let w = normalize_weights(&array![[0.5, 1.0, 0.25], [2.0, 4.0, 4.0]]);
        assert_eq!(w, array![[0.5, 1.0, 0.25], [0.5, 1.0, 1.0]]);
    }
}
