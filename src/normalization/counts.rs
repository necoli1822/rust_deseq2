//! Normalized count transformations
//!
//! Provides FPM (Fragments Per Million) and FPKM (Fragments Per Kilobase per Million)
//! normalized count calculations.

use ndarray::{Array2, ArrayView2, Axis};
use crate::error::{DeseqError, Result};

/// Calculate Fragments Per Million (FPM)
///
/// R equivalent: fpm(object, robust=TRUE/FALSE)
///
/// robust=true (default):
///   library_sizes = sizeFactors * exp(mean(log(colSums(counts))))
///   FPM = counts * 1e6 / library_sizes
///
/// robust=false:
///   library_sizes = colSums(counts)
///   FPM = counts * 1e6 / library_sizes
///
/// # Arguments
/// * `counts` - Raw count matrix (genes x samples)
/// * `size_factors` - Size factors for each sample (used when robust=true)
/// * `robust` - If true, use size factors with geometric mean normalization
pub fn fpm(counts: ArrayView2<f64>, size_factors: &[f64], robust: bool) -> Result<Array2<f64>> {
    let (n_genes, n_samples) = counts.dim();

    if size_factors.len() != n_samples {
        return Err(DeseqError::InvalidInput {
            reason: format!(
                "Size factor length ({}) doesn't match sample count ({})",
                size_factors.len(), n_samples
            ),
        });
    }

    // Compute raw column sums of count matrix
    let col_sums: Vec<f64> = counts
        .axis_iter(Axis(1))
        .map(|col| col.sum())
        .collect();

    // Compute library sizes based on robust flag
    let library_sizes: Vec<f64> = if robust {
        // robust=true: library_sizes = sizeFactors * exp(mean(log(colSums(counts))))
        // Geometric mean of column sums
        let log_col_sums: Vec<f64> = col_sums.iter()
            .map(|&s| if s > 0.0 { s.ln() } else { 0.0 })
            .collect();
        let mean_log = log_col_sums.iter().sum::<f64>() / n_samples as f64;
        let geo_mean = mean_log.exp();

        size_factors.iter().map(|&sf| sf * geo_mean).collect()
    } else {
        // robust=false: library_sizes = colSums(counts)
        col_sums
    };

    // FPM = counts * 1e6 / library_sizes (column-wise division)
    let mut result = counts.to_owned();
    for j in 0..n_samples {
        let lib_size = library_sizes[j].max(1.0);
        for i in 0..n_genes {
            result[[i, j]] = result[[i, j]] * 1e6 / lib_size;
        }
    }

    Ok(result)
}

/// Calculate Fragments Per Kilobase per Million (FPKM)
///
/// FPKM = FPM / (gene_length_in_kb)
///
/// R equivalent: fpkm(object, robust=TRUE)
///
/// # Arguments
/// * `counts` - Raw count matrix (genes x samples)
/// * `size_factors` - Size factors for each sample
/// * `gene_lengths` - Gene lengths in base pairs
/// * `robust` - If true, use size factors with geometric mean normalization
pub fn fpkm(
    counts: ArrayView2<f64>,
    size_factors: &[f64],
    gene_lengths: &[f64],
    robust: bool,
) -> Result<Array2<f64>> {
    let (n_genes, _n_samples) = counts.dim();

    if gene_lengths.len() != n_genes {
        return Err(DeseqError::InvalidInput {
            reason: format!(
                "Gene lengths count ({}) doesn't match gene count ({})",
                gene_lengths.len(), n_genes
            ),
        });
    }

    // First calculate FPM
    let mut result = fpm(counts, size_factors, robust)?;

    // Then divide by gene length in kilobases
    for i in 0..n_genes {
        let length_kb = gene_lengths[i] / 1000.0;
        if length_kb > 0.0 {
            for j in 0..result.ncols() {
                result[[i, j]] /= length_kb;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fpm() {
        let counts = array![
            [100.0, 200.0],
            [300.0, 400.0],
        ];
        let sf = vec![1.0, 2.0];

        // robust=true: library_sizes = sf * exp(mean(log(colSums)))
        // colSums = [400, 600], log = [5.991, 6.397], mean = 6.194, exp = 490.0
        // lib_sizes = [1.0*490, 2.0*490] = [490, 980]
        let result = fpm(counts.view(), &sf, true).unwrap();
        assert!(result[[0, 0]] > 0.0);
        assert!(result[[1, 0]] > 0.0);

        // robust=false: library_sizes = colSums = [400, 600]
        let result_nr = fpm(counts.view(), &sf, false).unwrap();
        assert!(result_nr[[0, 0]] > 0.0);
    }

    #[test]
    fn test_fpkm() {
        let counts = array![
            [100.0, 200.0],
            [300.0, 400.0],
        ];
        let sf = vec![1.0, 1.0];
        let gene_lengths = vec![1000.0, 2000.0]; // 1kb, 2kb

        let result = fpkm(counts.view(), &sf, &gene_lengths, true).unwrap();
        // gene2 has 2x length of gene1, so FPKM should be half of FPM
        assert!(result[[0, 0]] > 0.0);
    }
}
