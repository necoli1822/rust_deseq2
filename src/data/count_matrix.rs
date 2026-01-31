//! Count matrix representation for RNA-seq data

use std::collections::HashMap;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{DeseqError, Result};

/// Deduplicate names by appending _1, _2, etc. to duplicates (R auto-renames behavior)
fn deduplicate_names(names: Vec<String>) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut result = Vec::with_capacity(names.len());
    for name in &names {
        *seen.entry(name.clone()).or_insert(0) += 1;
    }
    // Only process if there are duplicates
    let has_dups = seen.values().any(|&c| c > 1);
    if !has_dups {
        return names;
    }
    seen.clear();
    for name in names {
        let count = seen.entry(name.clone()).or_insert(0);
        *count += 1;
        if *count == 1 {
            result.push(name);
        } else {
            let new_name = format!("{}_{}", name, *count - 1);
            log::warn!("Duplicate gene name '{}' renamed to '{}'", name, new_name);
            result.push(new_name);
        }
    }
    result
}

/// A count matrix representing RNA-seq read counts
/// R equivalent: counts(dds) / SummarizedExperiment assay in SummarizedExperiment package
/// Rows are genes, columns are samples
#[derive(Debug, Clone)]
pub struct CountMatrix {
    /// Raw count data (genes x samples)
    counts: Array2<f64>,
    /// Gene identifiers
    gene_ids: Vec<String>,
    /// Sample identifiers
    sample_ids: Vec<String>,
}

impl CountMatrix {
    /// Create a new count matrix from raw data
    pub fn new(
        counts: Array2<f64>,
        gene_ids: Vec<String>,
        sample_ids: Vec<String>,
    ) -> Result<Self> {
        let (n_genes, n_samples) = counts.dim();

        if gene_ids.len() != n_genes {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} gene IDs", n_genes),
                got: format!("{} gene IDs", gene_ids.len()),
            });
        }

        if sample_ids.len() != n_samples {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} sample IDs", n_samples),
                got: format!("{} sample IDs", sample_ids.len()),
            });
        }

        // Validate counts are non-negative finite numbers
        if counts.iter().any(|&x| x < 0.0 || x.is_nan() || x.is_infinite()) {
            return Err(DeseqError::InvalidCountMatrix {
                reason: "Counts must be non-negative finite values".to_string(),
            });
        }

        // Bug 6: R checks if all counts are zero
        // R: if (all(assay(se) == 0)) stop("all samples have 0 counts for all genes")
        if !counts.is_empty() && counts.iter().all(|&x| x == 0.0) {
            return Err(DeseqError::InvalidCountMatrix {
                reason: "All samples have 0 counts for all genes".to_string(),
            });
        }

        // Bug 7: Warn if any counts have fractional parts (R enforces integer counts)
        if counts.iter().any(|&x| x != x.round()) {
            log::warn!(
                "Some count values are not integers. DESeq2 expects integer counts. \
                 Non-integer values may affect statistical modeling."
            );
        }

        // Bug 8: Detect and rename duplicate gene names (R auto-appends .1, .2, etc.)
        let gene_ids = deduplicate_names(gene_ids);

        Ok(Self {
            counts,
            gene_ids,
            sample_ids,
        })
    }

    /// Create from integer counts
    pub fn from_integers(
        counts: Array2<u32>,
        gene_ids: Vec<String>,
        sample_ids: Vec<String>,
    ) -> Result<Self> {
        let float_counts = counts.mapv(|x| x as f64);
        Self::new(float_counts, gene_ids, sample_ids)
    }

    /// Get the number of genes
    pub fn n_genes(&self) -> usize {
        self.counts.nrows()
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.counts.ncols()
    }

    /// Get the raw counts as a view
    pub fn counts(&self) -> ArrayView2<'_, f64> {
        self.counts.view()
    }

    /// Get mutable reference to counts
    pub fn counts_mut(&mut self) -> &mut Array2<f64> {
        &mut self.counts
    }

    /// Set a specific count value
    pub fn set_count(&mut self, gene_idx: usize, sample_idx: usize, value: f64) -> Result<()> {
        if value < 0.0 || value.is_nan() || value.is_infinite() {
            return Err(DeseqError::InvalidCountMatrix {
                reason: "Count value must be a non-negative finite number".to_string(),
            });
        }
        self.counts[[gene_idx, sample_idx]] = value;
        Ok(())
    }

    /// Get gene IDs
    pub fn gene_ids(&self) -> &[String] {
        &self.gene_ids
    }

    /// Get sample IDs
    pub fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    /// Get counts for a specific gene
    pub fn gene_counts(&self, gene_idx: usize) -> ArrayView1<'_, f64> {
        self.counts.row(gene_idx)
    }

    /// Get counts for a specific sample
    pub fn sample_counts(&self, sample_idx: usize) -> ArrayView1<'_, f64> {
        self.counts.column(sample_idx)
    }

    /// Get gene index by ID
    pub fn gene_index(&self, gene_id: &str) -> Option<usize> {
        self.gene_ids.iter().position(|id| id == gene_id)
    }

    /// Get sample index by ID
    pub fn sample_index(&self, sample_id: &str) -> Option<usize> {
        self.sample_ids.iter().position(|id| id == sample_id)
    }

    /// Calculate sum of counts per sample (library size)
    pub fn library_sizes(&self) -> Vec<f64> {
        self.counts
            .axis_iter(Axis(1))
            .map(|col| col.sum())
            .collect()
    }

    /// Calculate mean counts per gene across samples
    pub fn gene_means(&self) -> Vec<f64> {
        let n = self.n_samples() as f64;
        self.counts
            .axis_iter(Axis(0))
            .map(|row| row.sum() / n)
            .collect()
    }

    /// Filter genes by minimum count threshold
    pub fn filter_low_counts(&self, min_count: f64, min_samples: usize) -> Result<Self> {
        let keep_genes: Vec<usize> = (0..self.n_genes())
            .filter(|&i| {
                let above_threshold = self.counts.row(i).iter().filter(|&&x| x >= min_count).count();
                above_threshold >= min_samples
            })
            .collect();

        if keep_genes.is_empty() {
            return Err(DeseqError::EmptyData {
                reason: "No genes passed the filtering threshold".to_string(),
            });
        }

        let new_counts = self.counts.select(Axis(0), &keep_genes);
        let new_gene_ids: Vec<String> = keep_genes.iter().map(|&i| self.gene_ids[i].clone()).collect();

        Self::new(new_counts, new_gene_ids, self.sample_ids.clone())
    }

    /// Subset to specific samples
    pub fn subset_samples(&self, sample_indices: &[usize]) -> Result<Self> {
        let new_counts = self.counts.select(Axis(1), sample_indices);
        let new_sample_ids: Vec<String> = sample_indices
            .iter()
            .map(|&i| self.sample_ids[i].clone())
            .collect();

        Self::new(new_counts, self.gene_ids.clone(), new_sample_ids)
    }

    /// Subset to specific genes
    pub fn subset_genes(&self, gene_indices: &[usize]) -> Result<Self> {
        let new_counts = self.counts.select(Axis(0), gene_indices);
        let new_gene_ids: Vec<String> = gene_indices
            .iter()
            .map(|&i| self.gene_ids[i].clone())
            .collect();

        Self::new(new_counts, new_gene_ids, self.sample_ids.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_count_matrix_creation() {
        let counts = array![[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]];
        let gene_ids = vec!["gene1".to_string(), "gene2".to_string()];
        let sample_ids = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];

        let matrix = CountMatrix::new(counts, gene_ids, sample_ids).unwrap();
        assert_eq!(matrix.n_genes(), 2);
        assert_eq!(matrix.n_samples(), 3);
    }

    #[test]
    fn test_negative_counts_rejected() {
        let counts = array![[10.0, -5.0], [5.0, 15.0]];
        let gene_ids = vec!["gene1".to_string(), "gene2".to_string()];
        let sample_ids = vec!["s1".to_string(), "s2".to_string()];

        let result = CountMatrix::new(counts, gene_ids, sample_ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_library_sizes() {
        let counts = array![[10.0, 20.0], [5.0, 15.0]];
        let gene_ids = vec!["gene1".to_string(), "gene2".to_string()];
        let sample_ids = vec!["s1".to_string(), "s2".to_string()];

        let matrix = CountMatrix::new(counts, gene_ids, sample_ids).unwrap();
        let lib_sizes = matrix.library_sizes();
        assert_eq!(lib_sizes, vec![15.0, 35.0]);
    }
}
