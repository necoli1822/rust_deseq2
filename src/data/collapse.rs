//! Collapse technical replicates by summing counts
//!
//! R equivalent: collapseReplicates()

use std::collections::HashMap;
use ndarray::Array2;

use super::{CountMatrix, SampleMetadata, DESeqDataSet};
use crate::error::{DeseqError, Result};

/// Collapse technical replicates by summing counts
///
/// R equivalent: collapseReplicates(object, groupby, run, renameCols)
///
/// Groups samples by the `groupby` column in metadata and sums their counts.
/// The resulting dataset has one sample per unique group value.
///
/// # Arguments
/// * `dds` - DESeqDataSet with technical replicates
/// * `groupby` - Column name in metadata that identifies biological replicates
///   (samples with the same value are summed)
/// * `rename_cols` - If true (default), use group names as new column names.
///   If false, keep the original column name of the first sample in each group.
///
/// # Note
/// The `run` parameter from R's collapseReplicates is not yet implemented.
/// R uses it to create column names like "group_run1_run2".
///
/// # Returns
/// A new DESeqDataSet with collapsed counts
pub fn collapse_replicates(
    dds: &DESeqDataSet,
    groupby: &str,
    rename_cols: bool,
) -> Result<DESeqDataSet> {
    let metadata = dds.sample_metadata();

    if !metadata.has_condition(groupby) {
        return Err(DeseqError::InvalidInput {
            reason: format!("Column '{}' not found in metadata", groupby),
        });
    }

    let n_genes = dds.n_genes();
    let n_samples = dds.n_samples();
    let counts = dds.counts().counts();

    // Group samples by the groupby column
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

    for i in 0..n_samples {
        let group = metadata.get_value(groupby, i)
            .map_err(|e| DeseqError::InvalidInput { reason: e.to_string() })?;
        groups.entry(group).or_default().push(i);
    }

    // Sort groups alphabetically to match R's factor level ordering
    let mut group_order: Vec<String> = groups.keys().cloned().collect();
    group_order.sort();

    let n_new_samples = group_order.len();

    // Sum counts within groups
    let mut new_counts = Array2::zeros((n_genes, n_new_samples));
    for (new_j, group_name) in group_order.iter().enumerate() {
        let sample_indices = &groups[group_name];
        for &old_j in sample_indices {
            for i in 0..n_genes {
                new_counts[[i, new_j]] += counts[[i, old_j]];
            }
        }
    }

    // Create new sample IDs
    let new_sample_ids: Vec<String> = if rename_cols {
        group_order.clone()
    } else {
        // Keep the original column name of the first sample in each group
        group_order.iter().map(|g| {
            let first_idx = groups[g][0];
            dds.counts().sample_ids()[first_idx].clone()
        }).collect()
    };
    let gene_ids = dds.counts().gene_ids().to_vec();

    // Create new count matrix
    let new_count_matrix = CountMatrix::new(new_counts.clone(), gene_ids, new_sample_ids.clone())?;

    // Create new metadata: take metadata from the first sample in each group
    let mut new_metadata = SampleMetadata::new(new_sample_ids);

    // Copy all condition columns
    for condition_name in metadata.condition_names() {
        let values: Vec<String> = group_order.iter().map(|group_name| {
            let first_sample = groups[group_name][0];
            metadata.get_value(&condition_name, first_sample).unwrap_or_default()
        }).collect();
        new_metadata.add_condition(&condition_name, values)?;
    }

    // Verify total count is preserved (R's collapseReplicates check)
    let original_total: f64 = counts.sum();
    let collapsed_total: f64 = new_counts.sum();
    if (original_total - collapsed_total).abs() > 1e-6 {
        return Err(DeseqError::InvalidInput {
            reason: format!(
                "Total count mismatch after collapsing: original={}, collapsed={}",
                original_total, collapsed_total
            ),
        });
    }

    // Create new dataset
    DESeqDataSet::new(new_count_matrix, new_metadata, dds.design_variable())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_collapse_replicates() {
        // Create test data with technical replicates
        let counts = CountMatrix::new(
            array![
                [10.0, 20.0, 15.0, 25.0, 30.0, 35.0],
                [100.0, 200.0, 150.0, 250.0, 300.0, 350.0],
                [5.0, 10.0, 8.0, 12.0, 15.0, 18.0]
            ],
            vec!["gene1".to_string(), "gene2".to_string(), "gene3".to_string()],
            vec![
                "s1_rep1".to_string(),
                "s1_rep2".to_string(),
                "s2_rep1".to_string(),
                "s2_rep2".to_string(),
                "s3_rep1".to_string(),
                "s3_rep2".to_string(),
            ],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec![
            "s1_rep1".to_string(),
            "s1_rep2".to_string(),
            "s2_rep1".to_string(),
            "s2_rep2".to_string(),
            "s3_rep1".to_string(),
            "s3_rep2".to_string(),
        ]);

        // Add sample identifier (for grouping)
        metadata
            .add_condition(
                "sample",
                vec![
                    "s1".to_string(),
                    "s1".to_string(),
                    "s2".to_string(),
                    "s2".to_string(),
                    "s3".to_string(),
                    "s3".to_string(),
                ],
            )
            .unwrap();

        // Add treatment condition
        metadata
            .add_condition(
                "treatment",
                vec![
                    "control".to_string(),
                    "control".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                    "control".to_string(),
                    "control".to_string(),
                ],
            )
            .unwrap();

        let dds = DESeqDataSet::new(counts, metadata, "treatment").unwrap();

        // Collapse by sample (rename_cols=true is default R behavior)
        let collapsed = collapse_replicates(&dds, "sample", true).unwrap();

        // Should have 3 samples now (s1, s2, s3)
        assert_eq!(collapsed.n_samples(), 3);
        assert_eq!(collapsed.n_genes(), 3);

        // Check that counts are summed correctly
        // s1: 10 + 20 = 30, s2: 15 + 25 = 40, s3: 30 + 35 = 65
        let collapsed_counts = collapsed.counts().counts();
        assert_eq!(collapsed_counts[[0, 0]], 30.0); // gene1, s1
        assert_eq!(collapsed_counts[[0, 1]], 40.0); // gene1, s2
        assert_eq!(collapsed_counts[[0, 2]], 65.0); // gene1, s3

        // gene2: s1: 100 + 200 = 300, s2: 150 + 250 = 400, s3: 300 + 350 = 650
        assert_eq!(collapsed_counts[[1, 0]], 300.0);
        assert_eq!(collapsed_counts[[1, 1]], 400.0);
        assert_eq!(collapsed_counts[[1, 2]], 650.0);

        // Check that metadata is preserved
        let collapsed_meta = collapsed.sample_metadata();
        assert_eq!(
            collapsed_meta.get_value("treatment", 0).unwrap(),
            "control"
        );
        assert_eq!(
            collapsed_meta.get_value("treatment", 1).unwrap(),
            "treated"
        );
        assert_eq!(
            collapsed_meta.get_value("treatment", 2).unwrap(),
            "control"
        );
    }

    #[test]
    fn test_collapse_replicates_invalid_column() {
        let counts = CountMatrix::new(
            array![[10.0, 20.0], [100.0, 200.0]],
            vec!["gene1".to_string(), "gene2".to_string()],
            vec!["s1".to_string(), "s2".to_string()],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec!["s1".to_string(), "s2".to_string()]);
        metadata
            .add_condition(
                "treatment",
                vec!["control".to_string(), "treated".to_string()],
            )
            .unwrap();
        let dds = DESeqDataSet::new(counts, metadata, "treatment").unwrap();

        // Try to collapse by non-existent column
        let result = collapse_replicates(&dds, "nonexistent", true);
        assert!(result.is_err());
    }
}
