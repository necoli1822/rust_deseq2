//! Metadata structures for samples and genes

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{DeseqError, Result};

/// Sample metadata containing experimental conditions
/// R equivalent: colData(dds) in SummarizedExperiment package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMetadata {
    /// Sample identifiers
    sample_ids: Vec<String>,
    /// Experimental conditions (column name -> values for each sample)
    conditions: HashMap<String, Vec<String>>,
    /// Continuous variables (column name -> values for each sample)
    continuous: HashMap<String, Vec<f64>>,
}

impl SampleMetadata {
    /// Create new sample metadata
    pub fn new(sample_ids: Vec<String>) -> Self {
        // Check for duplicate sample IDs
        {
            let mut seen = std::collections::HashSet::new();
            for id in &sample_ids {
                if !seen.insert(id) {
                    log::warn!("Duplicate sample ID detected: '{}'. Sample IDs should be unique.", id);
                }
            }
        }
        Self {
            sample_ids,
            conditions: HashMap::new(),
            continuous: HashMap::new(),
        }
    }

    /// Add a condition column (categorical factor)
    pub fn add_condition(&mut self, name: &str, values: Vec<String>) -> Result<()> {
        if values.len() != self.sample_ids.len() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} values", self.sample_ids.len()),
                got: format!("{} values", values.len()),
            });
        }
        self.conditions.insert(name.to_string(), values);
        Ok(())
    }

    /// Add a continuous variable column
    pub fn add_continuous(&mut self, name: &str, values: Vec<f64>) -> Result<()> {
        if values.len() != self.sample_ids.len() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} values", self.sample_ids.len()),
                got: format!("{} values", values.len()),
            });
        }
        self.continuous.insert(name.to_string(), values);
        Ok(())
    }

    /// Check if a condition exists
    pub fn has_condition(&self, name: &str) -> bool {
        self.conditions.contains_key(name)
    }

    /// Check if a continuous variable exists
    pub fn has_continuous(&self, name: &str) -> bool {
        self.continuous.contains_key(name)
    }

    /// Get the value of a condition for a specific sample
    pub fn get_value(&self, condition: &str, sample_idx: usize) -> Result<String> {
        self.conditions
            .get(condition)
            .and_then(|v| v.get(sample_idx))
            .cloned()
            .ok_or_else(|| DeseqError::InvalidInput {
                reason: format!(
                    "condition '{}' or sample index {} not found",
                    condition, sample_idx
                ),
            })
    }

    /// Get the value of a continuous variable for a specific sample
    pub fn get_continuous_value(&self, name: &str, sample_idx: usize) -> Result<f64> {
        self.continuous
            .get(name)
            .and_then(|v| v.get(sample_idx))
            .copied()
            .ok_or_else(|| DeseqError::InvalidInput {
                reason: format!(
                    "continuous variable '{}' or sample index {} not found",
                    name, sample_idx
                ),
            })
    }

    /// Get unique levels for a condition (sorted)
    pub fn get_levels(&self, condition: &str) -> Result<Vec<String>> {
        self.conditions
            .get(condition)
            .map(|values| {
                let mut unique: Vec<String> = values.iter().cloned().collect();
                unique.sort();
                unique.dedup();
                unique
            })
            .ok_or_else(|| DeseqError::InvalidInput {
                reason: format!("condition '{}' not found", condition),
            })
    }

    /// Get sample IDs
    pub fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.sample_ids.len()
    }

    /// Get condition values for a specific column
    pub fn condition(&self, name: &str) -> Option<&Vec<String>> {
        self.conditions.get(name)
    }

    /// Get all condition names
    pub fn condition_names(&self) -> Vec<&str> {
        self.conditions.keys().map(|s| s.as_str()).collect()
    }

    /// Get unique levels for a condition
    pub fn levels(&self, condition_name: &str) -> Option<Vec<String>> {
        self.conditions.get(condition_name).map(|values| {
            let mut unique: Vec<String> = values.iter().cloned().collect();
            unique.sort();
            unique.dedup();
            unique
        })
    }

    /// Get sample indices for a specific condition level
    pub fn samples_with_level(&self, condition_name: &str, level: &str) -> Vec<usize> {
        self.conditions
            .get(condition_name)
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.as_str() == level)
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Subset metadata to specific samples
    pub fn subset(&self, sample_indices: &[usize]) -> Result<Self> {
        let new_ids: Vec<String> = sample_indices
            .iter()
            .map(|&i| self.sample_ids[i].clone())
            .collect();

        let mut new_meta = SampleMetadata::new(new_ids);

        for (name, values) in &self.conditions {
            let new_values: Vec<String> = sample_indices
                .iter()
                .map(|&i| values[i].clone())
                .collect();
            new_meta.add_condition(name, new_values)?;
        }

        Ok(new_meta)
    }
}

/// Gene metadata containing gene-level information
/// R equivalent: mcols(dds) / rowData(dds) in SummarizedExperiment package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneMetadata {
    /// Gene identifiers
    gene_ids: Vec<String>,
    /// Additional gene annotations
    annotations: HashMap<String, Vec<String>>,
}

impl GeneMetadata {
    /// Create new gene metadata
    pub fn new(gene_ids: Vec<String>) -> Self {
        Self {
            gene_ids,
            annotations: HashMap::new(),
        }
    }

    /// Add an annotation column
    pub fn add_annotation(&mut self, name: &str, values: Vec<String>) -> Result<()> {
        if values.len() != self.gene_ids.len() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} values", self.gene_ids.len()),
                got: format!("{} values", values.len()),
            });
        }
        self.annotations.insert(name.to_string(), values);
        Ok(())
    }

    /// Get gene IDs
    pub fn gene_ids(&self) -> &[String] {
        &self.gene_ids
    }

    /// Get number of genes
    pub fn n_genes(&self) -> usize {
        self.gene_ids.len()
    }

    /// Get annotation values
    pub fn annotation(&self, name: &str) -> Option<&Vec<String>> {
        self.annotations.get(name)
    }

    /// Subset metadata to specific genes
    pub fn subset(&self, gene_indices: &[usize]) -> Result<Self> {
        let new_ids: Vec<String> = gene_indices
            .iter()
            .map(|&i| self.gene_ids[i].clone())
            .collect();

        let mut new_meta = GeneMetadata::new(new_ids);

        for (name, values) in &self.annotations {
            let new_values: Vec<String> = gene_indices
                .iter()
                .map(|&i| values[i].clone())
                .collect();
            new_meta.add_annotation(name, new_values)?;
        }

        Ok(new_meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_metadata() {
        let mut meta = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ]);

        meta.add_condition(
            "treatment",
            vec![
                "control".to_string(),
                "control".to_string(),
                "treated".to_string(),
                "treated".to_string(),
            ],
        )
        .unwrap();

        let levels = meta.levels("treatment").unwrap();
        assert_eq!(levels, vec!["control", "treated"]);

        let control_samples = meta.samples_with_level("treatment", "control");
        assert_eq!(control_samples, vec![0, 1]);
    }
}
