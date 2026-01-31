//! Builder pattern for DESeqDataSet
//!
//! Provides a fluent API for constructing DESeqDataSet with complex designs.

use ndarray::Array2;
use std::collections::HashMap;

use super::{CountMatrix, DESeqDataSet, SampleMetadata};
use crate::error::{DeseqError, Result};

/// Builder for DESeqDataSet
/// No direct R equivalent -- Rust-specific builder pattern
///
/// # Example
///
/// ```ignore
/// let dds = DESeqDataSetBuilder::new()
///     .counts(count_matrix)
///     .metadata(sample_metadata)
///     .main_effect("treatment")
///     .covariate("batch")
///     .interaction(&["treatment", "batch"])
///     .build()?;
/// ```
#[derive(Default)]
pub struct DESeqDataSetBuilder {
    counts: Option<CountMatrix>,
    metadata: Option<SampleMetadata>,
    main_effect: Option<String>,
    factors: Vec<String>,
    continuous: Vec<String>,
    interactions: Vec<(String, String)>,
    reference_levels: HashMap<String, String>,
}

impl DESeqDataSetBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the count matrix (required)
    pub fn counts(mut self, counts: CountMatrix) -> Self {
        self.counts = Some(counts);
        self
    }

    /// Set the sample metadata (required)
    pub fn metadata(mut self, metadata: SampleMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the main effect variable for differential expression testing (required)
    ///
    /// This is the primary variable of interest (e.g., "treatment", "condition")
    /// Note: The main effect is tracked separately from factors (covariates)
    pub fn main_effect(mut self, variable: &str) -> Self {
        self.main_effect = Some(variable.to_string());
        self
    }

    /// Add a categorical factor (covariate) to the model
    ///
    /// Use this for batch effects or other categorical covariates
    pub fn factor(mut self, name: &str) -> Self {
        if !self.factors.contains(&name.to_string()) {
            self.factors.push(name.to_string());
        }
        self
    }

    /// Add a continuous covariate to the model
    ///
    /// Use this for variables like age, weight, or other numeric covariates
    pub fn continuous(mut self, name: &str) -> Self {
        if !self.continuous.contains(&name.to_string()) {
            self.continuous.push(name.to_string());
        }
        self
    }

    /// Add an interaction term between two factors
    pub fn interaction(mut self, factor1: &str, factor2: &str) -> Self {
        let pair = (factor1.to_string(), factor2.to_string());
        if !self.interactions.contains(&pair) {
            self.interactions.push(pair);
        }
        self
    }

    /// Set the reference level for a factor
    ///
    /// The reference level is the baseline for comparisons (default: alphabetically first)
    pub fn reference_level(mut self, factor: &str, level: &str) -> Self {
        self.reference_levels
            .insert(factor.to_string(), level.to_string());
        self
    }

    /// Build the DESeqDataSet
    ///
    /// Returns an error if required fields are missing or validation fails
    pub fn build(self) -> Result<DESeqDataSet> {
        // Validate required fields
        let counts = self.counts.ok_or_else(|| DeseqError::InvalidInput {
            reason: "counts is required".to_string(),
        })?;

        let metadata = self.metadata.ok_or_else(|| DeseqError::InvalidInput {
            reason: "metadata is required".to_string(),
        })?;

        let main_effect = self.main_effect.ok_or_else(|| DeseqError::InvalidInput {
            reason: "main_effect is required".to_string(),
        })?;

        // Validate that main_effect exists in metadata
        if !metadata.has_condition(&main_effect) {
            return Err(DeseqError::InvalidInput {
                reason: format!("main_effect '{}' not found in metadata", main_effect),
            });
        }

        // Validate all factors exist in metadata
        for factor in &self.factors {
            if !metadata.has_condition(factor) {
                return Err(DeseqError::InvalidInput {
                    reason: format!("factor '{}' not found in metadata", factor),
                });
            }
        }

        // Validate continuous variables exist in metadata
        for cont in &self.continuous {
            if !metadata.has_continuous(cont) {
                return Err(DeseqError::InvalidInput {
                    reason: format!("continuous variable '{}' not found in metadata", cont),
                });
            }
        }

        // Validate reference levels actually exist in their factor's values
        for (factor, level) in &self.reference_levels {
            let factor_name = if factor == &main_effect {
                &main_effect
            } else if self.factors.contains(factor) {
                factor
            } else {
                return Err(DeseqError::InvalidInput {
                    reason: format!(
                        "Reference level set for unknown factor '{}'. \
                         Factor must be the main effect or a covariate.",
                        factor
                    ),
                });
            };
            if let Some(levels) = metadata.levels(factor_name) {
                if !levels.contains(level) {
                    return Err(DeseqError::InvalidInput {
                        reason: format!(
                            "Reference level '{}' does not exist in factor '{}'. \
                             Available levels: {:?}",
                            level, factor_name, levels
                        ),
                    });
                }
            }
        }

        // Create base DESeqDataSet
        let mut dds = DESeqDataSet::new(counts, metadata, &main_effect)?;

        // Store additional design information
        dds.set_factors(self.factors);
        dds.set_continuous(self.continuous);
        dds.set_interactions(self.interactions);
        dds.set_reference_levels(self.reference_levels);

        Ok(dds)
    }
}

/// Extended design matrix builder
/// No direct R equivalent -- Rust-specific builder pattern
pub struct DesignMatrixBuilder<'a> {
    metadata: &'a SampleMetadata,
    factors: Vec<String>,
    continuous: Vec<String>,
    interactions: Vec<(String, String)>,
    reference_levels: HashMap<String, String>,
}

impl<'a> DesignMatrixBuilder<'a> {
    /// Create a new design matrix builder
    pub fn new(metadata: &'a SampleMetadata) -> Self {
        Self {
            metadata,
            factors: Vec::new(),
            continuous: Vec::new(),
            interactions: Vec::new(),
            reference_levels: HashMap::new(),
        }
    }

    /// Add a categorical factor
    pub fn add_factor(mut self, name: &str) -> Self {
        self.factors.push(name.to_string());
        self
    }

    /// Add a continuous variable
    pub fn add_continuous(mut self, name: &str) -> Self {
        self.continuous.push(name.to_string());
        self
    }

    /// Add an interaction term
    pub fn add_interaction(mut self, factor1: &str, factor2: &str) -> Self {
        self.interactions
            .push((factor1.to_string(), factor2.to_string()));
        self
    }

    /// Set reference level for a factor
    pub fn set_reference(mut self, factor: &str, level: &str) -> Self {
        self.reference_levels
            .insert(factor.to_string(), level.to_string());
        self
    }

    /// Build the design matrix
    pub fn build(self) -> Result<(Array2<f64>, DesignMatrixInfo)> {
        let n_samples = self.metadata.sample_ids().len();

        // Calculate total number of columns
        let mut n_cols = 1; // intercept
        let mut column_names = vec!["Intercept".to_string()];
        let mut factor_info: HashMap<String, Vec<String>> = HashMap::new();

        // Add factor columns (dummy encoding)
        for factor in &self.factors {
            let levels = self.metadata.get_levels(factor)?;
            let ref_level = self
                .reference_levels
                .get(factor)
                .cloned()
                .unwrap_or_else(|| levels[0].clone());

            let non_ref_levels: Vec<String> = levels
                .into_iter()
                .filter(|l| l != &ref_level)
                .collect();

            for level in &non_ref_levels {
                column_names.push(format!("{}_{}", factor, level));
            }
            n_cols += non_ref_levels.len();
            factor_info.insert(factor.clone(), non_ref_levels);
        }

        // Add continuous variable columns
        for cont in &self.continuous {
            column_names.push(cont.clone());
            n_cols += 1;
        }

        // Add interaction columns
        for (f1, f2) in &self.interactions {
            let levels1 = factor_info.get(f1).cloned().unwrap_or_default();
            let levels2 = factor_info.get(f2).cloned().unwrap_or_default();

            for l1 in &levels1 {
                for l2 in &levels2 {
                    column_names.push(format!("{}_{}_x_{}_{}", f1, l1, f2, l2));
                    n_cols += 1;
                }
            }
        }

        // Build the matrix
        let mut design = Array2::zeros((n_samples, n_cols));

        for i in 0..n_samples {
            let mut col = 0;

            // Intercept
            design[[i, col]] = 1.0;
            col += 1;

            // Factor columns
            for factor in &self.factors {
                let sample_level = self.metadata.get_value(factor, i)?;
                let non_ref_levels = factor_info.get(factor).unwrap();

                for level in non_ref_levels {
                    design[[i, col]] = if sample_level == *level { 1.0 } else { 0.0 };
                    col += 1;
                }
            }

            // Continuous columns
            for cont in &self.continuous {
                design[[i, col]] = self.metadata.get_continuous_value(cont, i)?;
                col += 1;
            }

            // Interaction columns
            for (f1, f2) in &self.interactions {
                let sample_l1 = self.metadata.get_value(f1, i)?;
                let sample_l2 = self.metadata.get_value(f2, i)?;
                let levels1 = factor_info.get(f1).unwrap();
                let levels2 = factor_info.get(f2).unwrap();

                for l1 in levels1 {
                    for l2 in levels2 {
                        let val = if sample_l1 == *l1 && sample_l2 == *l2 {
                            1.0
                        } else {
                            0.0
                        };
                        design[[i, col]] = val;
                        col += 1;
                    }
                }
            }
        }

        let info = DesignMatrixInfo {
            column_names,
            factors: self.factors,
            continuous: self.continuous,
            interactions: self.interactions,
            reference_levels: self.reference_levels,
        };

        Ok((design, info))
    }
}

/// Information about the design matrix structure
/// No direct R equivalent -- Rust-specific design matrix metadata
#[derive(Debug, Clone)]
pub struct DesignMatrixInfo {
    pub column_names: Vec<String>,
    pub factors: Vec<String>,
    pub continuous: Vec<String>,
    pub interactions: Vec<(String, String)>,
    pub reference_levels: HashMap<String, String>,
}

impl DesignMatrixInfo {
    /// Get the column index for a specific coefficient
    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.column_names.iter().position(|n| n == name)
    }

    /// Get column index for a factor level comparison
    pub fn get_factor_index(&self, factor: &str, level: &str) -> Option<usize> {
        let col_name = format!("{}_{}", factor, level);
        self.get_column_index(&col_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let counts = CountMatrix::new(
            ndarray::array![[100.0, 200.0], [150.0, 250.0]],
            vec!["gene1".to_string(), "gene2".to_string()],
            vec!["s1".to_string(), "s2".to_string()],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec!["s1".to_string(), "s2".to_string()]);
        metadata
            .add_condition("treatment", vec!["control".to_string(), "treated".to_string()])
            .unwrap();

        let dds = DESeqDataSetBuilder::new()
            .counts(counts)
            .metadata(metadata)
            .main_effect("treatment")
            .build();

        assert!(dds.is_ok());
    }

    #[test]
    fn test_builder_missing_counts() {
        let metadata = SampleMetadata::new(vec!["s1".to_string(), "s2".to_string()]);

        let result = DESeqDataSetBuilder::new()
            .metadata(metadata)
            .main_effect("treatment")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_batch() {
        let counts = CountMatrix::new(
            ndarray::array![[100.0, 200.0, 150.0, 250.0], [110.0, 210.0, 160.0, 260.0]],
            vec!["gene1".to_string(), "gene2".to_string()],
            vec![
                "s1".to_string(),
                "s2".to_string(),
                "s3".to_string(),
                "s4".to_string(),
            ],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ]);
        metadata
            .add_condition(
                "treatment",
                vec![
                    "control".to_string(),
                    "control".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                ],
            )
            .unwrap();
        metadata
            .add_condition(
                "batch",
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                ],
            )
            .unwrap();

        let dds = DESeqDataSetBuilder::new()
            .counts(counts)
            .metadata(metadata)
            .main_effect("treatment")
            .factor("batch")
            .build();

        assert!(dds.is_ok());
        let dds = dds.unwrap();
        assert!(dds.factors().contains(&"batch".to_string()));
    }
}
