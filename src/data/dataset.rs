//! DESeqDataSet - Main data structure for differential expression analysis

use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

use super::{CountMatrix, GeneMetadata, SampleMetadata};
use crate::error::{DeseqError, Result};

/// Main data structure for DESeq2 analysis
/// Contains count data, metadata, and analysis results
#[derive(Debug, Clone)]
pub struct DESeqDataSet {
    /// Raw count matrix
    counts: CountMatrix,
    /// Sample metadata (experimental conditions)
    sample_metadata: SampleMetadata,
    /// Gene metadata (optional annotations)
    gene_metadata: GeneMetadata,
    /// Design formula/variable name for differential expression
    design_variable: String,

    // Extended design specification
    /// All factors in the model (including design_variable)
    factors: Vec<String>,
    /// Continuous covariates in the model
    continuous_vars: Vec<String>,
    /// Interaction terms (pairs of factors)
    interactions: Vec<(String, String)>,
    /// Reference levels for each factor
    reference_levels: HashMap<String, String>,

    // Normalization results
    /// Size factors for each sample
    size_factors: Option<Array1<f64>>,
    /// Custom normalization factors matrix (genes x samples)
    /// R equivalent: normalizationFactors(dds)
    /// When set, used instead of size factors for normalization
    normalization_factors: Option<Array2<f64>>,
    /// Normalized counts (counts / size_factors or counts / normalization_factors)
    normalized_counts: Option<Array2<f64>>,

    // Dispersion estimation results
    /// Gene-wise dispersion estimates (MLE)
    gene_dispersions: Option<Array1<f64>>,
    /// Trended dispersion estimates
    trended_dispersions: Option<Array1<f64>>,
    /// Final MAP dispersion estimates
    map_dispersions: Option<Array1<f64>>,
    /// Expected counts (mu) from GLM fitting during gene-wise dispersion estimation
    /// This is reused for MAP dispersion estimation (DESeq2 stores this in assays[["mu"]])
    mu: Option<Array2<f64>>,
    /// Dispersion function coefficients from parametric fit: (asymptDisp, extraPois)
    /// Formula: dispersion(mean) = asymptDisp + extraPois / mean
    dispersion_function: Option<(f64, f64)>,
    /// Cached dispersion prior variance (R's dispPriorVar) for refit after outlier replacement
    dispersion_prior_var: Option<f64>,
    /// Which genes were flagged as dispersion outliers during MAP estimation
    /// R equivalent: mcols(dds)$dispOutlier
    dispersion_outliers: Option<Vec<bool>>,
    /// Variance of log dispersion residuals (varLogDispEsts)
    /// Used for outlier detection threshold calculation
    var_log_disp_ests: Option<f64>,

    // GLM results
    /// Design matrix
    design_matrix: Option<Array2<f64>>,
    /// GLM coefficients (beta) for each gene
    coefficients: Option<Array2<f64>>,
    /// Standard errors of coefficients
    standard_errors: Option<Array2<f64>>,
    /// Covariance matrices of coefficients (n_genes x n_coefs x n_coefs)
    /// Required for complex contrasts
    covariances: Option<Array3<f64>>,
    /// Hat matrix diagonals for each gene and sample (used for Cook's distance)
    hat_diagonals: Option<Array2<f64>>,
    /// Convergence status for each gene
    converged: Option<Vec<bool>>,
    /// Column names for design matrix
    design_column_names: Option<Vec<String>>,
    /// Deviance for each gene: -2 * log_likelihood
    /// Used for downstream analysis (matches R's mcols(object)$deviance)
    deviance: Option<Array1<f64>>,
}

impl DESeqDataSet {
    /// Create a new DESeqDataSet
    pub fn new(
        counts: CountMatrix,
        sample_metadata: SampleMetadata,
        design_variable: &str,
    ) -> Result<Self> {
        // Validate sample IDs match
        if counts.sample_ids() != sample_metadata.sample_ids() {
            return Err(DeseqError::InvalidMetadata {
                reason: "Sample IDs in counts and metadata do not match".to_string(),
            });
        }

        // Validate design variable exists
        if sample_metadata.condition(design_variable).is_none() {
            return Err(DeseqError::InvalidDesignMatrix {
                reason: format!("Design variable '{}' not found in metadata", design_variable),
            });
        }

        // Check that the design variable has more than one level
        // R: if all samples have the same condition, the model is degenerate for DE testing
        // This is a warning rather than an error because intercept-only designs (blind mode)
        // legitimately use a single-level factor.
        if let Some(levels) = sample_metadata.levels(design_variable) {
            if levels.len() < 2 {
                log::warn!(
                    "Design variable '{}' has only one level ('{}'); \
                     differential expression testing requires at least two levels",
                    design_variable,
                    levels.first().map(|s| s.as_str()).unwrap_or(""),
                );
            }
        }

        let gene_metadata = GeneMetadata::new(counts.gene_ids().to_vec());

        Ok(Self {
            counts,
            sample_metadata,
            gene_metadata,
            design_variable: design_variable.to_string(),
            factors: Vec::new(),  // For simple designs, factors should be empty
            continuous_vars: Vec::new(),
            interactions: Vec::new(),
            reference_levels: HashMap::new(),
            size_factors: None,
            normalization_factors: None,
            normalized_counts: None,
            gene_dispersions: None,
            trended_dispersions: None,
            map_dispersions: None,
            mu: None,
            dispersion_function: None,
            dispersion_prior_var: None,
            dispersion_outliers: None,
            var_log_disp_ests: None,
            design_matrix: None,
            coefficients: None,
            standard_errors: None,
            covariances: None,
            hat_diagonals: None,
            converged: None,
            design_column_names: None,
            deviance: None,
        })
    }

    /// Create a builder for more complex designs
    pub fn builder() -> super::builder::DESeqDataSetBuilder {
        super::builder::DESeqDataSetBuilder::new()
    }

    // Getters
    pub fn counts(&self) -> &CountMatrix {
        &self.counts
    }

    pub fn sample_metadata(&self) -> &SampleMetadata {
        &self.sample_metadata
    }

    pub fn gene_metadata(&self) -> &GeneMetadata {
        &self.gene_metadata
    }

    pub fn design_variable(&self) -> &str {
        &self.design_variable
    }

    pub fn n_genes(&self) -> usize {
        self.counts.n_genes()
    }

    pub fn n_samples(&self) -> usize {
        self.counts.n_samples()
    }

    pub fn size_factors(&self) -> Option<&Array1<f64>> {
        self.size_factors.as_ref()
    }

    pub fn normalized_counts(&self) -> Option<&Array2<f64>> {
        self.normalized_counts.as_ref()
    }

    /// Get normalization factors matrix
    /// R equivalent: normalizationFactors()
    pub fn normalization_factors(&self) -> Option<&Array2<f64>> {
        self.normalization_factors.as_ref()
    }

    pub fn gene_dispersions(&self) -> Option<&Array1<f64>> {
        self.gene_dispersions.as_ref()
    }

    pub fn trended_dispersions(&self) -> Option<&Array1<f64>> {
        self.trended_dispersions.as_ref()
    }

    pub fn map_dispersions(&self) -> Option<&Array1<f64>> {
        self.map_dispersions.as_ref()
    }

    /// Get the expected counts (mu) matrix from gene-wise dispersion estimation
    pub fn mu(&self) -> Option<&Array2<f64>> {
        self.mu.as_ref()
    }

    /// Get dispersion function coefficients (asymptDisp, extraPois) from parametric fit
    pub fn dispersion_function(&self) -> Option<(f64, f64)> {
        self.dispersion_function
    }

    /// Get cached dispersion prior variance (R's dispPriorVar)
    pub fn dispersion_prior_var(&self) -> Option<f64> {
        self.dispersion_prior_var
    }

    /// Set dispersion prior variance (cached for refit after outlier replacement)
    pub fn set_dispersion_prior_var(&mut self, var: f64) {
        self.dispersion_prior_var = Some(var);
    }

    /// Get dispersion outlier flags
    pub fn dispersion_outliers(&self) -> Option<&Vec<bool>> {
        self.dispersion_outliers.as_ref()
    }

    /// Set dispersion outlier flags (from MAP estimation)
    pub fn set_dispersion_outliers(&mut self, outliers: Vec<bool>) {
        self.dispersion_outliers = Some(outliers);
    }

    /// Get variance of log dispersion estimates
    pub fn var_log_disp_ests(&self) -> Option<f64> {
        self.var_log_disp_ests
    }

    /// Set variance of log dispersion estimates (from trend fitting)
    pub fn set_var_log_disp_ests(&mut self, var: f64) {
        self.var_log_disp_ests = Some(var);
    }

    pub fn dispersions(&self) -> Option<&Array1<f64>> {
        // Return MAP dispersions if available, otherwise gene-wise
        self.map_dispersions
            .as_ref()
            .or(self.gene_dispersions.as_ref())
    }

    pub fn design_matrix(&self) -> Option<&Array2<f64>> {
        self.design_matrix.as_ref()
    }

    pub fn coefficients(&self) -> Option<&Array2<f64>> {
        self.coefficients.as_ref()
    }

    pub fn standard_errors(&self) -> Option<&Array2<f64>> {
        self.standard_errors.as_ref()
    }

    pub fn covariances(&self) -> Option<&Array3<f64>> {
        self.covariances.as_ref()
    }

    /// Get hat matrix diagonals (for Cook's distance calculation)
    pub fn hat_diagonals(&self) -> Option<&Array2<f64>> {
        self.hat_diagonals.as_ref()
    }

    /// Get all factors in the model
    pub fn factors(&self) -> &[String] {
        &self.factors
    }

    /// Get continuous covariates in the model
    pub fn continuous_vars(&self) -> &[String] {
        &self.continuous_vars
    }

    /// Get interaction terms
    pub fn interactions(&self) -> &[(String, String)] {
        &self.interactions
    }

    /// Get reference levels
    pub fn reference_levels(&self) -> &HashMap<String, String> {
        &self.reference_levels
    }

    /// Get design matrix column names
    pub fn design_column_names(&self) -> Option<&Vec<String>> {
        self.design_column_names.as_ref()
    }

    /// Check if the model has batch effects (covariates beyond the main design variable)
    pub fn has_batch_effect(&self) -> bool {
        !self.factors.is_empty()
    }

    /// Check if the model has interactions
    pub fn has_interactions(&self) -> bool {
        !self.interactions.is_empty()
    }

    /// Check if the model has continuous covariates
    pub fn has_continuous(&self) -> bool {
        !self.continuous_vars.is_empty()
    }

    // Setters for design specification (used by builder)
    pub(crate) fn set_factors(&mut self, factors: Vec<String>) {
        self.factors = factors;
    }

    pub(crate) fn set_continuous(&mut self, continuous: Vec<String>) {
        self.continuous_vars = continuous;
    }

    pub(crate) fn set_interactions(&mut self, interactions: Vec<(String, String)>) {
        self.interactions = interactions;
    }

    pub(crate) fn set_reference_levels(&mut self, levels: HashMap<String, String>) {
        self.reference_levels = levels;
    }

    pub fn set_design_column_names(&mut self, names: Vec<String>) {
        self.design_column_names = Some(names);
    }

    // Setters (for internal use during analysis)
    pub fn set_size_factors(&mut self, size_factors: Array1<f64>) -> Result<()> {
        if size_factors.len() != self.n_samples() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} size factors", self.n_samples()),
                got: format!("{}", size_factors.len()),
            });
        }
        if size_factors.iter().any(|&x| x.is_nan() || x.is_infinite() || x <= 0.0) {
            return Err(DeseqError::InvalidInput {
                reason: "size factors must be positive finite values".to_string(),
            });
        }
        self.size_factors = Some(size_factors);
        self.compute_normalized_counts();
        Ok(())
    }

    /// Set normalization factors matrix (genes x samples)
    /// R equivalent: normalizationFactors<-()
    /// When set, these are used instead of size factors for normalization.
    /// The matrix should have dimensions (n_genes, n_samples).
    pub fn set_normalization_factors(&mut self, mut nf: Array2<f64>) -> Result<()> {
        if nf.nrows() != self.n_genes() || nf.ncols() != self.n_samples() {
            return Err(DeseqError::InvalidInput {
                reason: format!(
                    "Normalization factor matrix dimensions ({}, {}) don't match data ({}, {})",
                    nf.nrows(), nf.ncols(), self.n_genes(), self.n_samples()
                ),
            });
        }
        if nf.iter().any(|&x| x.is_nan() || x.is_infinite() || x <= 0.0) {
            return Err(DeseqError::InvalidInput {
                reason: "normalization factors must be positive finite values".to_string(),
            });
        }
        // Center by row-wise geometric mean (R: nf / exp(rowMeans(log(nf))))
        let (n_genes, n_samples) = nf.dim();
        for i in 0..n_genes {
            let row_log_mean = nf.row(i).iter()
                .map(|&x| x.ln())
                .sum::<f64>() / n_samples as f64;
            let center = row_log_mean.exp();
            if center > 0.0 && center.is_finite() {
                for j in 0..n_samples {
                    nf[[i, j]] /= center;
                }
            }
        }
        self.normalization_factors = Some(nf);
        self.compute_normalized_counts();
        Ok(())
    }

    pub fn set_gene_dispersions(&mut self, dispersions: Array1<f64>) -> Result<()> {
        if dispersions.len() != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} dispersions", self.n_genes()),
                got: format!("{}", dispersions.len()),
            });
        }
        self.gene_dispersions = Some(dispersions);
        Ok(())
    }

    pub fn set_trended_dispersions(&mut self, dispersions: Array1<f64>) -> Result<()> {
        if dispersions.len() != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} dispersions", self.n_genes()),
                got: format!("{}", dispersions.len()),
            });
        }
        self.trended_dispersions = Some(dispersions);
        Ok(())
    }

    pub fn set_map_dispersions(&mut self, dispersions: Array1<f64>) -> Result<()> {
        if dispersions.len() != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} dispersions", self.n_genes()),
                got: format!("{}", dispersions.len()),
            });
        }
        self.map_dispersions = Some(dispersions);
        Ok(())
    }

    /// Set the expected counts (mu) matrix from gene-wise dispersion estimation
    /// This should be called during gene-wise dispersion estimation so it can be
    /// reused during MAP dispersion estimation (matching DESeq2's behavior)
    pub fn set_mu(&mut self, mu: Array2<f64>) -> Result<()> {
        if mu.nrows() != self.n_genes() || mu.ncols() != self.n_samples() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{}x{} mu matrix", self.n_genes(), self.n_samples()),
                got: format!("{}x{}", mu.nrows(), mu.ncols()),
            });
        }
        self.mu = Some(mu);
        Ok(())
    }

    /// Set dispersion function coefficients from parametric trend fit
    pub fn set_dispersion_function(&mut self, asympt_disp: f64, extra_pois: f64) {
        self.dispersion_function = Some((asympt_disp, extra_pois));
    }

    pub fn set_design_matrix(&mut self, matrix: Array2<f64>) -> Result<()> {
        if matrix.nrows() != self.n_samples() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} rows in design matrix", self.n_samples()),
                got: format!("{}", matrix.nrows()),
            });
        }
        self.design_matrix = Some(matrix);
        Ok(())
    }

    pub fn set_coefficients(&mut self, coefficients: Array2<f64>) -> Result<()> {
        if coefficients.nrows() != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} genes", self.n_genes()),
                got: format!("{}", coefficients.nrows()),
            });
        }
        self.coefficients = Some(coefficients);
        Ok(())
    }

    pub fn set_standard_errors(&mut self, se: Array2<f64>) -> Result<()> {
        if se.nrows() != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} genes", self.n_genes()),
                got: format!("{}", se.nrows()),
            });
        }
        self.standard_errors = Some(se);
        Ok(())
    }

    pub fn set_covariances(&mut self, cov: Array3<f64>) -> Result<()> {
        if cov.shape()[0] != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} genes", self.n_genes()),
                got: format!("{}", cov.shape()[0]),
            });
        }
        self.covariances = Some(cov);
        Ok(())
    }

    pub fn set_hat_diagonals(&mut self, h: Array2<f64>) -> Result<()> {
        if h.nrows() != self.n_genes() || h.ncols() != self.n_samples() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{}x{} hat diagonals", self.n_genes(), self.n_samples()),
                got: format!("{}x{}", h.nrows(), h.ncols()),
            });
        }
        self.hat_diagonals = Some(h);
        Ok(())
    }

    pub fn set_converged(&mut self, converged: Vec<bool>) {
        self.converged = Some(converged);
    }

    pub fn converged(&self) -> Option<&Vec<bool>> {
        self.converged.as_ref()
    }

    /// Get deviance for each gene (-2 * log_likelihood)
    pub fn deviance(&self) -> Option<&Array1<f64>> {
        self.deviance.as_ref()
    }

    /// Set deviance for each gene
    pub fn set_deviance(&mut self, dev: Array1<f64>) -> Result<()> {
        if dev.len() != self.n_genes() {
            return Err(DeseqError::DimensionMismatch {
                expected: format!("{} genes", self.n_genes()),
                got: format!("{}", dev.len()),
            });
        }
        self.deviance = Some(dev);
        Ok(())
    }

    /// Compute normalized counts from raw counts and size factors or normalization factors
    fn compute_normalized_counts(&mut self) {
        let raw = self.counts.counts();
        let n_genes = self.n_genes();
        let n_samples = self.n_samples();

        // If normalization_factors are set, use them instead of size_factors
        if let Some(nf) = &self.normalization_factors {
            let mut normalized = raw.to_owned();
            for i in 0..n_genes {
                for j in 0..n_samples {
                    normalized[[i, j]] /= nf[[i, j]].max(1e-10);
                }
            }
            self.normalized_counts = Some(normalized);
        } else if let Some(sf) = &self.size_factors {
            let mut normalized = raw.to_owned();

            for (j, &s) in sf.iter().enumerate() {
                if s > 0.0 {
                    for i in 0..n_genes {
                        normalized[[i, j]] /= s;
                    }
                }
            }

            self.normalized_counts = Some(normalized);
        }
    }

    /// Check if size factors have been estimated
    pub fn has_size_factors(&self) -> bool {
        self.size_factors.is_some()
    }

    /// Check if dispersions have been estimated
    pub fn has_dispersions(&self) -> bool {
        self.gene_dispersions.is_some() || self.map_dispersions.is_some()
    }

    /// Check if GLM has been fitted
    pub fn has_glm_fit(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Replace a count value for a specific gene and sample
    /// Used for outlier replacement
    pub fn replace_count(&mut self, gene_idx: usize, sample_idx: usize, new_value: f64) -> Result<()> {
        self.counts.set_count(gene_idx, sample_idx, new_value)?;
        // Recompute normalized counts if size factors are available
        if self.size_factors.is_some() {
            self.compute_normalized_counts();
        }
        Ok(())
    }

    /// Get mutable access to counts for batch replacement
    pub fn counts_mut(&mut self) -> &mut CountMatrix {
        &mut self.counts
    }

    /// Get the reference level for the design variable
    pub fn reference_level(&self) -> Option<String> {
        self.sample_metadata
            .levels(&self.design_variable)
            .and_then(|levels| levels.first().cloned())
    }

    /// Get all levels for the design variable
    pub fn levels(&self) -> Option<Vec<String>> {
        self.sample_metadata.levels(&self.design_variable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn create_test_dataset() -> DESeqDataSet {
        let counts = CountMatrix::new(
            array![
                [100.0, 200.0, 50.0, 150.0],
                [500.0, 600.0, 400.0, 550.0],
                [10.0, 20.0, 15.0, 25.0]
            ],
            vec!["gene1".to_string(), "gene2".to_string(), "gene3".to_string()],
            vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()],
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

        DESeqDataSet::new(counts, metadata, "treatment").unwrap()
    }

    #[test]
    fn test_dataset_creation() {
        let dds = create_test_dataset();
        assert_eq!(dds.n_genes(), 3);
        assert_eq!(dds.n_samples(), 4);
        assert_eq!(dds.design_variable(), "treatment");
    }

    #[test]
    fn test_size_factor_setting() {
        let mut dds = create_test_dataset();
        let sf = Array1::from_vec(vec![1.0, 1.5, 0.8, 1.2]);
        dds.set_size_factors(sf).unwrap();

        assert!(dds.has_size_factors());
        assert!(dds.normalized_counts().is_some());
    }

    #[test]
    fn test_normalization_factors_setting() {
        let mut dds = create_test_dataset();
        // Create a 3x4 normalization factor matrix (genes x samples)
        let nf = array![
            [1.0, 1.5, 0.8, 1.2],
            [1.1, 1.4, 0.9, 1.3],
            [0.9, 1.6, 0.7, 1.1],
        ];
        dds.set_normalization_factors(nf.clone()).unwrap();

        assert!(dds.normalization_factors().is_some());
        assert!(dds.normalized_counts().is_some());

        // After centering, each row's geometric mean should be ~1.0
        let stored_nf = dds.normalization_factors().unwrap();
        for i in 0..dds.n_genes() {
            let row_geo_mean = (stored_nf.row(i).iter()
                .map(|&x| x.ln())
                .sum::<f64>() / dds.n_samples() as f64).exp();
            assert!((row_geo_mean - 1.0).abs() < 1e-9,
                "Row {} geometric mean should be ~1.0, got {}", i, row_geo_mean);
        }

        // Verify normalized counts are computed using the CENTERED normalization factors
        let normalized = dds.normalized_counts().unwrap();
        let counts = dds.counts().counts();
        for i in 0..dds.n_genes() {
            for j in 0..dds.n_samples() {
                let expected = counts[[i, j]] / stored_nf[[i, j]];
                assert!((normalized[[i, j]] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_normalization_factors_override_size_factors() {
        let mut dds = create_test_dataset();

        // First set size factors
        let sf = Array1::from_vec(vec![1.0, 1.5, 0.8, 1.2]);
        dds.set_size_factors(sf).unwrap();
        let normalized_with_sf = dds.normalized_counts().unwrap().clone();

        // Then set normalization factors - should override
        // Use non-uniform rows so centering produces distinct values
        let nf = array![
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [0.5, 1.0, 1.5, 2.0],
        ];
        dds.set_normalization_factors(nf).unwrap();
        let normalized_with_nf = dds.normalized_counts().unwrap();

        // Results should be different since normalization factors override size factors
        assert_ne!(normalized_with_sf[[0, 0]], normalized_with_nf[[0, 0]]);
    }
}
