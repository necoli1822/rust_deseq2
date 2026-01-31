//! DESeq2 results structure

use serde::{Deserialize, Serialize};

/// Results from DESeq2 differential expression analysis
/// R equivalent: DESeqResults S4 class in AllClasses.R
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DESeqResults {
    /// Gene identifiers
    pub gene_ids: Vec<String>,
    /// Base mean expression across all samples
    pub base_means: Vec<f64>,
    /// Base variance of normalized counts across all samples
    /// R equivalent: mcols(dds)$baseVar
    pub base_vars: Vec<f64>,
    /// Log2 fold change
    pub log2_fold_changes: Vec<f64>,
    /// Standard error of log2 fold change
    pub lfc_se: Vec<f64>,
    /// Test statistic (Wald z-statistic or LRT chi-square)
    pub stat: Vec<f64>,
    /// Raw p-values
    pub pvalues: Vec<f64>,
    /// Adjusted p-values (BH corrected)
    pub padj: Vec<f64>,
    /// Dispersion estimates used for each gene (MAP dispersions)
    pub dispersions: Vec<f64>,
    /// Gene-wise dispersion estimates (before shrinkage)
    pub gene_wise_dispersions: Vec<f64>,
    /// Trended dispersion estimates
    pub trended_dispersions: Vec<f64>,
    /// Contrast specification
    pub contrast: Contrast,
}

/// Contrast specification for differential expression
/// R equivalent: contrast parameter in results() in results.R
/// Supports three types (matching DESeq2):
/// 1. Simple: variable + numerator vs denominator
/// 2. Name: coefficient name directly from resultsNames()
/// 3. Numeric: numeric contrast vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contrast {
    /// Variable name (e.g., "treatment")
    pub variable: String,
    /// Numerator level (e.g., "treated")
    pub numerator: String,
    /// Denominator level (e.g., "control")
    pub denominator: String,
}

/// Extended contrast specification (DESeq2-style)
/// R equivalent: contrast/name/list parameter in results() in results.R
#[derive(Debug, Clone)]
pub enum ContrastSpec {
    /// Simple contrast: c("variable", "numerator", "denominator")
    /// Most common usage for two-group comparisons
    Simple {
        variable: String,
        numerator: String,
        denominator: String,
    },

    /// Name-based contrast: directly specify coefficient name
    /// from resultsNames(dds)
    Name(String),

    /// Numeric contrast: specify weights for each coefficient
    /// Length must match number of coefficients
    Numeric(Vec<f64>),

    /// List contrast: combine multiple coefficients
    /// Useful for comparing average of multiple conditions
    /// e.g., list(c("condition_B_vs_A", "condition_C_vs_A"))
    /// R's `listValues` parameter controls the weights: default [1, -1]
    List {
        numerator_coefs: Vec<String>,
        denominator_coefs: Vec<String>,
        /// Weights for numerator and denominator groups.
        /// Default: [1.0, -1.0] matching R's listValues=c(1,-1)
        list_values: (f64, f64),
    },
}

impl ContrastSpec {
    /// Create a simple contrast
    pub fn simple(variable: &str, numerator: &str, denominator: &str) -> Self {
        ContrastSpec::Simple {
            variable: variable.to_string(),
            numerator: numerator.to_string(),
            denominator: denominator.to_string(),
        }
    }

    /// Create a name-based contrast (for coefficient name from resultsNames)
    pub fn name(coef_name: &str) -> Self {
        ContrastSpec::Name(coef_name.to_string())
    }

    /// Create a numeric contrast
    pub fn numeric(weights: Vec<f64>) -> Self {
        ContrastSpec::Numeric(weights)
    }

    /// Create a list contrast for combining coefficients
    /// Uses default listValues of (1.0, -1.0) matching R's default
    pub fn list(numerator_coefs: Vec<String>, denominator_coefs: Vec<String>) -> Self {
        ContrastSpec::List {
            numerator_coefs,
            denominator_coefs,
            list_values: (1.0, -1.0),
        }
    }

    /// Create a list contrast with custom listValues
    /// R equivalent: results(dds, contrast=list(...), listValues=c(num_weight, denom_weight))
    pub fn list_with_values(
        numerator_coefs: Vec<String>,
        denominator_coefs: Vec<String>,
        list_values: (f64, f64),
    ) -> Self {
        ContrastSpec::List {
            numerator_coefs,
            denominator_coefs,
            list_values,
        }
    }

    /// Convert to legacy Contrast struct for compatibility
    pub fn to_legacy(&self) -> Option<Contrast> {
        match self {
            ContrastSpec::Simple {
                variable,
                numerator,
                denominator,
            } => Some(Contrast {
                variable: variable.clone(),
                numerator: numerator.clone(),
                denominator: denominator.clone(),
            }),
            _ => None,
        }
    }
}

impl DESeqResults {
    /// Create new empty results
    pub fn new(gene_ids: Vec<String>, contrast: Contrast) -> Self {
        let n = gene_ids.len();
        Self {
            gene_ids,
            base_means: vec![f64::NAN; n],
            base_vars: vec![f64::NAN; n],
            log2_fold_changes: vec![f64::NAN; n],
            lfc_se: vec![f64::NAN; n],
            stat: vec![f64::NAN; n],
            pvalues: vec![f64::NAN; n],
            padj: vec![f64::NAN; n],
            dispersions: vec![f64::NAN; n],
            gene_wise_dispersions: vec![f64::NAN; n],
            trended_dispersions: vec![f64::NAN; n],
            contrast,
        }
    }

    /// Get number of genes
    pub fn n_genes(&self) -> usize {
        self.gene_ids.len()
    }

    /// Get significant genes at given alpha level
    pub fn significant_genes(&self, alpha: f64) -> Vec<&str> {
        self.gene_ids
            .iter()
            .zip(self.padj.iter())
            .filter(|(_, &p)| p.is_finite() && p < alpha)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get up-regulated genes (positive log2FC, significant)
    pub fn upregulated_genes(&self, alpha: f64, min_lfc: f64) -> Vec<&str> {
        self.gene_ids
            .iter()
            .zip(self.padj.iter().zip(self.log2_fold_changes.iter()))
            .filter(|(_, (&p, &lfc))| p.is_finite() && p < alpha && lfc >= min_lfc)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get down-regulated genes (negative log2FC, significant)
    pub fn downregulated_genes(&self, alpha: f64, min_lfc: f64) -> Vec<&str> {
        self.gene_ids
            .iter()
            .zip(self.padj.iter().zip(self.log2_fold_changes.iter()))
            .filter(|(_, (&p, &lfc))| p.is_finite() && p < alpha && lfc <= -min_lfc)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Summary statistics
    pub fn summary(&self, alpha: f64) -> ResultsSummary {
        let total = self.n_genes();
        let tested = self.pvalues.iter().filter(|p| p.is_finite()).count();
        let significant = self.significant_genes(alpha).len();
        let upregulated = self.upregulated_genes(alpha, 0.0).len();
        let downregulated = self.downregulated_genes(alpha, 0.0).len();

        ResultsSummary {
            total_genes: total,
            genes_tested: tested,
            significant,
            upregulated,
            downregulated,
            alpha,
        }
    }
}

/// Summary of DESeq2 results
/// R equivalent: summary(res) output in results.R
#[derive(Debug, Clone)]
pub struct ResultsSummary {
    pub total_genes: usize,
    pub genes_tested: usize,
    pub significant: usize,
    pub upregulated: usize,
    pub downregulated: usize,
    pub alpha: f64,
}

impl std::fmt::Display for ResultsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DESeq2 Results Summary")?;
        writeln!(f, "======================")?;
        writeln!(f, "Total genes: {}", self.total_genes)?;
        writeln!(f, "Genes tested: {}", self.genes_tested)?;
        writeln!(
            f,
            "Significant (padj < {}): {}",
            self.alpha, self.significant
        )?;
        writeln!(f, "  Up-regulated: {}", self.upregulated)?;
        writeln!(f, "  Down-regulated: {}", self.downregulated)?;
        Ok(())
    }
}
