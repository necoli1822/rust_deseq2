//! RustDESeq2: DESeq2 differential expression analysis in Rust
//!
//! This crate provides a Rust implementation of the DESeq2 algorithm for
//! differential expression analysis of RNA-seq data.
//!
//! # Example
//!
//! ```ignore
//! use rust_deseq2::prelude::*;
//!
//! // Load data
//! let counts = read_count_matrix("counts.csv")?;
//! let metadata = read_metadata("metadata.csv")?;
//!
//! // Create dataset
//! let mut dds = DESeqDataSet::new(counts, metadata, "treatment")?;
//!
//! // Run analysis
//! run_deseq(&mut dds)?;
//!
//! // Get results
//! let results = results(&dds, contrast)?;
//! ```

pub mod cli;
pub mod data;
pub mod dispersion;
pub mod error;
pub mod filter;
pub mod glm;
pub mod io;
pub mod normalization;
pub mod rng;
pub mod shrinkage;
pub mod stats;
pub mod testing;
pub mod transform;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data::{collapse_replicates, CountMatrix, DESeqDataSet, GeneMetadata, SampleMetadata};
    pub use crate::dispersion::{estimate_dispersions, DispersionParams, TrendFitMethod};
    pub use crate::error::{DeseqError, Result};
    pub use crate::filter::independent_filtering;
    pub use crate::glm::{fit_glm, DesignInfo, GlmFitParams};
    pub use crate::io::{read_count_matrix, read_metadata, write_results, Contrast, ContrastSpec, DESeqResults};
    pub use crate::normalization::{estimate_size_factors, SizeFactorMethod, fpm, fpkm};
    pub use crate::shrinkage::shrink_lfc_normal;
    pub use crate::testing::{benjamini_hochberg, results, results_extended, results_names, wald_test};
    pub use crate::transform::{rlog, vst, RlogResult, VstMethod, VstResult};
}

use prelude::*;

/// Run the complete DESeq2 analysis pipeline
/// R equivalent: DESeq() in core.R
pub fn run_deseq(dds: &mut DESeqDataSet) -> Result<DesignInfo> {
    // Step 1: Estimate size factors (skip if already set, matching R DESeq2 behavior)
    if !dds.has_size_factors() {
        estimate_size_factors(dds, SizeFactorMethod::Ratio)?;
    }

    // Step 2: Estimate dispersions
    estimate_dispersions(dds, TrendFitMethod::Parametric, &DispersionParams::default())?;

    // Step 3: Fit GLM
    let design_info = fit_glm(dds, &GlmFitParams::default())?;

    Ok(design_info)
}

/// Run analysis and get results for a specific contrast
/// R equivalent: results(dds, contrast=c(...)) in results.R
pub fn deseq_results(
    dds: &DESeqDataSet,
    design_info: &DesignInfo,
    numerator: &str,
    denominator: &str,
    alpha: f64,
) -> Result<DESeqResults> {
    let contrast = Contrast {
        variable: dds.design_variable().to_string(),
        numerator: numerator.to_string(),
        denominator: denominator.to_string(),
    };

    results(dds, design_info, contrast, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_full_pipeline() {
        // Create test data with differential expression
        let counts = CountMatrix::new(
            array![
                [100.0, 110.0, 90.0, 400.0, 420.0, 380.0],  // Up-regulated
                [500.0, 520.0, 480.0, 500.0, 510.0, 490.0], // No change
                [300.0, 310.0, 290.0, 75.0, 80.0, 70.0],    // Down-regulated
                [50.0, 55.0, 45.0, 50.0, 52.0, 48.0],       // No change (low)
                [200.0, 220.0, 180.0, 200.0, 210.0, 190.0], // No change (medium)
                [150.0, 160.0, 140.0, 300.0, 320.0, 280.0], // Up-regulated 2
                [400.0, 420.0, 380.0, 100.0, 110.0, 90.0],  // Down-regulated 2
                [80.0, 85.0, 75.0, 80.0, 82.0, 78.0],       // No change (low-med)
                [600.0, 620.0, 580.0, 600.0, 610.0, 590.0], // No change (high)
                [250.0, 260.0, 240.0, 500.0, 520.0, 480.0], // Up-regulated 3
            ],
            vec![
                "gene_up".to_string(),
                "gene_nc1".to_string(),
                "gene_down".to_string(),
                "gene_nc2".to_string(),
                "gene_nc3".to_string(),
                "gene_up2".to_string(),
                "gene_down2".to_string(),
                "gene_nc4".to_string(),
                "gene_nc5".to_string(),
                "gene_up3".to_string(),
            ],
            vec![
                "s1".to_string(),
                "s2".to_string(),
                "s3".to_string(),
                "s4".to_string(),
                "s5".to_string(),
                "s6".to_string(),
            ],
        )
        .unwrap();

        let mut metadata = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
            "s5".to_string(),
            "s6".to_string(),
        ]);
        metadata
            .add_condition(
                "treatment",
                vec![
                    "control".to_string(),
                    "control".to_string(),
                    "control".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                ],
            )
            .unwrap();

        let mut dds = DESeqDataSet::new(counts, metadata, "treatment").unwrap();

        // Run full pipeline
        let design_info = run_deseq(&mut dds).unwrap();

        // Get results
        let results = deseq_results(&dds, &design_info, "treated", "control", 0.05).unwrap();

        // Check results
        assert_eq!(results.gene_ids.len(), 10);

        // gene_up should have positive log2FC
        assert!(results.log2_fold_changes[0] > 1.0, "gene_up should be up-regulated");

        // gene_down should have negative log2FC
        assert!(results.log2_fold_changes[2] < -1.0, "gene_down should be down-regulated");

        // Print summary
        let summary = results.summary(0.05);
        println!("{}", summary);
    }
}
