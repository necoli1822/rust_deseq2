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
pub mod optim;
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
    // Step 1: Estimate size factors (skip if size or normalization factors are
    // already set, matching R DESeq2 behavior)
    if !dds.has_size_or_norm_factors() {
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

    fn weights_dataset() -> DESeqDataSet {
        let counts = CountMatrix::new(
            array![
                [100.0, 110.0, 90.0, 95.0, 400.0, 420.0, 380.0, 300.0],
                [500.0, 520.0, 480.0, 450.0, 500.0, 510.0, 490.0, 530.0],
                [300.0, 310.0, 290.0, 250.0, 75.0, 80.0, 70.0, 95.0],
                [50.0, 55.0, 45.0, 60.0, 50.0, 52.0, 48.0, 40.0],
                [200.0, 220.0, 180.0, 230.0, 200.0, 210.0, 190.0, 160.0],
                [150.0, 160.0, 140.0, 120.0, 300.0, 320.0, 280.0, 350.0],
                [400.0, 420.0, 380.0, 450.0, 100.0, 110.0, 90.0, 85.0],
                [80.0, 85.0, 75.0, 60.0, 80.0, 82.0, 78.0, 99.0],
                [600.0, 620.0, 580.0, 650.0, 600.0, 610.0, 590.0, 560.0],
                [250.0, 260.0, 240.0, 210.0, 500.0, 520.0, 480.0, 600.0],
            ],
            (0..10).map(|i| format!("g{}", i)).collect(),
            (0..8).map(|j| format!("s{}", j)).collect(),
        )
        .unwrap();
        let mut metadata = SampleMetadata::new((0..8).map(|j| format!("s{}", j)).collect());
        metadata
            .add_condition("treatment", (0..8).map(|j| if j < 4 { "a" } else { "b" }.to_string()).collect())
            .unwrap();
        DESeqDataSet::new(counts, metadata, "treatment").unwrap()
    }

    fn weights_matrix() -> ndarray::Array2<f64> {
        ndarray::Array2::from_shape_fn((10, 8), |(i, j)| 0.2 + 0.8 * (((i * 7 + j * 3) % 11) as f64) / 10.0)
    }

    /// The fits use weights normalised by their row maximum, so scaling all
    /// weights leaves the tests unchanged while baseMean (raw weights) scales.
    #[test]
    fn test_weights_scale_invariance() {
        let run = |scale: f64| {
            let mut dds = weights_dataset();
            dds.set_weights(weights_matrix() * scale).unwrap();
            let info = run_deseq(&mut dds).unwrap();
            deseq_results(&dds, &info, "b", "a", 0.1).unwrap()
        };
        let (r1, r3) = (run(1.0), run(3.0));
        for i in 0..10 {
            assert!((r1.log2_fold_changes[i] - r3.log2_fold_changes[i]).abs() < 1e-9);
            assert!((r1.lfc_se[i] - r3.lfc_se[i]).abs() < 1e-9);
            assert!((r1.pvalues[i] - r3.pvalues[i]).abs() < 1e-9);
            assert!((3.0 * r1.base_means[i] - r3.base_means[i]).abs() < 1e-9 * r3.base_means[i]);
        }
        // weighted results differ from unweighted ones
        let mut dds = weights_dataset();
        let info = run_deseq(&mut dds).unwrap();
        let r0 = deseq_results(&dds, &info, "b", "a", 0.1).unwrap();
        assert!((r0.log2_fold_changes[0] - r1.log2_fold_changes[0]).abs() > 1e-6);
    }

    /// R getAndCheckWeights(): a gene whose weights leave a coefficient
    /// inestimable is flagged weightsFail and treated as allZero (NA results).
    #[test]
    fn test_weights_fail_gives_na() {
        let mut dds = weights_dataset();
        let mut w = weights_matrix();
        for j in 4..8 {
            w[[2, j]] = 0.0;
        }
        dds.set_weights(w).unwrap();
        let info = run_deseq(&mut dds).unwrap();
        let res = deseq_results(&dds, &info, "b", "a", 0.1).unwrap();
        assert_eq!(dds.weights_fail().unwrap().iter().filter(|&&f| f).count(), 1);
        assert!(dds.weights_fail().unwrap()[2]);
        assert!(res.log2_fold_changes[2].is_nan() && res.pvalues[2].is_nan());
        assert!(res.pvalues.iter().enumerate().all(|(i, p)| i == 2 || p.is_finite()));
        assert!(dds.set_weights(ndarray::Array2::ones((3, 8))).is_err());
        assert!(dds.set_weights(ndarray::Array2::from_elem((10, 8), -1.0)).is_err());
    }
}
