//! Statistical testing for differential expression

mod fdr;
mod lrt;
mod pvalue;
mod wald;

pub use fdr::{benjamini_hochberg, bonferroni};
pub use lrt::likelihood_ratio_test;
pub use pvalue::calculate_pvalue;
pub use wald::{wald_test, wald_test_extended};

use crate::data::DESeqDataSet;
use crate::error::Result;
use crate::glm::DesignInfo;
use crate::io::{Contrast, ContrastSpec, DESeqResults};

/// Perform statistical testing and return results
/// R equivalent: results() in results.R
pub fn results(
    dds: &DESeqDataSet,
    design_info: &DesignInfo,
    contrast: Contrast,
    alpha: f64,
) -> Result<DESeqResults> {
    wald_test(dds, design_info, contrast, alpha, false, 0.0)
}

/// Perform statistical testing with extended contrast specification
/// R equivalent: results() with numeric/list contrast in results.R
/// Supports DESeq2-style contrast types:
/// - Simple: c("variable", "numerator", "denominator")
/// - Name: coefficient name from resultsNames()
/// - Numeric: contrast vector
/// - List: combine multiple coefficients
pub fn results_extended(
    dds: &DESeqDataSet,
    design_info: &DesignInfo,
    contrast: ContrastSpec,
    alpha: f64,
) -> Result<DESeqResults> {
    wald_test_extended(dds, design_info, contrast, alpha, false, 0.0)
}

/// Get the names of available coefficients (equivalent to DESeq2's resultsNames())
/// R equivalent: resultsNames() in results.R
pub fn results_names(dds: &DESeqDataSet) -> Option<Vec<String>> {
    dds.design_column_names().cloned()
}
