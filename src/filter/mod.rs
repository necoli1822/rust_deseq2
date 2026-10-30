//! Filtering methods for DESeq2 results

mod cooks;
mod independent;

pub use cooks::{
    calculate_cooks_distance, default_cooks_cutoff, filter_by_cooks, filter_by_cooks_with_counts,
    max_cooks_per_gene, n_or_more_in_cell, replace_outliers, robust_method_of_moments_disp,
};
pub use independent::independent_filtering;
