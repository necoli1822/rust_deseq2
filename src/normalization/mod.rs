//! Normalization methods for RNA-seq count data

mod size_factors;
mod counts;

pub use size_factors::{estimate_size_factors, SizeFactorMethod};
pub use counts::{fpm, fpkm};
