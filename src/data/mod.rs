//! Data structures for DESeq2 analysis

pub mod builder;
mod collapse;
mod count_matrix;
mod dataset;
mod metadata;

pub use builder::{DESeqDataSetBuilder, DesignMatrixBuilder, DesignMatrixInfo};
pub use collapse::collapse_replicates;
pub use count_matrix::CountMatrix;
pub use dataset::DESeqDataSet;
pub use metadata::{GeneMetadata, SampleMetadata};
