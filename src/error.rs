//! Error types for RustDESeq2

use thiserror::Error;

/// Main error type for DESeq2 operations
#[derive(Error, Debug)]
pub enum DeseqError {
    #[error("Invalid count matrix: {reason}")]
    InvalidCountMatrix { reason: String },

    #[error("Invalid metadata: {reason}")]
    InvalidMetadata { reason: String },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("GLM convergence failed for gene {gene_id}: {reason}")]
    GLMConvergenceFailed { gene_id: String, reason: String },

    #[error("Dispersion estimation failed for gene {gene_id}: {reason}")]
    DispersionEstimationFailed { gene_id: String, reason: String },

    #[error("Optimization failed: {reason}")]
    OptimizationFailed { reason: String },

    #[error("Numerical instability in {operation}: {details}")]
    NumericalInstability { operation: String, details: String },

    #[error("Invalid design matrix: {reason}")]
    InvalidDesignMatrix { reason: String },

    #[error("Invalid contrast specification: {reason}")]
    InvalidContrast { reason: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Empty data: {reason}")]
    EmptyData { reason: String },

    #[error("Size factor estimation failed: {reason}")]
    SizeFactorFailed { reason: String },

    #[error("Trend fitting failed: {reason}")]
    TrendFittingFailed { reason: String },

    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },
}

/// Result type alias for DESeq2 operations
pub type Result<T> = std::result::Result<T, DeseqError>;
