//! Log fold change shrinkage methods
//! R equivalent: lfcShrink() dispatcher in lfcShrink.R
//!
//! Provides three shrinkage methods following DESeq2:
//! - `normal`: Simple normal prior shrinkage
//! - `apeglm`: Approximate posterior estimation using Cauchy prior
//! - `ashr`: Adaptive shrinkage using mixture prior and EM algorithm

mod apeglm;
mod ashr;
mod normal;

pub use apeglm::{shrink_lfc_apeglm, ApeglmParams};
pub use ashr::{apply_ashr_shrinkage, shrink_lfc_ashr, AshrFit, AshrParams};
pub use normal::shrink_lfc_normal;
