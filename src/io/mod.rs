//! Input/Output operations for DESeq2

mod csv;
mod htseq;
mod results;
pub mod tximport;

pub use self::csv::{read_aligned_matrix, read_count_matrix, read_metadata, read_tximport_counts, read_weight_matrix, write_results};
pub use htseq::read_htseq_count;
pub use tximport::{read_quant, read_salmon, QuantType, Txi};
pub use results::{Contrast, ContrastSpec, DESeqResults, ResultsSummary};
