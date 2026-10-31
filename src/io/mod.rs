//! Input/Output operations for DESeq2

mod csv;
mod htseq;
mod results;

pub use self::csv::{read_count_matrix, read_metadata, read_weight_matrix, write_results};
pub use htseq::read_htseq_count;
pub use results::{Contrast, ContrastSpec, DESeqResults, ResultsSummary};
