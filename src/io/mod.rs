//! Input/Output operations for DESeq2

mod csv;
mod results;

pub use self::csv::{read_count_matrix, read_metadata, write_results};
pub use results::{Contrast, ContrastSpec, DESeqResults, ResultsSummary};
