//! Count input from htseq-count output files
//!
//! R equivalent: DESeqDataSetFromHTSeqCount(sampleTable, directory, design) in AllClasses.R

use std::fs;
use std::path::Path;

use ndarray::Array2;

use crate::data::{CountMatrix, SampleMetadata};
use crate::error::{DeseqError, Result};

/// htseq-count's special counters from before they gained the `__` prefix
const OLD_SPECIAL_NAMES: [&str; 5] = ["no_feature", "ambiguous", "too_low_aQual", "not_aligned", "alignment_not_unique"];

/// Read htseq-count files listed in a sample table
/// R equivalent: DESeqDataSetFromHTSeqCount()
///
/// The sample table (CSV or TSV, with a header) has the sample name in its first
/// column, the file name (relative to `directory`) in its second, and the sample
/// variables in the remaining columns, which become the sample metadata. As in R:
/// - each file is read as whitespace-separated columns (`read.table(fill = TRUE)`),
///   the first column being the gene ID and the last the count, so files written
///   with `--additional-attr` work;
/// - the gene IDs must be identical, in the same order, in every file;
/// - htseq-count's counter rows (IDs starting with `_`, e.g. `__no_feature`, and the
///   older `no_feature`, `ambiguous`, ... names) are dropped.
pub fn read_htseq_count<P: AsRef<Path>, Q: AsRef<Path>>(
    sample_table: P,
    directory: Q,
) -> Result<(CountMatrix, SampleMetadata)> {
    let (header, rows) = read_table(sample_table.as_ref())?;
    if header.len() < 2 {
        return Err(DeseqError::InvalidMetadata {
            reason: "HTSeq sample table needs at least two columns: sample name and file name".to_string(),
        });
    }

    let sample_names: Vec<String> = rows.iter().map(|r| r[0].clone()).collect();
    let mut gene_ids: Option<Vec<String>> = None;
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(rows.len());

    for row in &rows {
        let path = directory.as_ref().join(&row[1]);
        let (ids, counts) = read_htseq_file(&path)?;
        match &gene_ids {
            None => gene_ids = Some(ids),
            Some(first) => {
                if *first != ids {
                    return Err(DeseqError::InvalidInput {
                        reason: format!("Gene IDs (first column) differ between files ({})", path.display()),
                    });
                }
            }
        }
        columns.push(counts);
    }
    let gene_ids = gene_ids.ok_or_else(|| DeseqError::EmptyData {
        reason: "HTSeq sample table lists no samples".to_string(),
    })?;

    // specialRows <- substr(rownames(tbl), 1, 1) == "_" | rownames(tbl) %in% oldSpecialNames
    let keep: Vec<usize> = (0..gene_ids.len())
        .filter(|&i| !gene_ids[i].starts_with('_') && !OLD_SPECIAL_NAMES.contains(&gene_ids[i].as_str()))
        .collect();
    let mut matrix = Array2::zeros((keep.len(), columns.len()));
    for (j, col) in columns.iter().enumerate() {
        for (r, &i) in keep.iter().enumerate() {
            matrix[[r, j]] = col[i];
        }
    }
    let kept_ids: Vec<String> = keep.iter().map(|&i| gene_ids[i].clone()).collect();
    let counts = CountMatrix::new(matrix, kept_ids, sample_names.clone())?;

    // colData = sampleTable[, -(1:2)]
    let mut metadata = SampleMetadata::new(sample_names);
    for (c, name) in header.iter().enumerate().skip(2) {
        metadata.add_condition(name, rows.iter().map(|r| r[c].clone()).collect())?;
    }
    Ok((counts, metadata))
}

/// Read one htseq-count output file: (gene IDs, counts from the last column)
fn read_htseq_file(path: &Path) -> Result<(Vec<String>, Vec<f64>)> {
    let text = fs::read_to_string(path).map_err(|e| DeseqError::InvalidInput {
        reason: format!("Cannot read HTSeq count file {}: {}", path.display(), e),
    })?;
    let mut ids = Vec::new();
    let mut counts = Vec::new();
    for line in text.lines() {
        // read.table: '#' starts a comment, blank lines are skipped
        let line = line.split('#').next().unwrap_or("");
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if fields.len() < 2 {
            return Err(DeseqError::InvalidInput {
                reason: format!("{}: line without a count: '{}'", path.display(), line),
            });
        }
        let value = fields[fields.len() - 1];
        let count: f64 = value.parse().map_err(|_| DeseqError::InvalidInput {
            reason: format!("{}: non-numeric count '{}' for {}", path.display(), value, fields[0]),
        })?;
        ids.push(fields[0].to_string());
        counts.push(count);
    }
    Ok((ids, counts))
}

/// A delimited table with a header (CSV or TSV, auto-detected), quotes stripped
fn read_table(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let text = fs::read_to_string(path).map_err(|e| DeseqError::InvalidInput {
        reason: format!("Cannot read sample table {}: {}", path.display(), e),
    })?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header_line = lines.next().ok_or_else(|| DeseqError::EmptyData {
        reason: "Empty HTSeq sample table".to_string(),
    })?;
    let delimiter = if header_line.contains('\t') { '\t' } else { ',' };
    let clean = |s: &str| s.trim().trim_matches('"').trim_matches('\'').to_string();
    let header: Vec<String> = header_line.split(delimiter).map(clean).collect();
    let mut rows = Vec::new();
    for line in lines {
        let row: Vec<String> = line.split(delimiter).map(clean).collect();
        if row.len() != header.len() {
            return Err(DeseqError::InvalidMetadata {
                reason: format!("Sample table row has {} columns, expected {}", row.len(), header.len()),
            });
        }
        rows.push(row);
    }
    Ok((header, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_htseq_count_drops_counters_and_reads_last_column() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| {
            let mut f = fs::File::create(dir.path().join(name)).unwrap();
            f.write_all(body.as_bytes()).unwrap();
        };
        write("a.txt", "g1\t10\ng2\t0\nno_feature\t3\n__ambiguous\t4\n");
        // --additional-attr: an extra column before the count
        write("b.txt", "g1 nameA 7\ng2 nameB 2\nno_feature x 1\n__ambiguous y 9\n");
        write("table.csv", "sampleName,fileName,condition\nS1,a.txt,ctrl\nS2,b.txt,trt\n");

        let (counts, meta) = read_htseq_count(dir.path().join("table.csv"), dir.path()).unwrap();
        assert_eq!(counts.gene_ids(), &["g1".to_string(), "g2".to_string()]);
        assert_eq!(counts.sample_ids(), &["S1".to_string(), "S2".to_string()]);
        assert_eq!(counts.counts()[[0, 1]], 7.0);
        assert_eq!(counts.counts()[[1, 1]], 2.0);
        assert_eq!(meta.condition("condition").unwrap(), &vec!["ctrl".to_string(), "trt".to_string()]);
        assert!(!meta.has_condition("fileName"));
    }

    #[test]
    fn test_read_htseq_count_rejects_mismatched_gene_ids() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "g1\t1\ng2\t2\n").unwrap();
        fs::write(dir.path().join("b.txt"), "g2\t1\ng1\t2\n").unwrap();
        fs::write(dir.path().join("t.tsv"), "sample\tfile\tcondition\nA\ta.txt\tx\nB\tb.txt\ty\n").unwrap();
        assert!(read_htseq_count(dir.path().join("t.tsv"), dir.path()).is_err());
    }
}
