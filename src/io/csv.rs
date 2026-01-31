//! CSV reading and writing for count matrices and metadata

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use ndarray::Array2;

use crate::data::{CountMatrix, SampleMetadata};
use crate::error::{DeseqError, Result};

/// Strip surrounding quotes from a string
fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len()-1].to_string()
    } else {
        s.to_string()
    }
}

/// Read a count matrix from a CSV file
/// No direct R equivalent -- Rust-specific I/O helper
/// Expected format: first column is gene IDs, first row is sample IDs
pub fn read_count_matrix<P: AsRef<Path>>(path: P) -> Result<CountMatrix> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let header_line = lines
        .next()
        .ok_or_else(|| DeseqError::EmptyData {
            reason: "Empty CSV file".to_string(),
        })??;

    let header: Vec<&str> = header_line.split('\t').collect();
    if header.len() < 2 {
        // Try comma separator
        let header: Vec<&str> = header_line.split(',').collect();
        if header.len() < 2 {
            return Err(DeseqError::InvalidCountMatrix {
                reason: "Not enough columns in header".to_string(),
            });
        }
    }

    // Detect delimiter
    let delimiter = if header_line.contains('\t') { '\t' } else { ',' };

    let header: Vec<&str> = header_line.split(delimiter).collect();
    let sample_ids: Vec<String> = header[1..].iter().map(|s| strip_quotes(s.trim())).collect();
    let n_samples = sample_ids.len();

    let mut gene_ids: Vec<String> = Vec::new();
    let mut counts_data: Vec<Vec<f64>> = Vec::new();

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).collect();
        if fields.len() != n_samples + 1 {
            return Err(DeseqError::InvalidCountMatrix {
                reason: format!(
                    "Row has {} columns, expected {}",
                    fields.len(),
                    n_samples + 1
                ),
            });
        }

        gene_ids.push(strip_quotes(fields[0].trim()));

        let row_counts: Result<Vec<f64>> = fields[1..]
            .iter()
            .map(|s| {
                let val = strip_quotes(s.trim());
                val.parse::<f64>().map_err(|_| DeseqError::InvalidCountMatrix {
                    reason: format!("Invalid count value: {}", val),
                })
            })
            .collect();

        counts_data.push(row_counts?);
    }

    if gene_ids.is_empty() {
        return Err(DeseqError::EmptyData {
            reason: "No genes found in count matrix".to_string(),
        });
    }

    let n_genes = gene_ids.len();
    let mut counts = Array2::zeros((n_genes, n_samples));

    for (i, row) in counts_data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            counts[[i, j]] = val;
        }
    }

    CountMatrix::new(counts, gene_ids, sample_ids)
}

/// Read sample metadata from a CSV file
/// No direct R equivalent -- Rust-specific I/O helper
/// Expected format: first column is sample IDs, remaining columns are conditions
pub fn read_metadata<P: AsRef<Path>>(path: P) -> Result<SampleMetadata> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let header_line = lines
        .next()
        .ok_or_else(|| DeseqError::EmptyData {
            reason: "Empty metadata file".to_string(),
        })??;

    // Detect delimiter
    let delimiter = if header_line.contains('\t') { '\t' } else { ',' };

    let header: Vec<&str> = header_line.split(delimiter).collect();
    let condition_names: Vec<String> = header[1..].iter().map(|s| strip_quotes(s.trim())).collect();

    let mut sample_ids: Vec<String> = Vec::new();
    let mut conditions: HashMap<String, Vec<String>> = HashMap::new();

    for name in &condition_names {
        conditions.insert(name.clone(), Vec::new());
    }

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).collect();
        if fields.len() != condition_names.len() + 1 {
            return Err(DeseqError::InvalidMetadata {
                reason: format!(
                    "Row has {} columns, expected {}",
                    fields.len(),
                    condition_names.len() + 1
                ),
            });
        }

        sample_ids.push(strip_quotes(fields[0].trim()));

        for (i, name) in condition_names.iter().enumerate() {
            conditions
                .get_mut(name)
                .unwrap()
                .push(strip_quotes(fields[i + 1].trim()));
        }
    }

    if sample_ids.is_empty() {
        return Err(DeseqError::EmptyData {
            reason: "No samples found in metadata".to_string(),
        });
    }

    let mut metadata = SampleMetadata::new(sample_ids);
    for (name, values) in conditions {
        metadata.add_condition(&name, values)?;
    }

    Ok(metadata)
}

/// Write DESeq2 results to a CSV file
/// No direct R equivalent -- Rust-specific I/O helper
pub fn write_results<P: AsRef<Path>>(
    path: P,
    gene_ids: &[String],
    results: &super::results::DESeqResults,
) -> Result<()> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "gene_id\tbaseMean\tbaseVar\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj\tdispersion\tgene_wise_disp\ttrended_disp")?;

    // Write data
    for (i, gene_id) in gene_ids.iter().enumerate() {
        writeln!(
            file,
            "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            gene_id,
            results.base_means.get(i).copied().unwrap_or(f64::NAN),
            results.base_vars.get(i).copied().unwrap_or(f64::NAN),
            results.log2_fold_changes.get(i).copied().unwrap_or(f64::NAN),
            results.lfc_se.get(i).copied().unwrap_or(f64::NAN),
            results.stat.get(i).copied().unwrap_or(f64::NAN),
            results.pvalues.get(i).copied().unwrap_or(f64::NAN),
            results.padj.get(i).copied().unwrap_or(f64::NAN),
            results.dispersions.get(i).copied().unwrap_or(f64::NAN),
            results.gene_wise_dispersions.get(i).copied().unwrap_or(f64::NAN),
            results.trended_dispersions.get(i).copied().unwrap_or(f64::NAN),
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_count_matrix() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "gene_id\ts1\ts2\ts3").unwrap();
        writeln!(file, "gene1\t100\t200\t150").unwrap();
        writeln!(file, "gene2\t50\t75\t60").unwrap();

        let matrix = read_count_matrix(file.path()).unwrap();
        assert_eq!(matrix.n_genes(), 2);
        assert_eq!(matrix.n_samples(), 3);
    }
}
