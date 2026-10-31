//! Transcript-level quantifications summarized to genes (tximport)
//!
//! R equivalent: tximport::tximport(files, type = "salmon", tx2gene = tx2gene) with
//! its defaults (countsFromAbundance = "no", txOut = FALSE, ignoreTxVersion = FALSE),
//! followed by DESeqDataSetFromTximport(txi, colData, design).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use ndarray::Array2;

use crate::data::{r_collate, SampleMetadata};
use crate::error::{DeseqError, Result};
use crate::stats::ld_mean;

/// Gene-level matrices returned by tximport (genes x samples)
/// R equivalent: the list returned by tximport() with txOut = FALSE
#[derive(Debug, Clone)]
pub struct Txi {
    /// Gene IDs, in R's sorted order of the tx2gene gene factor levels
    pub gene_ids: Vec<String>,
    /// Sample names
    pub sample_ids: Vec<String>,
    /// txi$abundance: summed TPM
    pub abundance: Array2<f64>,
    /// txi$counts: summed estimated counts (not rounded)
    pub counts: Array2<f64>,
    /// txi$length: abundance-weighted mean of the transcripts' effective lengths
    pub length: Array2<f64>,
}

/// One salmon `quant.sf` file
#[derive(Debug, Clone)]
pub struct SalmonQuant {
    /// Name column
    pub names: Vec<String>,
    /// EffectiveLength column
    pub effective_length: Vec<f64>,
    /// TPM column
    pub tpm: Vec<f64>,
    /// NumReads column
    pub num_reads: Vec<f64>,
}

/// R's round(x) (digits = 0): the nearest integer, halves to even
pub fn r_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Parse a number as R's read.delim() / type.convert() / as.numeric() do
/// R equivalent: R_strtod() (arithmetic.c)
///
/// R accumulates the decimal digits in a long double and divides by the power of
/// ten in long double, then rounds to double, so a few decimal strings end up one
/// ulp away from the correctly rounded value Rust's parser gives (e.g. "77.919041").
/// This reproduces that double rounding (64-bit then 53-bit significand, ties to
/// even) for plain decimals with fewer than 18 digits, the case R handles this
/// way; anything else (more digits, which R hands to the C library's strtod, or
/// NaN/Inf/hex forms) is parsed by Rust's correctly rounded parser.
pub fn r_strtod(text: &str) -> Option<f64> {
    let t = text.trim();
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    let fallback = || t.parse::<f64>().ok();
    let bytes = body.as_bytes();
    let mut k = 0;
    let (mut mant, mut ndigits, mut expn) = (0u128, 0usize, 0i64);
    while k < bytes.len() && bytes[k].is_ascii_digit() {
        mant = mant * 10 + (bytes[k] - b'0') as u128;
        ndigits += 1;
        k += 1;
        if ndigits > 30 {
            return fallback();
        }
    }
    if k < bytes.len() && bytes[k] == b'.' {
        k += 1;
        while k < bytes.len() && bytes[k].is_ascii_digit() {
            mant = mant * 10 + (bytes[k] - b'0') as u128;
            ndigits += 1;
            expn -= 1;
            k += 1;
            if ndigits > 30 {
                return fallback();
            }
        }
    }
    if ndigits == 0 {
        return fallback();
    }
    if k < bytes.len() && (bytes[k] == b'e' || bytes[k] == b'E') {
        k += 1;
        let mut esign = 1i64;
        if k < bytes.len() && (bytes[k] == b'-' || bytes[k] == b'+') {
            if bytes[k] == b'-' {
                esign = -1;
            }
            k += 1;
        }
        let start = k;
        let mut e = 0i64;
        while k < bytes.len() && bytes[k].is_ascii_digit() {
            e = (e * 10 + (bytes[k] - b'0') as i64).min(100_000);
            k += 1;
        }
        if k == start {
            return fallback();
        }
        expn += esign * e;
    }
    if k != bytes.len() || ndigits >= 18 || mant == 0 || expn.abs() > 27 {
        return fallback();
    }
    // 10^|expn| (<= 10^27) is exact in a long double, as is the digit accumulation
    let p10 = 10u128.pow(expn.unsigned_abs() as u32);
    let value = if expn < 0 {
        long_double_quotient(mant, p10)
    } else {
        match mant.checked_mul(p10) {
            Some(n) => long_double_quotient(n, 1),
            None => return fallback(),
        }
    };
    Some(sign * value)
}

/// num / den rounded to a 64-bit significand (an x87 long double) and then to a
/// double, both to nearest with ties to even
fn long_double_quotient(num: u128, den: u128) -> f64 {
    // binary digits of num / den from the most significant one: 64 significand
    // bits plus a round bit, and whether anything non-zero follows (sticky)
    let mut bits: Vec<bool> = Vec::with_capacity(65);
    let mut exp2: i64; // value = 0.b1 b2 b3 ... * 2^exp2
    let q = num / den;
    let mut r = num % den;
    let mut sticky = false;
    if q > 0 {
        let nb = 128 - q.leading_zeros() as i64;
        exp2 = nb;
        for i in (0..nb).rev() {
            let b = (q >> i) & 1 == 1;
            if bits.len() < 65 {
                bits.push(b);
            } else if b {
                sticky = true;
            }
        }
    } else {
        exp2 = 0;
        // skip leading zero bits of the fraction
        loop {
            r *= 2;
            if r >= den {
                r -= den;
                bits.push(true);
                break;
            }
            exp2 -= 1;
        }
    }
    while bits.len() < 65 {
        r *= 2;
        if r >= den {
            r -= den;
            bits.push(true);
        } else {
            bits.push(false);
        }
    }
    sticky |= r != 0;
    let mut m64: u128 = 0;
    for &b in &bits[..64] {
        m64 = (m64 << 1) | b as u128;
    }
    // value = m64 * 2^(exp2 - 64); round to 64 bits
    let mut e = exp2 - 64;
    if bits[64] && (sticky || m64 & 1 == 1) {
        m64 += 1;
        if m64 >> 64 == 1 {
            m64 >>= 1;
            e += 1;
        }
    }
    // round the 64-bit significand to 53 bits
    let low = m64 & 0x7FF;
    let mut m53 = m64 >> 11;
    e += 11;
    if low > 0x400 || (low == 0x400 && m53 & 1 == 1) {
        m53 += 1;
        if m53 >> 53 == 1 {
            m53 >>= 1;
            e += 1;
        }
    }
    (m53 as f64) * 2f64.powi(e as i32)
}

/// Read one salmon `quant.sf` file
/// R equivalent: the importer of tximport(type = "salmon"): read.delim(file), with
/// txIdCol = "Name", abundanceCol = "TPM", countsCol = "NumReads",
/// lengthCol = "EffectiveLength" (columns found by name)
pub fn read_salmon_quant<P: AsRef<Path>>(path: P) -> Result<SalmonQuant> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|e| DeseqError::InvalidInput {
        reason: format!("Cannot read salmon quantification {}: {}", path.display(), e),
    })?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| DeseqError::EmptyData { reason: format!("{} is empty", path.display()) })?
        .split('\t')
        .map(|s| s.trim())
        .collect();
    let col = |name: &str| -> Result<usize> {
        header.iter().position(|&h| h == name).ok_or_else(|| DeseqError::InvalidInput {
            reason: format!("{}: no '{}' column (salmon quant.sf expected)", path.display(), name),
        })
    };
    let (c_name, c_len, c_tpm, c_reads) = (col("Name")?, col("EffectiveLength")?, col("TPM")?, col("NumReads")?);
    let mut q = SalmonQuant { names: Vec::new(), effective_length: Vec::new(), tpm: Vec::new(), num_reads: Vec::new() };
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != header.len() {
            return Err(DeseqError::InvalidInput {
                reason: format!("{}: row has {} columns, expected {}", path.display(), fields.len(), header.len()),
            });
        }
        let num = |c: usize| -> Result<f64> {
            r_strtod(fields[c]).ok_or_else(|| DeseqError::InvalidInput {
                reason: format!("{}: non-numeric value '{}' for {}", path.display(), fields[c], fields[c_name]),
            })
        };
        q.names.push(fields[c_name].trim().to_string());
        q.effective_length.push(num(c_len)?);
        q.tpm.push(num(c_tpm)?);
        q.num_reads.push(num(c_reads)?);
    }
    Ok(q)
}

/// Read a transcript-to-gene table: the first two columns (transcript ID, gene ID)
/// of a CSV or TSV file; further columns are ignored.
/// R equivalent: the tx2gene data.frame of tximport(), whose first two columns
/// cleanTx2Gene() takes by position (`colnames(tx2gene) <- c("tx", "gene")`)
///
/// Every line is read as a mapping. A header line is harmless: summarizeToGene()
/// keeps only the rows whose transcript is quantified, so a header is dropped
/// unless its first field is a transcript ID.
pub fn read_tx2gene<P: AsRef<Path>>(path: P) -> Result<Vec<(String, String)>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|e| DeseqError::InvalidInput {
        reason: format!("Cannot read tx2gene file {}: {}", path.display(), e),
    })?;
    let clean = |s: &str| s.trim().trim_matches('"').trim_matches('\'').to_string();
    let mut map = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let delimiter = if line.contains('\t') { '\t' } else { ',' };
        let fields: Vec<&str> = line.split(delimiter).collect();
        if fields.len() < 2 {
            return Err(DeseqError::InvalidInput {
                reason: format!("{}: tx2gene needs two columns (transcript, gene): '{}'", path.display(), line),
            });
        }
        map.push((clean(fields[0]), clean(fields[1])));
    }
    if map.is_empty() {
        return Err(DeseqError::EmptyData { reason: format!("{} is empty", path.display()) });
    }
    Ok(map)
}

/// Summarize transcript-level matrices (transcripts x samples) to genes
/// R equivalent: tximport's summarizeToGene() with countsFromAbundance = "no" and
/// ignoreTxVersion = ignoreAfterBar = FALSE:
/// - tx2gene rows with a duplicated transcript are dropped (cleanTx2Gene, first kept);
///   transcripts missing from tx2gene are dropped (with a message);
/// - genes are ordered as the levels of factor(gene), i.e. sorted;
/// - abundance and counts are sums over the gene's transcripts (rowsum);
/// - length = rowsum(abundance * length) / abundance; where the gene's abundance is
///   zero in a sample (NaN), replaceMissingLength() takes the geometric mean of the
///   gene's lengths in the other samples, or, if it is zero in every sample, the
///   mean over its transcripts of their mean effective length across samples.
pub fn summarize_to_gene(
    tx_ids: &[String],
    abundance: &Array2<f64>,
    counts: &Array2<f64>,
    length: &Array2<f64>,
    tx2gene: &[(String, String)],
    sample_ids: Vec<String>,
) -> Result<Txi> {
    let n_samples = abundance.ncols();
    // cleanTx2Gene(): drop duplicated transcripts, keeping the first row
    let mut gene_of: HashMap<&str, &str> = HashMap::new();
    for (tx, gene) in tx2gene {
        gene_of.entry(tx.as_str()).or_insert(gene.as_str());
    }
    if !tx_ids.iter().any(|t| gene_of.contains_key(t.as_str())) {
        return Err(DeseqError::InvalidInput {
            reason: format!(
                "None of the transcripts in the quantification files are present in the first column of tx2gene \
                 (file: {:?}, tx2gene: {:?})",
                tx_ids.iter().take(3).collect::<Vec<_>>(),
                tx2gene.iter().take(3).map(|p| &p.0).collect::<Vec<_>>()
            ),
        });
    }
    let kept: Vec<usize> = (0..tx_ids.len()).filter(|&t| gene_of.contains_key(tx_ids[t].as_str())).collect();
    let n_missing = tx_ids.len() - kept.len();
    if n_missing > 0 {
        log::info!("transcripts missing from tx2gene: {}", n_missing);
    }

    // geneId <- droplevels(factor(gene))[match(txId, tx)]; rowsum() groups by its levels
    let mut genes: Vec<&str> = kept.iter().map(|&t| gene_of[tx_ids[t].as_str()]).collect::<HashSet<_>>().into_iter().collect();
    genes.sort_by(|a, b| r_collate(a, b).then_with(|| a.cmp(b)));
    let gene_index: HashMap<&str, usize> = genes.iter().enumerate().map(|(g, &id)| (id, g)).collect();
    let group: Vec<usize> = kept.iter().map(|&t| gene_index[gene_of[tx_ids[t].as_str()]]).collect();
    let n_genes = genes.len();

    // rowsum(): sums in transcript order
    let mut abundance_gene: Array2<f64> = Array2::zeros((n_genes, n_samples));
    let mut counts_gene: Array2<f64> = Array2::zeros((n_genes, n_samples));
    let mut weighted_length: Array2<f64> = Array2::zeros((n_genes, n_samples));
    for (&t, &g) in kept.iter().zip(&group) {
        for j in 0..n_samples {
            abundance_gene[[g, j]] += abundance[[t, j]];
            counts_gene[[g, j]] += counts[[t, j]];
            weighted_length[[g, j]] += abundance[[t, j]] * length[[t, j]];
        }
    }
    let mut length_gene = Array2::from_shape_fn((n_genes, n_samples), |(g, j)| weighted_length[[g, j]] / abundance_gene[[g, j]]);

    // aveLengthSamp <- rowMeans(lengthMatTx); aveLengthSampGene <- tapply(aveLengthSamp, geneId, mean)
    let mut tx_means: Vec<Vec<f64>> = vec![Vec::new(); n_genes];
    for (&t, &g) in kept.iter().zip(&group) {
        tx_means[g].push(ld_mean(length.row(t).iter().copied()));
    }
    // replaceMissingLength()
    for g in 0..n_genes {
        let nan: Vec<bool> = (0..n_samples).map(|j| length_gene[[g, j]].is_nan()).collect();
        if !nan.iter().any(|&b| b) {
            continue;
        }
        let fill = if nan.iter().all(|&b| b) {
            ld_mean(tx_means[g].iter().copied())
        } else {
            ld_mean((0..n_samples).filter(|&j| !nan[j]).map(|j| length_gene[[g, j]].ln())).exp()
        };
        for j in 0..n_samples {
            if nan[j] {
                length_gene[[g, j]] = fill;
            }
        }
    }

    Ok(Txi {
        gene_ids: genes.iter().map(|s| s.to_string()).collect(),
        sample_ids,
        abundance: abundance_gene,
        counts: counts_gene,
        length: length_gene,
    })
}

/// Import salmon quantifications and summarize them to genes
/// R equivalent: tximport(files, type = "salmon", tx2gene = tx2gene)
///
/// Every file must list the same transcripts in the same order
/// (R: stopifnot(all(txId == raw[[txIdCol]]))).
pub fn tximport_salmon<P: AsRef<Path>>(files: &[P], sample_ids: Vec<String>, tx2gene: &[(String, String)]) -> Result<Txi> {
    if files.is_empty() {
        return Err(DeseqError::EmptyData { reason: "no salmon quantification files".to_string() });
    }
    let quants: Vec<SalmonQuant> = files.iter().map(read_salmon_quant).collect::<Result<_>>()?;
    let tx_ids = quants[0].names.clone();
    for (q, f) in quants.iter().zip(files) {
        if q.names != tx_ids {
            return Err(DeseqError::InvalidInput {
                reason: format!("{}: transcripts differ from the first file", f.as_ref().display()),
            });
        }
    }
    let n_tx = tx_ids.len();
    let n = quants.len();
    let abundance = Array2::from_shape_fn((n_tx, n), |(t, j)| quants[j].tpm[t]);
    let counts = Array2::from_shape_fn((n_tx, n), |(t, j)| quants[j].num_reads[t]);
    let length = Array2::from_shape_fn((n_tx, n), |(t, j)| quants[j].effective_length[t]);
    summarize_to_gene(&tx_ids, &abundance, &counts, &length, tx2gene, sample_ids)
}

/// Read salmon `quant.sf` files listed in a sample table and summarize them to genes
///
/// The sample table (CSV or TSV, with a header) has the sample name in its first
/// column, the path of the sample's `quant.sf` (relative to `directory`) in its
/// second, and the sample variables in the remaining columns, which become the
/// sample metadata. `tx2gene` is read with [`read_tx2gene`].
/// R equivalent: tximport(file.path(directory, sampleTable[, 2]), type = "salmon",
/// tx2gene = tx2gene) with the files named by sampleTable[, 1]; colData is
/// sampleTable[, -(1:2)]
pub fn read_salmon<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
    sample_table: P,
    directory: Q,
    tx2gene: R,
) -> Result<(Txi, SampleMetadata)> {
    let (header, rows) = super::htseq::read_table(sample_table.as_ref())?;
    if header.len() < 2 {
        return Err(DeseqError::InvalidMetadata {
            reason: "Salmon sample table needs at least two columns: sample name and quant.sf path".to_string(),
        });
    }
    if rows.is_empty() {
        return Err(DeseqError::EmptyData { reason: "Salmon sample table lists no samples".to_string() });
    }
    let sample_names: Vec<String> = rows.iter().map(|r| r[0].clone()).collect();
    let files: Vec<std::path::PathBuf> = rows.iter().map(|r| directory.as_ref().join(&r[1])).collect();
    let map = read_tx2gene(tx2gene)?;
    let txi = tximport_salmon(&files, sample_names.clone(), &map)?;

    let mut metadata = SampleMetadata::new(sample_names);
    for (c, name) in header.iter().enumerate().skip(2) {
        metadata.add_condition(name, rows.iter().map(|r| r[c].clone()).collect())?;
    }
    Ok((txi, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_r_round_halves_to_even() {
        for (x, r) in [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (-2.5, -2.0), (2.4999, 2.0), (2.5001, 3.0), (7.0, 7.0)] {
            assert_eq!(r_round(x), r, "round({})", x);
        }
    }

    #[test]
    fn test_r_strtod_matches_r() {
        // R 4.5.3: sprintf("%.17g", as.numeric(x)); the first five are one ulp off the
        // correctly rounded value
        let cases = [
            ("4489.912554", 4489.9125540000005),
            ("77.919041", 77.919040999999993),
            ("0.399506", 0.39950600000000003),
            ("61.246143", 61.246143000000004),
            ("1.546431", 1.5464310000000001),
            ("4235.955", 4235.955),
            ("102.000", 102.0),
            ("-2.5e-3", -0.0025),
            ("1e5", 100000.0),
            ("0", 0.0),
        ];
        for (text, r) in cases {
            assert_eq!(r_strtod(text), Some(r), "{}", text);
        }
        assert_ne!(r_strtod("77.919041"), "77.919041".parse::<f64>().ok());
        assert!(r_strtod("abc").is_none());
    }

    #[test]
    fn test_summarize_to_gene() {
        // tx1, tx2 -> gB; tx3 -> gA; tx4 not in tx2gene
        let tx: Vec<String> = ["tx1", "tx2", "tx3", "tx4"].iter().map(|s| s.to_string()).collect();
        let abundance = array![[10.0, 0.0], [30.0, 0.0], [5.0, 5.0], [1.0, 1.0]];
        let counts = array![[100.5, 0.0], [20.0, 0.0], [7.0, 9.0], [3.0, 3.0]];
        let length = array![[1000.0, 1100.0], [2000.0, 2100.0], [500.0, 400.0], [9.0, 9.0]];
        let map: Vec<(String, String)> = [("TXNAME", "GENEID"), ("tx1", "gB"), ("tx2", "gB"), ("tx3", "gA"), ("tx1", "gA")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let txi = summarize_to_gene(&tx, &abundance, &counts, &length, &map, vec!["s1".into(), "s2".into()]).unwrap();
        assert_eq!(txi.gene_ids, vec!["gA".to_string(), "gB".to_string()]);
        assert_eq!(txi.counts, array![[7.0, 9.0], [120.5, 0.0]]);
        assert_eq!(txi.abundance, array![[5.0, 5.0], [40.0, 0.0]]);
        assert_eq!(txi.length[[0, 0]], 500.0);
        assert_eq!(txi.length[[1, 0]], (10.0 * 1000.0 + 30.0 * 2000.0) / 40.0);
        // zero abundance in s2: the geometric mean of the gene's other samples
        assert!((txi.length[[1, 1]] - txi.length[[1, 0]]).abs() < 1e-9);
    }
}
