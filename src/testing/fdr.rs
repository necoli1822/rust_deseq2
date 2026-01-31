//! P-value adjustment methods for multiple testing correction
//!
//! Currently implements:
//! - Benjamini-Hochberg (BH) FDR correction (DESeq2 default)
//! - Bonferroni family-wise error rate correction
//!
//! R's p.adjust() supports additional methods (holm, hochberg, hommel, BY)
//! which are not currently implemented. BH is the standard for DESeq2.

/// Apply Benjamini-Hochberg FDR correction to p-values
/// R equivalent: p.adjust(method="BH") via pvalueAdjustment() in results.R
///
/// Returns adjusted p-values (q-values) that control the false discovery rate.
pub fn benjamini_hochberg(pvalues: &[f64]) -> Vec<f64> {
    let n = pvalues.len();
    if n == 0 {
        return vec![];
    }

    // Create indices for sorting
    let mut indices: Vec<usize> = (0..n).collect();

    // Sort indices by p-value (handling NaN)
    indices.sort_by(|&a, &b| {
        let pa = pvalues[a];
        let pb = pvalues[b];

        // Put NaN at the end
        if pa.is_nan() && pb.is_nan() {
            std::cmp::Ordering::Equal
        } else if pa.is_nan() {
            std::cmp::Ordering::Greater
        } else if pb.is_nan() {
            std::cmp::Ordering::Less
        } else {
            pa.partial_cmp(&pb).unwrap()
        }
    });

    // Count non-NaN p-values
    let m = pvalues.iter().filter(|p| p.is_finite()).count();

    if m == 0 {
        return vec![f64::NAN; n];
    }

    // Calculate adjusted p-values
    let mut padj = vec![f64::NAN; n];
    let mut cummin = f64::INFINITY;
    let mut rank = m;

    for &i in indices.iter().rev() {
        let p = pvalues[i];

        if p.is_finite() {
            // BH formula: p_adj = p * m / rank
            let adj = (p * m as f64 / rank as f64).min(1.0);
            cummin = cummin.min(adj);
            padj[i] = cummin;
            rank -= 1;
        }
    }

    padj
}

/// Apply Bonferroni correction to p-values
/// R equivalent: p.adjust(method="bonferroni")
///
/// Simple and conservative: multiplies each p-value by the number of tests.
/// Controls the family-wise error rate (FWER) rather than the FDR.
pub fn bonferroni(pvalues: &[f64]) -> Vec<f64> {
    let m = pvalues.iter().filter(|p| p.is_finite()).count();
    if m == 0 {
        return vec![f64::NAN; pvalues.len()];
    }
    pvalues
        .iter()
        .map(|&p| {
            if p.is_nan() {
                f64::NAN
            } else {
                (p * m as f64).min(1.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bh_basic() {
        let pvalues = vec![0.01, 0.04, 0.03, 0.02];
        let padj = benjamini_hochberg(&pvalues);

        // All adjusted p-values should be >= original
        for (p, adj) in pvalues.iter().zip(padj.iter()) {
            assert!(*adj >= *p);
        }

        // Adjusted p-values should be <= 1
        for adj in &padj {
            assert!(*adj <= 1.0);
        }
    }

    #[test]
    fn test_bh_with_nan() {
        let pvalues = vec![0.01, f64::NAN, 0.03, 0.02];
        let padj = benjamini_hochberg(&pvalues);

        assert!(padj[0].is_finite());
        assert!(padj[1].is_nan());
        assert!(padj[2].is_finite());
        assert!(padj[3].is_finite());
    }

    #[test]
    fn test_bh_ordering() {
        let pvalues = vec![0.001, 0.01, 0.05, 0.1];
        let padj = benjamini_hochberg(&pvalues);

        // Adjusted p-values should preserve ordering
        for i in 0..padj.len() - 1 {
            assert!(padj[i] <= padj[i + 1]);
        }
    }
}
