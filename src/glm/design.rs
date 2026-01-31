//! Design matrix creation for GLM

use ndarray::Array2;
use std::collections::HashMap;

use crate::data::{DESeqDataSet, SampleMetadata};
use crate::error::{DeseqError, Result};

/// Information about the design matrix
/// R equivalent: No direct R equivalent (design metadata stored in DESeqDataSet S4 slots)
#[derive(Debug, Clone)]
pub struct DesignInfo {
    /// Names of the coefficients
    pub coef_names: Vec<String>,
    /// Reference level for the design variable
    pub reference_level: String,
    /// All levels of the design variable
    pub levels: Vec<String>,
    /// Factor to column indices mapping
    pub factor_columns: HashMap<String, Vec<usize>>,
    /// Continuous variable to column index mapping
    pub continuous_columns: HashMap<String, usize>,
    /// Interaction to column indices mapping
    pub interaction_columns: HashMap<(String, String), Vec<usize>>,
}

/// Create a design matrix from sample metadata
/// R equivalent: model.matrix() in stats package
///
/// Creates a model matrix for a single categorical variable with
/// treatment contrasts (reference level coded as 0).
pub fn create_design_matrix(
    metadata: &SampleMetadata,
    design_variable: &str,
) -> Result<(Array2<f64>, DesignInfo)> {
    let values = metadata.condition(design_variable).ok_or_else(|| {
        DeseqError::InvalidDesignMatrix {
            reason: format!("Variable '{}' not found in metadata", design_variable),
        }
    })?;

    let levels = metadata.levels(design_variable).ok_or_else(|| {
        DeseqError::InvalidDesignMatrix {
            reason: format!("No levels found for variable '{}'", design_variable),
        }
    })?;

    if levels.len() < 1 {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: "Design variable must have at least 1 level".to_string(),
        });
    }

    let n_samples = metadata.n_samples();

    // Intercept-only model (~1): single column of all 1s
    if levels.len() == 1 {
        let mut design = Array2::zeros((n_samples, 1));
        for i in 0..n_samples {
            design[[i, 0]] = 1.0;
        }
        let info = DesignInfo {
            coef_names: vec!["Intercept".to_string()],
            reference_level: levels[0].clone(),
            levels,
            factor_columns: HashMap::new(),
            continuous_columns: HashMap::new(),
            interaction_columns: HashMap::new(),
        };
        return Ok((design, info));
    }
    let n_coefs = levels.len(); // Intercept + (levels - 1) contrasts

    // Create design matrix
    // Column 0: Intercept (all 1s)
    // Columns 1..n: Indicator for each non-reference level
    let mut design = Array2::zeros((n_samples, n_coefs));

    // Reference level is the first level (alphabetically)
    let reference_level = &levels[0];

    for (i, value) in values.iter().enumerate() {
        // Intercept
        design[[i, 0]] = 1.0;

        // Contrast columns
        for (j, level) in levels.iter().enumerate().skip(1) {
            if value == level {
                design[[i, j]] = 1.0;
            }
        }
    }

    // Create coefficient names
    let mut coef_names = vec!["Intercept".to_string()];
    let mut factor_columns = HashMap::new();
    let mut col_indices = Vec::new();

    for (j, level) in levels.iter().enumerate().skip(1) {
        coef_names.push(format!("{}_{}_vs_{}", design_variable, level, reference_level));
        col_indices.push(j);
    }
    factor_columns.insert(design_variable.to_string(), col_indices);

    let info = DesignInfo {
        coef_names,
        reference_level: reference_level.clone(),
        levels,
        factor_columns,
        continuous_columns: HashMap::new(),
        interaction_columns: HashMap::new(),
    };

    check_full_rank(&design)?;
    Ok((design, info))
}

/// Create an extended design matrix supporting multiple factors, continuous covariates, and interactions
/// R equivalent: getModelMatrix() in core.R
pub fn create_extended_design_matrix(dds: &DESeqDataSet) -> Result<(Array2<f64>, DesignInfo)> {
    let metadata = dds.sample_metadata();
    let n_samples = dds.n_samples();
    let factors = dds.factors();
    let continuous = dds.continuous_vars();
    let interactions = dds.interactions();
    let ref_levels = dds.reference_levels();

    // Calculate total number of columns
    let mut n_cols = 1; // intercept
    let mut coef_names = vec!["Intercept".to_string()];
    let mut factor_columns: HashMap<String, Vec<usize>> = HashMap::new();
    let mut continuous_columns: HashMap<String, usize> = HashMap::new();
    let mut interaction_columns: HashMap<(String, String), Vec<usize>> = HashMap::new();
    let mut factor_levels: HashMap<String, Vec<String>> = HashMap::new();

    // Store reference levels for naming
    let mut factor_ref_levels: HashMap<String, String> = HashMap::new();

    // Process additional factors (covariates) first, then main design variable
    // This matches formula order: ~ covariate1 + covariate2 + design_variable
    let main_var = dds.design_variable();

    // First process covariates/factors
    for factor in factors {
        let levels = metadata.get_levels(factor)?;
        let ref_level = ref_levels
            .get(factor)
            .cloned()
            .unwrap_or_else(|| levels[0].clone());

        factor_ref_levels.insert(factor.clone(), ref_level.clone());

        let non_ref_levels: Vec<String> = levels
            .iter()
            .filter(|l| **l != ref_level)
            .cloned()
            .collect();

        let mut col_indices = Vec::new();
        for level in &non_ref_levels {
            // Use DESeq2-style naming: factor_level_vs_reference
            coef_names.push(format!("{}_{}_vs_{}", factor, level, ref_level));
            col_indices.push(n_cols);
            n_cols += 1;
        }
        factor_columns.insert(factor.clone(), col_indices);
        factor_levels.insert(factor.clone(), non_ref_levels);
    }

    // Then process the main design variable (comes last in formula order)
    {
        let levels = metadata.get_levels(main_var)?;
        let ref_level = ref_levels
            .get(main_var)
            .cloned()
            .unwrap_or_else(|| levels[0].clone());

        factor_ref_levels.insert(main_var.to_string(), ref_level.clone());

        let non_ref_levels: Vec<String> = levels
            .iter()
            .filter(|l| **l != ref_level)
            .cloned()
            .collect();

        let mut col_indices = Vec::new();
        for level in &non_ref_levels {
            coef_names.push(format!("{}_{}_vs_{}", main_var, level, ref_level));
            col_indices.push(n_cols);
            n_cols += 1;
        }
        factor_columns.insert(main_var.to_string(), col_indices);
        factor_levels.insert(main_var.to_string(), non_ref_levels);
    }

    // Process continuous covariates
    for cont in continuous {
        coef_names.push(cont.clone());
        continuous_columns.insert(cont.clone(), n_cols);
        n_cols += 1;
    }

    // Process interactions
    for (f1, f2) in interactions {
        let levels1 = factor_levels.get(f1).cloned().unwrap_or_default();
        let levels2 = factor_levels.get(f2).cloned().unwrap_or_default();

        let mut col_indices = Vec::new();
        for l1 in &levels1 {
            for l2 in &levels2 {
                coef_names.push(format!("{}_{}_x_{}_{}", f1, l1, f2, l2));
                col_indices.push(n_cols);
                n_cols += 1;
            }
        }
        interaction_columns.insert((f1.clone(), f2.clone()), col_indices);
    }

    // Build the design matrix
    let mut design = Array2::zeros((n_samples, n_cols));

    for i in 0..n_samples {
        let mut col = 0;

        // Intercept
        design[[i, col]] = 1.0;
        col += 1;

        // Factor columns (covariates first, then main design variable)
        for factor in factors {
            let sample_value = metadata.get_value(factor, i)?;
            let non_ref_levels = factor_levels.get(factor).unwrap();

            for level in non_ref_levels {
                design[[i, col]] = if sample_value == *level { 1.0 } else { 0.0 };
                col += 1;
            }
        }

        // Main design variable columns
        {
            let sample_value = metadata.get_value(main_var, i)?;
            let non_ref_levels = factor_levels.get(main_var).unwrap();

            for level in non_ref_levels {
                design[[i, col]] = if sample_value == *level { 1.0 } else { 0.0 };
                col += 1;
            }
        }

        // Continuous columns
        for cont in continuous {
            design[[i, col]] = metadata.get_continuous_value(cont, i)?;
            col += 1;
        }

        // Interaction columns
        for (f1, f2) in interactions {
            let val1 = metadata.get_value(f1, i)?;
            let val2 = metadata.get_value(f2, i)?;
            let levels1 = factor_levels.get(f1).unwrap();
            let levels2 = factor_levels.get(f2).unwrap();

            for l1 in levels1 {
                for l2 in levels2 {
                    let val = if val1 == *l1 && val2 == *l2 { 1.0 } else { 0.0 };
                    design[[i, col]] = val;
                    col += 1;
                }
            }
        }
    }

    // Get reference level for main design variable (already computed above)
    let main_levels = metadata.get_levels(main_var)?;
    let reference_level = factor_ref_levels
        .get(main_var)
        .cloned()
        .unwrap_or_else(|| main_levels[0].clone());

    let info = DesignInfo {
        coef_names,
        reference_level,
        levels: main_levels,
        factor_columns,
        continuous_columns,
        interaction_columns,
    };

    check_full_rank(&design)?;
    Ok((design, info))
}

/// Get the coefficient index and sign for a specific contrast
/// R equivalent: getContrast() in results.R
///
/// Returns (coefficient_index, sign) where sign is 1.0 for the standard direction
/// (numerator vs reference) and -1.0 for the reverse direction (reference vs non-reference).
///
/// In R, `results(dds, contrast=c("treatment", "A", "B"))` when B is the reference
/// returns the negative of the `treatment_A_vs_B` coefficient (i.e., flipped sign).
/// We support this by returning sign = -1.0 when the user's numerator is the reference level.
pub fn get_contrast_index(info: &DesignInfo, numerator: &str, denominator: &str) -> Result<(usize, f64)> {
    // Standard case: denominator is the reference level
    // Coefficient encodes "numerator vs reference" -> sign = +1
    if denominator == info.reference_level {
        let expected_name = format!(
            "{}_vs_{}",
            numerator, denominator
        );

        for (i, name) in info.coef_names.iter().enumerate() {
            if name.ends_with(&expected_name) {
                return Ok((i, 1.0));
            }
        }

        return Err(DeseqError::InvalidContrast {
            reason: format!("Contrast '{} vs {}' not found", numerator, denominator),
        });
    }

    // Reverse case: numerator is the reference level
    // R handles this by negating the coefficient for denominator_vs_reference
    if numerator == info.reference_level {
        let expected_name = format!(
            "{}_vs_{}",
            denominator, numerator
        );

        for (i, name) in info.coef_names.iter().enumerate() {
            if name.ends_with(&expected_name) {
                return Ok((i, -1.0));
            }
        }
    }

    // Neither is the reference level -- for multi-level factors, neither numerator
    // nor denominator is the reference. This requires a contrast vector approach.
    Err(DeseqError::InvalidContrast {
        reason: format!(
            "For contrast '{} vs {}': one of the levels must be the reference level '{}'. \
             Use --reference FACTOR={} to set the denominator as reference, \
             or use the extended contrast specification.",
            numerator, denominator, info.reference_level, denominator
        ),
    })
}

/// Check if a design matrix is full rank using QR decomposition with column pivoting.
/// R equivalent: checkFullRank() in DESeq2 (via modelMatrixType)
///
/// Uses Householder QR with column pivoting. The rank is determined by counting
/// diagonal elements of R whose absolute value exceeds the tolerance
/// `max(nrow, ncol) * eps * max(|diag(R)|)`, matching R's `qr()` behavior.
///
/// Returns `Ok(())` if full rank, or `Err(DeseqError::InvalidDesignMatrix)` with a
/// specific message distinguishing zero columns from linear combinations.
pub fn check_full_rank(matrix: &Array2<f64>) -> Result<()> {
    let nrow = matrix.nrows();
    let ncol = matrix.ncols();

    if nrow == 0 || ncol == 0 {
        return Err(DeseqError::InvalidDesignMatrix {
            reason: "Design matrix has zero rows or columns".to_string(),
        });
    }

    // Compute rank via Householder QR with column pivoting
    let rank = qr_rank(matrix);

    if rank < ncol {
        // Check if any column is entirely zero
        let has_zero_column = (0..ncol).any(|j| {
            matrix.column(j).iter().all(|&v| v == 0.0)
        });

        if has_zero_column {
            return Err(DeseqError::InvalidDesignMatrix {
                reason: "the model matrix is not full rank, so the model cannot be fit as specified.\n  \
                    Levels or combinations of levels without any samples have resulted in\n  \
                    column(s) of zeros in the model matrix."
                    .to_string(),
            });
        } else {
            return Err(DeseqError::InvalidDesignMatrix {
                reason: "the model matrix is not full rank, so the model cannot be fit as specified.\n  \
                    One or more variables or interaction terms in the design formula are linear\n  \
                    combinations of the others and must be removed."
                    .to_string(),
            });
        }
    }

    Ok(())
}

/// Compute the numerical rank of a matrix using Householder QR with column pivoting.
///
/// Returns the number of diagonal elements of R whose absolute value exceeds
/// `max(nrow, ncol) * f64::EPSILON * max(|diag(R)|)`.
fn qr_rank(matrix: &Array2<f64>) -> usize {
    let nrow = matrix.nrows();
    let ncol = matrix.ncols();
    let k = nrow.min(ncol);

    // Work on a mutable copy
    let mut r = matrix.to_owned();

    // Column norms squared (for pivoting)
    let mut col_norms_sq: Vec<f64> = (0..ncol)
        .map(|j| r.column(j).iter().map(|&v| v * v).sum())
        .collect();

    // Pivot permutation (tracks which original column is in each position)
    let mut piv: Vec<usize> = (0..ncol).collect();

    for step in 0..k {
        // Column pivoting: find column with largest remaining norm
        let mut best_col = step;
        let mut best_norm = col_norms_sq[step];
        for j in (step + 1)..ncol {
            if col_norms_sq[j] > best_norm {
                best_norm = col_norms_sq[j];
                best_col = j;
            }
        }

        // Swap columns if needed
        if best_col != step {
            for i in 0..nrow {
                let tmp = r[[i, step]];
                r[[i, step]] = r[[i, best_col]];
                r[[i, best_col]] = tmp;
            }
            col_norms_sq.swap(step, best_col);
            piv.swap(step, best_col);
        }

        // Compute Householder reflection for column `step`
        // Extract the sub-column from row `step` downward
        let mut alpha = 0.0f64;
        for i in step..nrow {
            alpha += r[[i, step]] * r[[i, step]];
        }
        alpha = alpha.sqrt();

        if alpha < f64::EPSILON * 1e3 {
            // Remaining columns are effectively zero; rank determined by earlier columns
            break;
        }

        // Choose sign to avoid cancellation
        if r[[step, step]] > 0.0 {
            alpha = -alpha;
        }

        let v0 = r[[step, step]] - alpha;
        r[[step, step]] = alpha;

        // Normalize the Householder vector (stored in-place below the diagonal)
        // v = [v0, r[step+1..nrow, step]]
        // tau = -2 / (v^T v)
        let mut v_norm_sq = v0 * v0;
        for i in (step + 1)..nrow {
            v_norm_sq += r[[i, step]] * r[[i, step]];
        }

        if v_norm_sq.abs() < f64::MIN_POSITIVE {
            continue;
        }

        let tau = 2.0 / v_norm_sq;

        // Apply Householder transformation to remaining columns
        for j in (step + 1)..ncol {
            // dot = v^T * r[:, j]
            let mut dot = v0 * r[[step, j]];
            for i in (step + 1)..nrow {
                dot += r[[i, step]] * r[[i, j]];
            }

            let scale = tau * dot;

            // r[:, j] -= scale * v
            r[[step, j]] -= scale * v0;
            for i in (step + 1)..nrow {
                r[[i, j]] -= scale * r[[i, step]];
            }
        }

        // Store the Householder vector below the diagonal (for completeness, not needed for rank)
        // Zero out the sub-diagonal of the current column in R view
        // (we keep them for the Householder vector but rank only needs diag)

        // Update column norms (downdate)
        for j in (step + 1)..ncol {
            col_norms_sq[j] -= r[[step, j]] * r[[step, j]];
            // Guard against negative due to floating-point
            if col_norms_sq[j] < 0.0 {
                col_norms_sq[j] = 0.0;
            }
        }
    }

    // Determine rank from diagonal of R
    // Tolerance: max(nrow, ncol) * eps * max(|diag(R)|)
    let max_dim = nrow.max(ncol) as f64;
    let max_abs_diag = (0..k)
        .map(|i| r[[i, i]].abs())
        .fold(0.0f64, f64::max);

    let tol = max_dim * f64::EPSILON * max_abs_diag;

    (0..k).filter(|&i| r[[i, i]].abs() > tol).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_design_matrix_creation() {
        let mut metadata = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ]);
        metadata
            .add_condition(
                "treatment",
                vec![
                    "control".to_string(),
                    "control".to_string(),
                    "treated".to_string(),
                    "treated".to_string(),
                ],
            )
            .unwrap();

        let (design, info) = create_design_matrix(&metadata, "treatment").unwrap();

        assert_eq!(design.dim(), (4, 2));
        assert_eq!(info.coef_names.len(), 2);
        assert_eq!(info.reference_level, "control");

        // Check design matrix values
        // s1, s2 are control (reference) -> [1, 0]
        // s3, s4 are treated -> [1, 1]
        assert_eq!(design[[0, 0]], 1.0);
        assert_eq!(design[[0, 1]], 0.0);
        assert_eq!(design[[2, 0]], 1.0);
        assert_eq!(design[[2, 1]], 1.0);
    }

    #[test]
    fn test_three_level_design() {
        let mut metadata = SampleMetadata::new(vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
            "s5".to_string(),
            "s6".to_string(),
        ]);
        metadata
            .add_condition(
                "dose",
                vec![
                    "high".to_string(),
                    "high".to_string(),
                    "low".to_string(),
                    "low".to_string(),
                    "medium".to_string(),
                    "medium".to_string(),
                ],
            )
            .unwrap();

        let (design, info) = create_design_matrix(&metadata, "dose").unwrap();

        assert_eq!(design.dim(), (6, 3)); // Intercept + 2 contrasts
        assert_eq!(info.levels.len(), 3);
        assert_eq!(info.reference_level, "high"); // Alphabetically first
    }

    #[test]
    fn test_check_full_rank_valid() {
        // Standard full-rank design matrix: intercept + treatment indicator
        let matrix = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 0.0, // control
                1.0, 0.0, // control
                1.0, 1.0, // treated
                1.0, 1.0, // treated
            ],
        )
        .unwrap();
        assert!(check_full_rank(&matrix).is_ok());
    }

    #[test]
    fn test_check_full_rank_identity() {
        // 3x3 identity matrix is full rank
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        assert!(check_full_rank(&matrix).is_ok());
    }

    #[test]
    fn test_check_full_rank_zero_column() {
        // Matrix with an all-zero column (level with no samples)
        let matrix = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, 0.0, // third column is all zeros
                1.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        let err = check_full_rank(&matrix).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("column(s) of zeros"), "Expected zero-column message, got: {}", msg);
    }

    #[test]
    fn test_check_full_rank_linear_combination() {
        // Column 3 = Column 1 + Column 2 (linear combination)
        let matrix = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 2.0,
                1.0, 1.0, 2.0,
            ],
        )
        .unwrap();
        let err = check_full_rank(&matrix).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("linear") && msg.contains("combinations"),
            "Expected linear-combination message, got: {}",
            msg
        );
    }

    #[test]
    fn test_check_full_rank_single_column() {
        // Intercept-only model: single column of ones, full rank
        let matrix = Array2::from_shape_vec(
            (3, 1),
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        assert!(check_full_rank(&matrix).is_ok());
    }

    #[test]
    fn test_check_full_rank_wide_matrix() {
        // More columns than rows: cannot be full column rank
        let matrix = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 1.0,
                0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        // Rank is at most 2 but ncol=3, so not full rank
        assert!(check_full_rank(&matrix).is_err());
    }
}
