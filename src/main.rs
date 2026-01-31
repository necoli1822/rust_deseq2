//! rust_deseq2 command-line interface

use clap::Parser;
use log::{info, LevelFilter};

use rust_deseq2::cli::{Cli, Commands};
use rust_deseq2::filter::{calculate_cooks_distance, default_cooks_cutoff, filter_by_cooks_with_counts, replace_outliers};
use rust_deseq2::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Find the first non-flag argument (potential subcommand)
    let first_positional = args.iter().skip(1).find(|a| !a.starts_with('-'));
    let subcommands = ["run", "normalize", "vst", "rlog", "help"];
    let has_subcommand = first_positional
        .map_or(false, |a| subcommands.contains(&a.as_str()));

    if !has_subcommand {
        // No subcommand — handle top-level help/version manually
        if args.len() == 1 {
            print_no_args();
            return;
        }
        if args.iter().any(|a| a == "--help") {
            print_long_help();
            return;
        }
        if args.iter().any(|a| a == "-h") {
            print_short_help();
            return;
        }
        if args.iter().any(|a| a == "-V" || a == "--version") {
            println!("rust_deseq2 {}", VERSION);
            return;
        }
        // Unknown flags without subcommand — show hint
        print_no_args();
        return;
    }

    let cli = Cli::parse();

    // Set up logging
    let log_level = if cli.verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp(None)
        .init();

    let result = match cli.command {
        Some(Commands::Run {
            counts,
            metadata,
            design,
            covariate,
            continuous,
            reference,
            test,
            reduced,
            numerator,
            denominator,
            output,
            alpha,
            shrinkage,
            shrinkage_method,
            sf_type,
            min_replicates,
            replace_outliers,
            lfc_threshold,
            fit_type,
            use_t,
            maxit,
            beta_tol,
            min_disp,
            disp_tol,
            kappa_0,
            outlier_sd,
            trim,
            upper_quantile,
            threads,
            no_independent_filtering,
            cooks_cutoff,
        }) => run_analysis(
            &counts,
            &metadata,
            &design,
            &covariate,
            &continuous,
            &reference,
            &test,
            reduced.as_deref(),
            numerator.as_deref(),
            denominator.as_deref(),
            &output,
            alpha,
            shrinkage,
            &shrinkage_method,
            &sf_type,
            min_replicates,
            replace_outliers,
            lfc_threshold,
            &fit_type,
            use_t,
            maxit,
            beta_tol,
            min_disp,
            disp_tol,
            kappa_0,
            outlier_sd,
            trim,
            upper_quantile,
            threads,
            no_independent_filtering,
            cooks_cutoff,
        ),
        Some(Commands::Normalize {
            counts,
            output,
            method,
        }) => run_normalize(&counts, &output, &method),
        Some(Commands::Vst {
            counts,
            metadata,
            design,
            output,
            method,
            blind,
        }) => run_vst(&counts, &metadata, &design, &output, &method, blind),
        Some(Commands::Rlog {
            counts,
            metadata,
            design,
            output,
            blind,
        }) => run_rlog(&counts, &metadata, &design, &output, blind),
        None => {
            // Should not reach here (handled above), but just in case
            print_no_args();
            return;
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Custom help output
// ---------------------------------------------------------------------------

fn print_no_args() {
    println!("rust_deseq2 v{}", VERSION);
    println!("Run `rust_deseq2 -h` for usage or `rust_deseq2 --help` for detailed information.");
}

fn print_short_help() {
    println!("rust_deseq2 v{}", VERSION);
    println!();
    println!("Usage: rust_deseq2 <COMMAND> [OPTIONS]");
    println!();
    println!("Commands:");
    println!("  run        Run full DESeq2 analysis");
    println!("  normalize  Normalize count data only");
    println!("  vst        Variance Stabilizing Transformation");
    println!("  rlog       Regularized Log Transformation");
    println!();
    println!("Run `rust_deseq2 <COMMAND> -h` for command-specific options.");
}

fn print_long_help() {
    println!("rust_deseq2 v{}", VERSION);
    println!("Pure Rust implementation of DESeq2 differential expression analysis");
    println!();
    println!("Usage: rust_deseq2 <COMMAND> [OPTIONS]");
    println!();
    println!("Commands:");
    println!("  run        Run full DESeq2 analysis");
    println!("               - Wald test or Likelihood Ratio Test (LRT)");
    println!("               - LFC shrinkage (normal, apeglm, ashr)");
    println!("               - Cook's distance outlier detection and replacement");
    println!("               - Multi-factor designs with covariates");
    println!("               - Dispersion fit types: parametric, local, mean");
    println!("               - t-distribution p-values for small sample sizes");
    println!("  normalize  Normalize count data using median-of-ratios method");
    println!("  vst        Variance Stabilizing Transformation");
    println!("  rlog       Regularized Log Transformation");
    println!();
    println!("Global Options:");
    println!("  -v, --verbose    Enable verbose output");
    println!("  -h               Print short help");
    println!("      --help       Print detailed help");
    println!("  -V, --version    Print version");
    println!();
    println!("Examples:");
    println!("  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\");
    println!("    --numerator treated --denominator control -o results.csv");
    println!();
    println!("  rust_deseq2 run -c counts.csv -m metadata.csv -d treatment \\");
    println!("    --covariate batch --test lrt --reduced \"~batch\" -o results.csv");
    println!();
    println!("  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\");
    println!("    --numerator treated --denominator control --fit-type local --use-t");
    println!();
    println!("  rust_deseq2 vst -c counts.csv -m metadata.csv -d condition --blind");
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn run_analysis(
    counts_path: &str,
    metadata_path: &str,
    design: &str,
    covariates: &[String],
    continuous_vars: &[String],
    reference_levels: &[String],
    test_type: &str,
    reduced_formula: Option<&str>,
    numerator: Option<&str>,
    denominator: Option<&str>,
    output_path: &str,
    alpha: f64,
    shrinkage: bool,
    shrinkage_method: &str,
    sf_type: &str,
    min_replicates_for_replace: usize,
    do_replace_outliers: bool,
    lfc_threshold: f64,
    fit_type: &str,
    use_t: bool,
    maxit: usize,
    beta_tol: f64,
    min_disp: f64,
    disp_tol: f64,
    kappa_0: f64,
    outlier_sd: f64,
    trim: f64,
    upper_quantile: f64,
    threads: usize,
    no_independent_filtering: bool,
    cooks_cutoff_override: Option<f64>,
) -> Result<()> {
    use rust_deseq2::testing::likelihood_ratio_test;

    // Configure thread pool
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    // Validate test type early
    if test_type != "wald" && test_type != "lrt" {
        return Err(DeseqError::InvalidInput {
            reason: format!("Unknown test type '{}'. Use 'wald' or 'lrt'.", test_type),
        });
    }

    // For Wald test, numerator and denominator are required
    if test_type == "wald" {
        if numerator.is_none() || denominator.is_none() {
            return Err(DeseqError::InvalidInput {
                reason: "--numerator and --denominator are required for Wald test".to_string(),
            });
        }
    }

    info!("Loading count matrix from: {}", counts_path);
    let counts = read_count_matrix(counts_path)?;
    info!("  {} genes, {} samples", counts.n_genes(), counts.n_samples());

    info!("Loading metadata from: {}", metadata_path);
    let metadata = read_metadata(metadata_path)?;

    // Cross-validate sample IDs between counts and metadata
    {
        let count_ids = counts.sample_ids();
        let meta_ids = metadata.sample_ids();
        if count_ids.len() != meta_ids.len() {
            return Err(DeseqError::InvalidMetadata {
                reason: format!(
                    "Sample count mismatch: count matrix has {} samples, metadata has {} samples",
                    count_ids.len(), meta_ids.len()
                ),
            });
        }
        let missing_in_meta: Vec<&str> = count_ids.iter()
            .filter(|id| !meta_ids.contains(id))
            .map(|s| s.as_str())
            .collect();
        let missing_in_counts: Vec<&str> = meta_ids.iter()
            .filter(|id| !count_ids.contains(id))
            .map(|s| s.as_str())
            .collect();
        if !missing_in_meta.is_empty() || !missing_in_counts.is_empty() {
            let mut msg = String::from("Sample IDs do not match between counts and metadata.");
            if !missing_in_meta.is_empty() {
                msg.push_str(&format!(" In counts but not metadata: {:?}.", missing_in_meta));
            }
            if !missing_in_counts.is_empty() {
                msg.push_str(&format!(" In metadata but not counts: {:?}.", missing_in_counts));
            }
            return Err(DeseqError::InvalidMetadata { reason: msg });
        }
    }

    // Build DESeqDataSet with multi-factor design support
    let mut dds = if covariates.is_empty() && continuous_vars.is_empty() && reference_levels.is_empty() {
        // Simple design: ~ design_variable (alphabetical reference)
        info!("Creating DESeqDataSet with design: ~ {}", design);
        DESeqDataSet::new(counts, metadata, design)?
    } else {
        // Multi-factor design: ~ covariate1 + covariate2 + ... + design_variable
        let formula_parts: Vec<&str> = covariates
            .iter()
            .map(|s| s.as_str())
            .chain(continuous_vars.iter().map(|s| s.as_str()))
            .chain(std::iter::once(design))
            .collect();
        info!("Creating DESeqDataSet with design: ~ {}", formula_parts.join(" + "));

        let mut builder = DESeqDataSet::builder()
            .counts(counts)
            .metadata(metadata)
            .main_effect(design);

        // Add categorical covariates (batch effects)
        for cov in covariates {
            builder = builder.factor(cov);
        }

        // Add continuous covariates
        for cont in continuous_vars {
            builder = builder.continuous(cont);
        }

        // Parse reference levels (format: factor=level)
        for ref_spec in reference_levels {
            if let Some((factor, level)) = ref_spec.split_once('=') {
                builder = builder.reference_level(factor, level);
            } else {
                return Err(DeseqError::InvalidInput {
                    reason: format!(
                        "Invalid reference format '{}'. Use: factor=level",
                        ref_spec
                    ),
                });
            }
        }

        builder.build()?
    };

    let sf_method = match sf_type {
        "poscounts" => SizeFactorMethod::PosCounts,
        "iterate" => SizeFactorMethod::Iterate,
        _ => SizeFactorMethod::Ratio,
    };
    info!("Estimating size factors (method: {})...", sf_type);
    estimate_size_factors(&mut dds, sf_method)?;

    let trend_method = match fit_type {
        "local" => TrendFitMethod::Local,
        "mean" => TrendFitMethod::Mean,
        _ => TrendFitMethod::Parametric,
    };
    let disp_params = DispersionParams {
        min_disp,
        disp_tol,
        kappa_0,
        maxit,
        outlier_sd,
    };
    info!("Estimating dispersions (fitType: {})...", fit_type);
    estimate_dispersions(&mut dds, trend_method, &disp_params)?;

    let glm_params = GlmFitParams {
        maxit,
        beta_tol,
    };
    info!("Fitting GLM...");
    let design_info = fit_glm(&mut dds, &glm_params)?;

    // -----------------------------------------------------------------------
    // Step 3: Run Wald/LRT test FIRST (before Cook's replacement)
    // R DESeq2 order: estimateDispersions -> nbinomWaldTest/nbinomLRT -> replaceOutliers -> refit
    // -----------------------------------------------------------------------
    let mut results = if test_type == "lrt" {
        let design_full = dds.design_matrix().ok_or_else(|| DeseqError::InvalidInput {
            reason: "Design matrix not available after GLM fit".to_string(),
        })?.clone();

        let design_reduced = build_reduced_design(&dds, reduced_formula, covariates, continuous_vars)?;

        let df_full = design_full.ncols();
        let df_reduced = design_reduced.ncols();
        info!(
            "Performing LRT (full: {} params, reduced: {} params, df: {})...",
            df_full,
            df_reduced,
            df_full - df_reduced
        );

        likelihood_ratio_test(&dds, &design_full, &design_reduced)?
    } else {
        let num = numerator.unwrap();
        let den = denominator.unwrap();
        info!("Performing Wald test for {} vs {}...", num, den);
        let contrast = Contrast {
            variable: design.to_string(),
            numerator: num.to_string(),
            denominator: den.to_string(),
        };

        wald_test(&dds, &design_info, contrast, alpha, use_t, lfc_threshold)?
    };

    // -----------------------------------------------------------------------
    // Step 4: Cook's outlier replacement + refit (AFTER initial test)
    // R DESeq2: replaceOutliers then refitWithoutOutliers, then re-run test
    // -----------------------------------------------------------------------
    let min_replicates = min_replicates_for_replace;
    let mut replaceable_samples: Vec<bool> = vec![false; dds.n_samples()];

    // Save ORIGINAL Cook's distances BEFORE any replacement (Bug 2 fix)
    // These will be used for maxCooks filtering later.
    let original_cooks = if do_replace_outliers {
        Some(calculate_cooks_distance(&dds)?)
    } else {
        None
    };

    if do_replace_outliers {
        let cooks = original_cooks.as_ref().unwrap();
        let n_coefs = dds.design_matrix().map(|m| m.ncols()).unwrap_or(2);
        let cutoff = cooks_cutoff_override.unwrap_or_else(|| default_cooks_cutoff(dds.n_samples(), n_coefs));

        // Determine which samples are "replaceable" (in cells with >= minReplicates)
        let cells: Vec<String> = if let Some(design_matrix) = dds.design_matrix() {
            (0..dds.n_samples())
                .map(|j| {
                    let row: Vec<String> = (0..design_matrix.ncols())
                        .map(|k| format!("{:.6}", design_matrix[[j, k]]))
                        .collect();
                    row.join("_")
                })
                .collect()
        } else {
            dds.sample_metadata()
                .condition(design)
                .cloned()
                .unwrap_or_else(|| vec!["unknown".to_string(); dds.n_samples()])
        };

        let mut cell_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for cell in &cells {
            *cell_counts.entry(cell.clone()).or_insert(0) += 1;
        }

        for (i, cell) in cells.iter().enumerate() {
            if cell_counts.get(cell).copied().unwrap_or(0) >= min_replicates {
                replaceable_samples[i] = true;
            }
        }

        let (modified_genes, flagged_genes) = replace_outliers(&mut dds, cooks, cutoff, min_replicates, trim)?;

        if !modified_genes.is_empty() {
            info!(
                "Replaced outlier counts in {} genes, {} genes flagged for refit (minReplicatesForReplace = {})",
                modified_genes.len(),
                flagged_genes.len(),
                min_replicates
            );

            // Bug 5: Check for genes that became all-zero after replacement
            let counts_matrix = dds.counts().counts();
            let n_samples = dds.n_samples();
            let new_all_zero: Vec<usize> = flagged_genes.iter().copied().filter(|&gene_idx| {
                (0..n_samples).all(|j| counts_matrix[[gene_idx, j]] == 0.0)
            }).collect();

            // Refit dispersions + GLM for replaced genes
            rust_deseq2::dispersion::refit_without_outliers(&mut dds, &flagged_genes)?;

            // Bug 4: Re-run Wald/LRT test after refit
            info!("Re-running {} test after outlier replacement...", test_type);
            results = if test_type == "lrt" {
                let design_full = dds.design_matrix().ok_or_else(|| DeseqError::InvalidInput {
                    reason: "Design matrix not available after refit".to_string(),
                })?.clone();
                let design_reduced = build_reduced_design(&dds, reduced_formula, covariates, continuous_vars)?;
                likelihood_ratio_test(&dds, &design_full, &design_reduced)?
            } else {
                let num = numerator.unwrap();
                let den = denominator.unwrap();
                let contrast = Contrast {
                    variable: design.to_string(),
                    numerator: num.to_string(),
                    denominator: den.to_string(),
                };
                wald_test(&dds, &design_info, contrast, alpha, use_t, lfc_threshold)?
            };

            // Bug 5: Set results to NA for genes that became all-zero after replacement
            if !new_all_zero.is_empty() {
                info!("{} genes became all-zero after outlier replacement, setting results to NA", new_all_zero.len());
                for &gene_idx in &new_all_zero {
                    results.log2_fold_changes[gene_idx] = f64::NAN;
                    results.lfc_se[gene_idx] = f64::NAN;
                    results.stat[gene_idx] = f64::NAN;
                    results.pvalues[gene_idx] = f64::NAN;
                    results.padj[gene_idx] = f64::NAN;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Cook's filtering on results using ORIGINAL (pre-replacement) Cook's distances
    // -----------------------------------------------------------------------
    // Check if any cell has 3+ samples for Cook's filtering
    let apply_cooks_filtering = if let Some(design_matrix) = dds.design_matrix() {
        let cells: Vec<String> = (0..dds.n_samples())
            .map(|j| {
                let row: Vec<String> = (0..design_matrix.ncols())
                    .map(|k| format!("{:.6}", design_matrix[[j, k]]))
                    .collect();
                row.join("_")
            })
            .collect();

        let mut cell_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for cell in &cells {
            *cell_counts.entry(cell.clone()).or_insert(0) += 1;
        }

        cell_counts.values().any(|&count| count >= 3)
    } else {
        let cells = dds.sample_metadata()
            .condition(design)
            .cloned()
            .unwrap_or_else(|| vec!["unknown".to_string(); dds.n_samples()]);

        let mut cell_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for cell in &cells {
            *cell_counts.entry(cell.clone()).or_insert(0) += 1;
        }

        cell_counts.values().any(|&count| count >= 3)
    };

    if apply_cooks_filtering {
        // Bug 2 fix: Use ORIGINAL Cook's distances (pre-replacement) for filtering.
        // If outlier replacement was done, use saved original_cooks; otherwise calculate fresh.
        let mut cooks = if let Some(ref orig) = original_cooks {
            orig.clone()
        } else {
            calculate_cooks_distance(&dds)?
        };
        let n_coefs = dds.design_matrix().map(|m| m.ncols()).unwrap_or(2);
        let cooks_cutoff = cooks_cutoff_override.unwrap_or_else(|| default_cooks_cutoff(dds.n_samples(), n_coefs));

        if do_replace_outliers && replaceable_samples.iter().any(|&r| r) {
            let n_genes = cooks.nrows();
            for gene_idx in 0..n_genes {
                for (sample_idx, &is_replaceable) in replaceable_samples.iter().enumerate() {
                    if is_replaceable {
                        cooks[[gene_idx, sample_idx]] = 0.0;
                    }
                }
            }
        }

        let n_levels = dds.sample_metadata().levels(design)
            .map(|levels| levels.len())
            .unwrap_or(0);
        let is_two_level_design = covariates.is_empty() && continuous_vars.is_empty() && n_levels == 2;
        let counts = dds.counts().counts().to_owned();
        filter_by_cooks_with_counts(&mut results, &cooks, cooks_cutoff, Some(&counts), is_two_level_design);
    }

    // Apply independent filtering (or skip if --no-independent-filtering)
    if no_independent_filtering {
        // Just apply BH adjustment to all genes without filtering
        let pvalues = &results.pvalues;
        results.padj = rust_deseq2::testing::benjamini_hochberg(pvalues);
    } else {
        independent_filtering(&mut results, alpha);
    }

    if shrinkage && test_type == "wald" {
        match shrinkage_method {
            "apeglm" => {
                info!("Applying apeglm LFC shrinkage...");
                use rust_deseq2::shrinkage::{shrink_lfc_apeglm, ApeglmParams};
                let params = ApeglmParams::default();
                shrink_lfc_apeglm(&dds, &mut results, 1, &params)?;
            }
            "ashr" => {
                info!("Applying ashr LFC shrinkage...");
                use rust_deseq2::shrinkage::{apply_ashr_shrinkage, AshrParams};
                let params = AshrParams::default();
                apply_ashr_shrinkage(&mut results, &params);
            }
            _ => {
                info!("Applying normal LFC shrinkage...");
                shrink_lfc_normal(&dds, &design_info, &mut results, upper_quantile);
            }
        }
    }

    info!("Writing results to: {}", output_path);
    write_results(output_path, &results.gene_ids, &results)?;

    // Print summary
    let summary = results.summary(alpha);
    println!("\n{}", summary);

    Ok(())
}

/// Build a reduced design matrix from the --reduced formula.
fn build_reduced_design(
    dds: &DESeqDataSet,
    reduced_formula: Option<&str>,
    _covariates: &[String],
    _continuous_vars: &[String],
) -> Result<ndarray::Array2<f64>> {
    use ndarray::Array2;

    let n_samples = dds.n_samples();
    let metadata = dds.sample_metadata();

    let formula = reduced_formula.unwrap_or("~1");
    let formula_trimmed = formula.trim();

    let formula_body = if formula_trimmed.starts_with('~') {
        formula_trimmed[1..].trim()
    } else {
        formula_trimmed
    };

    if formula_body == "1" || formula_body.is_empty() {
        info!("Reduced model: ~1 (intercept only)");
        return Ok(Array2::from_elem((n_samples, 1), 1.0));
    }

    let variables: Vec<&str> = formula_body
        .split('+')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if variables.is_empty() {
        info!("Reduced model: ~1 (intercept only)");
        return Ok(Array2::from_elem((n_samples, 1), 1.0));
    }

    info!("Reduced model: ~ {}", variables.join(" + "));

    let mut n_cols = 1; // intercept
    let mut var_info: Vec<(&str, Vec<String>)> = Vec::new();

    for &var in &variables {
        let levels = metadata.get_levels(var).map_err(|_| DeseqError::InvalidInput {
            reason: format!(
                "Variable '{}' in reduced formula not found in metadata",
                var
            ),
        })?;
        let non_ref_levels: Vec<String> = levels.into_iter().skip(1).collect();
        n_cols += non_ref_levels.len();
        var_info.push((var, non_ref_levels));
    }

    let mut design = Array2::zeros((n_samples, n_cols));
    for i in 0..n_samples {
        design[[i, 0]] = 1.0; // intercept

        let mut col = 1;
        for (var, non_ref_levels) in &var_info {
            let sample_value = metadata.get_value(var, i)?;
            for level in non_ref_levels {
                design[[i, col]] = if sample_value == *level { 1.0 } else { 0.0 };
                col += 1;
            }
        }
    }

    Ok(design)
}

fn run_normalize(counts_path: &str, output_path: &str, method: &str) -> Result<()> {
    info!("Loading count matrix from: {}", counts_path);
    let counts = read_count_matrix(counts_path)?;
    info!("  {} genes, {} samples", counts.n_genes(), counts.n_samples());

    let metadata = SampleMetadata::new(counts.sample_ids().to_vec());

    let mut dds = DESeqDataSet::new(counts, metadata, "dummy")
        .map_err(|_| DeseqError::InvalidMetadata {
            reason: "Failed to create dataset for normalization".to_string(),
        })?;

    let size_method = match method {
        "ratio" => SizeFactorMethod::Ratio,
        "poscounts" => SizeFactorMethod::PosCounts,
        "iterate" => SizeFactorMethod::Iterate,
        _ => {
            return Err(DeseqError::InvalidMetadata {
                reason: format!("Unknown normalization method: {}", method),
            });
        }
    };

    info!("Estimating size factors using {} method...", method);
    estimate_size_factors(&mut dds, size_method)?;

    info!("Writing normalized counts to: {}", output_path);

    let norm_counts = dds.normalized_counts().ok_or_else(|| DeseqError::EmptyData {
        reason: "Normalized counts not available".to_string(),
    })?;

    let mut file = std::fs::File::create(output_path)?;
    use std::io::Write;

    let sample_ids = dds.counts().sample_ids();
    writeln!(file, "gene_id\t{}", sample_ids.join("\t"))?;

    let gene_ids = dds.counts().gene_ids();
    for (i, gene_id) in gene_ids.iter().enumerate() {
        let row: Vec<String> = (0..dds.n_samples())
            .map(|j| format!("{:.4}", norm_counts[[i, j]]))
            .collect();
        writeln!(file, "{}\t{}", gene_id, row.join("\t"))?;
    }

    info!("Done!");
    Ok(())
}

fn run_vst(
    counts_path: &str,
    metadata_path: &str,
    design: &str,
    output_path: &str,
    method: &str,
    blind: bool,
) -> Result<()> {
    use rust_deseq2::transform::{vst, VstMethod};

    info!("Loading count matrix from: {}", counts_path);
    let counts = read_count_matrix(counts_path)?;
    info!("  {} genes, {} samples", counts.n_genes(), counts.n_samples());

    info!("Loading metadata from: {}", metadata_path);
    let metadata = read_metadata(metadata_path)?;

    info!("Creating DESeqDataSet with design: ~ {}", design);
    let mut dds = DESeqDataSet::new(counts, metadata, design)?;

    info!("Estimating size factors...");
    estimate_size_factors(&mut dds, SizeFactorMethod::Ratio)?;

    let vst_method = match method {
        "parametric" => VstMethod::Parametric,
        "mean" => VstMethod::Mean,
        "local" => VstMethod::Local,
        _ => {
            return Err(DeseqError::InvalidInput {
                reason: format!("Unknown VST method: {}. Use: parametric, mean, or local", method),
            });
        }
    };

    info!(
        "Applying VST transformation (method: {}, blind: {})...",
        method, blind
    );

    if blind {
        use ndarray::Array2;
        let n_samples = dds.n_samples();
        let intercept_design = Array2::from_elem((n_samples, 1), 1.0);
        dds.set_design_matrix(intercept_design)?;
    }
    estimate_dispersions(&mut dds, TrendFitMethod::Parametric, &DispersionParams::default())?;

    let vst_result = vst(&dds, vst_method, blind)?;

    info!("Writing VST-transformed data to: {}", output_path);

    let mut file = std::fs::File::create(output_path)?;
    use std::io::Write;

    let sample_ids = dds.counts().sample_ids();
    writeln!(file, "gene_id\t{}", sample_ids.join("\t"))?;

    let gene_ids = dds.counts().gene_ids();
    for (i, gene_id) in gene_ids.iter().enumerate() {
        let row: Vec<String> = (0..dds.n_samples())
            .map(|j| format!("{:.6}", vst_result.data[[i, j]]))
            .collect();
        writeln!(file, "{}\t{}", gene_id, row.join("\t"))?;
    }

    info!("Done! VST transformation complete.");
    Ok(())
}

fn run_rlog(
    counts_path: &str,
    metadata_path: &str,
    design: &str,
    output_path: &str,
    blind: bool,
) -> Result<()> {
    use rust_deseq2::transform::rlog;

    info!("Loading count matrix from: {}", counts_path);
    let counts = read_count_matrix(counts_path)?;
    info!("  {} genes, {} samples", counts.n_genes(), counts.n_samples());

    info!("Loading metadata from: {}", metadata_path);
    let metadata = read_metadata(metadata_path)?;

    let mut dds = if blind {
        let mut blind_meta = metadata.clone();
        let n = counts.n_samples();
        blind_meta.add_condition("_blind_intercept", vec!["1".to_string(); n])?;
        info!("Creating DESeqDataSet with blind design: ~1");
        DESeqDataSet::new(counts, blind_meta, "_blind_intercept")?
    } else {
        info!("Creating DESeqDataSet with design: ~ {}", design);
        DESeqDataSet::new(counts, metadata, design)?
    };

    info!("Estimating size factors...");
    estimate_size_factors(&mut dds, SizeFactorMethod::Ratio)?;

    info!(
        "Applying rlog transformation (blind: {})...",
        blind
    );

    estimate_dispersions(&mut dds, TrendFitMethod::Parametric, &DispersionParams::default())?;

    let rlog_result = rlog(&dds, blind)?;

    info!("Writing rlog-transformed data to: {}", output_path);

    let mut file = std::fs::File::create(output_path)?;
    use std::io::Write;

    let sample_ids = dds.counts().sample_ids();
    writeln!(file, "gene_id\t{}", sample_ids.join("\t"))?;

    let gene_ids = dds.counts().gene_ids();
    for (i, gene_id) in gene_ids.iter().enumerate() {
        let row: Vec<String> = (0..dds.n_samples())
            .map(|j| format!("{:.10}", rlog_result.data[[i, j]]))
            .collect();
        writeln!(file, "{}\t{}", gene_id, row.join("\t"))?;
    }

    info!("Done! rlog transformation complete.");
    Ok(())
}
