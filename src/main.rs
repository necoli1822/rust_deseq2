//! rust_deseq2 command-line interface

use clap::Parser;
use log::{info, LevelFilter};

use rust_deseq2::cli::{Cli, Commands, ContrastArgs, InputArgs, TximportArgs};
use rust_deseq2::testing::wald_test_extended;
use rust_deseq2::filter::{calculate_cooks_distance, default_cooks_cutoff, filter_by_cooks_with_counts, n_or_more_in_cell, replace_outliers};
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
            input,
            design,
            formula,
            covariate,
            continuous,
            reference,
            test,
            reduced,
            numerator,
            denominator,
            contrast,
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
            weights,
            cooks_cutoff,
        }) => run_analysis(
            &input,
            design.as_deref(),
            formula.as_deref(),
            &covariate,
            &continuous,
            &reference,
            &test,
            reduced.as_deref(),
            numerator.as_deref(),
            denominator.as_deref(),
            &contrast,
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
            weights.as_deref(),
            cooks_cutoff,
        ),
        Some(Commands::Normalize {
            counts,
            htseq,
            htseq_dir,
            tx,
            output,
            method,
        }) => run_normalize(counts.as_deref(), htseq.as_deref(), &htseq_dir, &tx, &output, &method),
        Some(Commands::Vst {
            input,
            design,
            formula,
            continuous,
            output,
            method,
            blind,
            nsub,
        }) => run_vst(&input, design.as_deref(), formula.as_deref(), &continuous, &output, &method, blind, nsub),
        Some(Commands::Rlog {
            input,
            design,
            formula,
            continuous,
            output,
            blind,
        }) => run_rlog(&input, design.as_deref(), formula.as_deref(), &continuous, &output, blind),
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
    println!("               - Multi-factor designs with covariates, or any R design formula");
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
    println!("  rust_deseq2 run -c counts.csv -m metadata.csv --formula \"~ genotype * treatment\" \\");
    println!("    --name genotypeKO.treatmenttrt");
    println!();
    println!("  rust_deseq2 vst -c counts.csv -m metadata.csv -d condition --blind");
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

/// Counts, sample metadata and (tximport) average transcript lengths
struct LoadedInput {
    counts: CountMatrix,
    /// None for `normalize -c`, which takes no metadata
    metadata: Option<SampleMetadata>,
    /// txi$length (R: assays(dds)[["avgTxLength"]])
    avg_tx_length: Option<ndarray::Array2<f64>>,
}

/// Read the counts and sample metadata: a count matrix with a metadata table, or
/// htseq-count files listed in a sample table (R: DESeqDataSetFromHTSeqCount)
fn load_input(input: &InputArgs) -> Result<(CountMatrix, SampleMetadata, Option<ndarray::Array2<f64>>)> {
    let loaded = load_counts(input.counts.as_deref(), input.metadata.as_deref(), input.htseq.as_deref(), &input.htseq_dir, &input.tx)?;
    let metadata = loaded.metadata.ok_or_else(|| DeseqError::InvalidInput {
        reason: "--metadata is required with --counts".to_string(),
    })?;
    Ok((loaded.counts, metadata, loaded.avg_tx_length))
}

/// Read counts from -c (with -m), --htseq or --quant.
/// R: DESeqDataSetFromMatrix() rejects non-integer counts ("some values in assay are
/// not integers"); tximport counts (--quant, --avg-tx-length, --tximport) are
/// rounded first, as DESeqDataSetFromTximport() does with round(txi$counts).
fn load_counts(
    counts_path: Option<&str>,
    metadata_path: Option<&str>,
    htseq: Option<&str>,
    htseq_dir: &str,
    tx: &TximportArgs,
) -> Result<LoadedInput> {
    use rust_deseq2::io::tximport::r_round;
    if let Some(table) = &tx.quant {
        let quant_type = rust_deseq2::io::QuantType::parse(&tx.quant_type)?;
        info!(
            "Loading {} quantifications listed in {} (directory {}, tx2gene {:?})",
            tx.quant_type, table, tx.quant_dir, tx.tx2gene
        );
        let (txi, metadata) =
            rust_deseq2::io::read_quant(table, &tx.quant_dir, quant_type, tx.tx2gene.as_deref(), tx.read_length)?;
        info!("  {} genes, {} samples (tximport, countsFromAbundance = \"no\")", txi.gene_ids.len(), txi.sample_ids.len());
        let counts = CountMatrix::new(txi.counts.mapv(r_round), txi.gene_ids, txi.sample_ids)?;
        return Ok(LoadedInput { counts, metadata: Some(metadata), avg_tx_length: Some(txi.length) });
    }
    let tximport = tx.tximport || tx.avg_tx_length.is_some();
    let (counts, metadata) = match (counts_path, htseq) {
        (_, Some(table)) => {
            let (counts, metadata) = load_htseq(table, htseq_dir)?;
            (counts, Some(metadata))
        }
        (Some(path), None) => {
            info!("Loading count matrix from: {}", path);
            let counts = if tximport {
                // DESeqDataSetFromTximport(): counts <- round(txi$counts)
                info!("  tximport counts: rounding them");
                rust_deseq2::io::read_tximport_counts(path)?
            } else {
                read_count_matrix(path)?
            };
            info!("  {} genes, {} samples", counts.n_genes(), counts.n_samples());
            let metadata = match metadata_path {
                Some(m) => {
                    info!("Loading metadata from: {}", m);
                    Some(read_metadata(m)?)
                }
                None => None,
            };
            (counts, metadata)
        }
        (None, None) => unreachable!("clap requires --counts, --htseq or --quant"),
    };
    // R: DESeqDataSetFromMatrix() stops with "some values in assay are not integers"
    if counts.counts().iter().any(|&x| x != r_round(x)) {
        return Err(DeseqError::InvalidCountMatrix {
            reason: "some values in the count matrix are not integers (for tximport counts use \
                     --tximport or --avg-tx-length)"
                .to_string(),
        });
    }
    let avg_tx_length = match &tx.avg_tx_length {
        Some(path) => {
            info!("Loading average transcript lengths from: {}", path);
            Some(rust_deseq2::io::read_aligned_matrix(path, counts.gene_ids(), counts.sample_ids(), "average transcript length")?)
        }
        None => None,
    };
    Ok(LoadedInput { counts, metadata, avg_tx_length })
}

/// Attach the tximport lengths and, if given, the normalization factors; then
/// estimate size factors unless normalization factors were supplied.
/// R: DESeqDataSetFromTximport() stores assays(dds)[["avgTxLength"]];
/// normalizationFactors(dds) <- nf; DESeq() skips estimateSizeFactors() when
/// size or normalization factors are already set.
fn normalize_dataset(
    dds: &mut DESeqDataSet,
    avg_tx_length: Option<ndarray::Array2<f64>>,
    tx: &TximportArgs,
    sf_method: SizeFactorMethod,
    sf_name: &str,
) -> Result<()> {
    if let Some(lengths) = avg_tx_length {
        dds.set_avg_tx_length(lengths)?;
    }
    if let Some(path) = &tx.norm_factors {
        info!("Loading normalization factors from: {} (size factor estimation skipped)", path);
        let nf = rust_deseq2::io::read_aligned_matrix(path, dds.counts().gene_ids(), dds.counts().sample_ids(), "normalization factor")?;
        return dds.set_normalization_factors(nf);
    }
    if dds.avg_tx_length().is_some() && sf_method != SizeFactorMethod::Iterate {
        info!("Estimating normalization factors from the average transcript lengths (method: {})...", sf_name);
    } else {
        info!("Estimating size factors (method: {})...", sf_name);
    }
    estimate_size_factors(dds, sf_method)
}

fn load_htseq(table: &str, directory: &str) -> Result<(CountMatrix, SampleMetadata)> {
    info!("Loading htseq-count files listed in {} (directory {})", table, directory);
    let (counts, metadata) = rust_deseq2::io::read_htseq_count(table, directory)?;
    info!("  {} genes, {} samples", counts.n_genes(), counts.n_samples());
    Ok((counts, metadata))
}

/// Register the --continuous columns of the metadata as numeric
/// read_metadata keeps every column as a categorical factor; the columns named by
/// --continuous are numeric covariates (R: a numeric colData column), so register them
/// as such or the design cannot find them.
fn register_continuous(metadata: &mut SampleMetadata, continuous_vars: &[String]) -> Result<()> {
    for name in continuous_vars {
        let raw = metadata.condition(name).cloned().ok_or_else(|| DeseqError::InvalidMetadata {
            reason: format!("Continuous variable '{}' not found in metadata", name),
        })?;
        let values = raw
            .iter()
            .map(|v| {
                v.trim().parse::<f64>().ok().filter(|x| x.is_finite()).ok_or_else(|| DeseqError::InvalidMetadata {
                    reason: format!("Continuous variable '{}' has non-numeric value '{}'", name, v),
                })
            })
            .collect::<Result<Vec<f64>>>()?;
        metadata.add_continuous(name, values)?;
    }
    Ok(())
}

/// A dataset whose design is the formula `text`
/// R equivalent: DESeqDataSetFromMatrix(countData, colData, design = formula), with the
/// messages DESeqDataSet() gives for numeric design variables
fn formula_dataset(
    counts: CountMatrix,
    metadata: SampleMetadata,
    text: &str,
    continuous_vars: &[String],
    reference_levels: &[String],
) -> Result<DESeqDataSet> {
    let formula = rust_deseq2::glm::Formula::parse(text)?;
    info!("Creating DESeqDataSet with design: {}", formula.display());
    for name in continuous_vars {
        if !formula.all_vars.contains(name) {
            return Err(DeseqError::InvalidInput {
                reason: format!("--continuous {} is not a variable of the formula {}", name, formula.display()),
            });
        }
    }
    // R: DESeqDataSet() looks at the numeric colData columns of all.vars(design)
    let numeric: Vec<&String> = formula.all_vars.iter().filter(|v| metadata.has_continuous(v)).collect();
    let values = |v: &str| (0..metadata.n_samples()).map(|i| metadata.get_continuous_value(v, i).unwrap()).collect::<Vec<f64>>();
    if numeric.iter().any(|v| values(v).iter().all(|x| *x == x.round())) {
        log::warn!(
            "the design formula contains one or more numeric variables with integer values, specifying a model \
             with increasing fold change for higher values. did you mean for this to be a factor? if so, leave \
             it out of --continuous"
        );
    }
    let large = numeric.iter().any(|v| {
        let x = values(v);
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let sd = (x.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        mean.abs() > 5.0 || sd > 5.0
    });
    if large {
        log::warn!(
            "the design formula contains one or more numeric variables that have mean or standard deviation \
             larger than 5 (an arbitrary threshold to trigger this message). Including numeric variables with \
             large mean can induce collinearity with the intercept. Users should center and scale numeric \
             variables in the design to improve GLM convergence."
        );
    }
    let mut builder = DESeqDataSet::builder().counts(counts).metadata(metadata).formula(text);
    for ref_spec in reference_levels {
        let (factor, level) = ref_spec.split_once('=').ok_or_else(|| DeseqError::InvalidInput {
            reason: format!("Invalid reference format '{}'. Use: factor=level", ref_spec),
        })?;
        builder = builder.reference_level(factor, level);
    }
    builder.build()
}

fn run_analysis(
    input: &InputArgs,
    design: Option<&str>,
    formula: Option<&str>,
    covariates: &[String],
    continuous_vars: &[String],
    reference_levels: &[String],
    test_type: &str,
    reduced_formula: Option<&str>,
    numerator: Option<&str>,
    denominator: Option<&str>,
    contrast_args: &ContrastArgs,
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
    weights_path: Option<&str>,
    cooks_cutoff_override: Option<f64>,
) -> Result<()> {
    use rust_deseq2::testing::likelihood_ratio_test_reduced;

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

    let (counts, mut metadata, avg_tx_length) = load_input(input)?;
    register_continuous(&mut metadata, continuous_vars)?;

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
    let mut dds = if let Some(text) = formula {
        formula_dataset(counts, metadata, text, continuous_vars, reference_levels)?
    } else if covariates.is_empty() && continuous_vars.is_empty() && reference_levels.is_empty() {
        let design = design.expect("clap requires --design without --formula");
        // Simple design: ~ design_variable (alphabetical reference)
        info!("Creating DESeqDataSet with design: ~ {}", design);
        DESeqDataSet::new(counts, metadata, design)?
    } else {
        let design = design.expect("clap requires --design without --formula");
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

    // Observation weights (R: assays(dds)[["weights"]])
    if let Some(path) = weights_path {
        if shrinkage && shrinkage_method != "ashr" {
            return Err(DeseqError::InvalidInput {
                reason: format!(
                    "--shrinkage-method {} does not support --weights (use ashr, which only needs the LFC and its SE)",
                    shrinkage_method
                ),
            });
        }
        info!("Loading observation weights from: {}", path);
        let w = rust_deseq2::io::read_weight_matrix(path, dds.counts().gene_ids(), dds.counts().sample_ids())?;
        dds.set_weights(w)?;
    }

    let sf_method = match sf_type {
        "poscounts" => SizeFactorMethod::PosCounts,
        "iterate" => SizeFactorMethod::Iterate,
        _ => SizeFactorMethod::Ratio,
    };
    normalize_dataset(&mut dds, avg_tx_length, &input.tx, sf_method, sf_type)?;

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
    info!("resultsNames: {}", design_info.coef_names.join(", "));

    // R results(): the contrast (character, name, numeric or list); without one,
    // R's default. An explicit contrast also picks the LFC column of an LRT.
    let explicit_contrast = numerator.is_some()
        || contrast_args.name.is_some()
        || contrast_args.contrast_vector.is_some()
        || contrast_args.contrast_list.is_some();
    let contrast_spec = resolve_contrast(&dds, &design_info, design, test_type, numerator, denominator, contrast_args)?;
    info!("Contrast: {}", describe_contrast(&contrast_spec));

    // -----------------------------------------------------------------------
    // Step 3: Run Wald/LRT test FIRST (before Cook's replacement)
    // R DESeq2 order: estimateDispersions -> nbinomWaldTest/nbinomLRT -> replaceOutliers -> refit
    // -----------------------------------------------------------------------
    let mut results = if test_type == "lrt" {
        let design_full = dds.design_matrix().ok_or_else(|| DeseqError::InvalidInput {
            reason: "Design matrix not available after GLM fit".to_string(),
        })?.clone();

        let (design_reduced, reduced_intercept) = build_reduced_design(&dds, &design_info, reduced_formula)?;

        let df_full = design_full.ncols();
        let df_reduced = design_reduced.ncols();
        info!(
            "Performing LRT (full: {} params, reduced: {} params, df: {})...",
            df_full,
            df_reduced,
            df_full - df_reduced
        );

        let mut lrt = likelihood_ratio_test_reduced(&dds, &design_full, &design_reduced, reduced_intercept)?;
        if explicit_contrast {
            lrt_contrast_columns(&dds, &design_info, &contrast_spec, &mut lrt)?;
        }
        lrt
    } else {
        info!("Performing Wald test ({})...", describe_contrast(&contrast_spec));
        wald_test_extended(&dds, &design_info, contrast_spec.clone(), alpha, use_t, lfc_threshold)?
    };

    // -----------------------------------------------------------------------
    // Step 4: Cook's outlier replacement + refit (AFTER initial test)
    // R DESeq2: replaceOutliers then refitWithoutOutliers, then re-run test
    // -----------------------------------------------------------------------
    let min_replicates = min_replicates_for_replace;
    let mut replaceable_samples: Vec<bool> = vec![false; dds.n_samples()];
    // Genes whose counts are all zero once outliers are replaced (R: "nowZero")
    let mut now_zero: Vec<usize> = Vec::new();
    // Whether any gene was refit after replacement (R: nrefit > 0)
    let mut refit_done = false;
    let mut original_counts: Option<ndarray::Array2<f64>> = None;

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

        // Samples in model-matrix cells with >= minReplicates are "replaceable"
        if let Some(design_matrix) = dds.design_matrix() {
            replaceable_samples = n_or_more_in_cell(design_matrix, min_replicates);
        }

        // R's low-count heuristic for Cook's filtering compares against the original
        // counts (refitWithoutOutliers restores them), so keep a copy.
        original_counts = Some(dds.counts().counts().to_owned());

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
            now_zero = flagged_genes.iter().copied().filter(|&gene_idx| {
                (0..n_samples).all(|j| counts_matrix[[gene_idx, j]] == 0.0)
            }).collect();

            // Refit dispersions + GLM for replaced genes
            rust_deseq2::dispersion::refit_without_outliers(&mut dds, &flagged_genes)?;
            refit_done = true;

            // Bug 4: Re-run Wald/LRT test after refit
            info!("Re-running {} test after outlier replacement...", test_type);
            results = if test_type == "lrt" {
                let design_full = dds.design_matrix().ok_or_else(|| DeseqError::InvalidInput {
                    reason: "Design matrix not available after refit".to_string(),
                })?.clone();
                let (design_reduced, reduced_intercept) = build_reduced_design(&dds, &design_info, reduced_formula)?;
                let mut lrt = likelihood_ratio_test_reduced(&dds, &design_full, &design_reduced, reduced_intercept)?;
                if explicit_contrast {
                    lrt_contrast_columns(&dds, &design_info, &contrast_spec, &mut lrt)?;
                }
                lrt
            } else {
                wald_test_extended(&dds, &design_info, contrast_spec.clone(), alpha, use_t, lfc_threshold)?
            };

        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Cook's filtering on results using ORIGINAL (pre-replacement) Cook's distances
    // -----------------------------------------------------------------------
    // R: recordMaxCooks takes the maximum over samples in cells with 3+ replicates
    // (none -> maxCooks NA -> no filtering); after a refit, replaceable samples are
    // zeroed because their outliers were already replaced.
    let mut max_cooks_samples = match dds.design_matrix() {
        Some(design_matrix) => n_or_more_in_cell(design_matrix, 3),
        None => vec![false; dds.n_samples()],
    };
    if refit_done {
        for (eligible, &replaceable) in max_cooks_samples.iter_mut().zip(&replaceable_samples) {
            *eligible &= !replaceable;
        }
    }

    if max_cooks_samples.iter().any(|&s| s) {
        // Use ORIGINAL Cook's distances (pre-replacement) for filtering.
        // If outlier replacement was done, use saved original_cooks; otherwise calculate fresh.
        let cooks = if let Some(ref orig) = original_cooks {
            orig.clone()
        } else {
            calculate_cooks_distance(&dds)?
        };
        let n_coefs = dds.design_matrix().map(|m| m.ncols()).unwrap_or(2);
        let cooks_cutoff = cooks_cutoff_override.unwrap_or_else(|| default_cooks_cutoff(dds.n_samples(), n_coefs));

        // R results(): the low-count heuristic applies when the design has a single
        // variable, a factor with two levels (all.vars(design) has length 1)
        let is_two_level_design = match design_info.design_vars.as_slice() {
            [v] => design_info.reference_levels.contains_key(v) && *v == design_info.design_variable && design_info.levels.len() == 2,
            _ => false,
        };
        let counts = original_counts.clone().unwrap_or_else(|| dds.counts().counts().to_owned());
        filter_by_cooks_with_counts(
            &mut results,
            &cooks,
            cooks_cutoff,
            Some(&counts),
            is_two_level_design,
            &max_cooks_samples,
        );
    }

    // R results(): genes that are all zero after outlier replacement report
    // LFC = 0, lfcSE = 0, stat = 0, pvalue = 1. This comes after Cook's filtering,
    // so it overrides an NA set there.
    if !now_zero.is_empty() {
        info!("{} genes became all-zero after outlier replacement", now_zero.len());
        for &gene_idx in &now_zero {
            results.log2_fold_changes[gene_idx] = 0.0;
            results.lfc_se[gene_idx] = 0.0;
            results.stat[gene_idx] = 0.0;
            results.pvalues[gene_idx] = 1.0;
        }
    }

    // Apply independent filtering (or skip if --no-independent-filtering)
    if no_independent_filtering {
        // Just apply BH adjustment to all genes without filtering
        let pvalues = &results.pvalues;
        results.padj = rust_deseq2::testing::benjamini_hochberg(pvalues);
    } else {
        independent_filtering(&mut results, alpha);
    }

    // After a refit R's counts(dds) are the original counts again
    // (refitWithoutOutliers restores them), and lfcShrink() fits those with the
    // refitted dispersions and MLE.
    let shrink_counts = if refit_done { original_counts.as_ref() } else { None };
    if shrinkage && test_type == "wald" {
        match shrinkage_method {
            "apeglm" => {
                info!("Applying apeglm LFC shrinkage...");
                use rust_deseq2::shrinkage::{shrink_lfc_apeglm_with_counts, ApeglmParams};
                let params = ApeglmParams::default();
                // apeglm shrinks one coefficient (R: lfcShrink(coef=...)), so it must be
                // the coefficient the Wald contrast reported. A reversed contrast
                // (numerator = reference) is that coefficient negated; the Cauchy prior
                // is symmetric, so negating the shrunken estimate is exact.
                let (coef_idx, sign) = single_coefficient(&design_info, &contrast_spec)?;
                shrink_lfc_apeglm_with_counts(&dds, shrink_counts, &mut results, coef_idx, &params)?;
                if sign < 0.0 {
                    results.log2_fold_changes.iter_mut().for_each(|lfc| *lfc = -*lfc);
                }
            }
            "ashr" => {
                info!("Applying ashr LFC shrinkage...");
                use rust_deseq2::shrinkage::{apply_ashr_shrinkage, AshrParams};
                let params = AshrParams::default();
                apply_ashr_shrinkage(&mut results, &params);
            }
            _ => {
                info!("Applying normal LFC shrinkage...");
                // R lfcShrink(type = "normal") refits with betaPrior = TRUE, which
                // needs an intercept and no interactions
                if design_info.interaction_present {
                    return Err(DeseqError::InvalidInput {
                        reason: "LFC shrinkage type='normal' not implemented for designs with interactions \
                                 (use --shrinkage-method apeglm or ashr)"
                            .to_string(),
                    });
                }
                if !design_info.has_intercept {
                    return Err(DeseqError::InvalidInput {
                        reason: "LFC shrinkage type='normal' (betaPrior=TRUE) can only be used if the design has an \
                                 intercept (use --shrinkage-method apeglm or ashr)"
                            .to_string(),
                    });
                }
                // Normal shrinkage refits one coefficient (R: lfcShrink(coef = ...)); a
                // contrast that is not a single coefficient is an error rather than an
                // unshrunken MLE written out as if it were shrunken.
                let (coef_idx, sign) = single_coefficient(&design_info, &contrast_spec)?;
                rust_deseq2::shrinkage::shrink_lfc_normal_coef(&dds, shrink_counts, &mut results, upper_quantile, coef_idx, sign);
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

/// The contrast of results() from the CLI options, or R's default
/// R equivalent: checkContrast() and the default in results(): without contrast and
/// name, `name <- lastCoefName(object)`, except for Wald tests of designs without an
/// intercept and without interactions, where R uses
/// `contrast = c(lastVarName, levels(lastVar)[nlvls], levels(lastVar)[1])` when the last
/// variable of the design is a factor
fn resolve_contrast(
    dds: &DESeqDataSet,
    info: &rust_deseq2::glm::DesignInfo,
    design: Option<&str>,
    test_type: &str,
    numerator: Option<&str>,
    denominator: Option<&str>,
    c: &ContrastArgs,
) -> Result<ContrastSpec> {
    let names = &info.coef_names;
    let unknown = |what: &str| DeseqError::InvalidContrast {
        reason: format!("Coefficient '{}' not found. Available (resultsNames): {}", what, names.join(", ")),
    };
    if let Some(name) = &c.name {
        if !names.contains(name) {
            return Err(unknown(name));
        }
        return Ok(ContrastSpec::Name(name.clone()));
    }
    if let Some(w) = &c.contrast_vector {
        // R checkContrast(): one element per resultsNames entry, not all zero
        if w.len() != names.len() {
            return Err(DeseqError::InvalidContrast {
                reason: format!(
                    "numeric contrast has {} elements but there are {} coefficients ({})",
                    w.len(),
                    names.len(),
                    names.join(", ")
                ),
            });
        }
        return Ok(ContrastSpec::Numeric(w.clone()));
    }
    if let Some(list) = &c.contrast_list {
        let (num, den) = list.split_once(';').unwrap_or((list.as_str(), ""));
        let split = |s: &str| -> Vec<String> {
            s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
        };
        let lv = c.list_values.clone().unwrap_or_else(|| vec![1.0, -1.0]);
        // R: stopifnot(length(listValues) == 2 & is.numeric(listValues))
        if lv.len() != 2 {
            return Err(DeseqError::InvalidContrast {
                reason: format!("--list-values needs two numbers, got {}", lv.len()),
            });
        }
        return Ok(ContrastSpec::List {
            numerator_coefs: split(num),
            denominator_coefs: split(den),
            list_values: (lv[0], lv[1]),
        });
    }
    if let (Some(num), Some(den)) = (numerator, denominator) {
        let variable = c.contrast_var.as_deref().or(design).ok_or_else(|| DeseqError::InvalidContrast {
            reason: "with --formula, --numerator/--denominator need --contrast-var FACTOR \
                     (R: contrast = c(factor, numerator, denominator))"
                .to_string(),
        })?;
        // R cleanContrast(): the contrast variable must be a factor column of colData
        let meta = dds.sample_metadata();
        if !meta.has_condition(variable) && !meta.has_continuous(variable) {
            return Err(DeseqError::InvalidContrast {
                reason: format!(
                    "{} should be the name of a factor in the colData of the DESeqDataSet (see --contrast-var)",
                    variable
                ),
            });
        }
        if meta.has_continuous(variable) {
            return Err(DeseqError::InvalidContrast {
                reason: format!(
                    "{} is not a factor, see ?results (it is a --continuous column; a formula term such as \
                     factor({}) does not make the column a factor, use --name with resultsNames instead)",
                    variable, variable
                ),
            });
        }
        // R checkContrast(): the two levels must differ
        if num == den {
            return Err(DeseqError::InvalidContrast {
                reason: format!("{} and {} should be different level names", num, den),
            });
        }
        return Ok(ContrastSpec::simple(variable, num, den));
    }
    // R's default without an intercept: the last level of the last variable against
    // its first level
    if test_type == "wald" && !info.has_intercept && !info.interaction_present {
        if let Some(last) = info.design_vars.last() {
            if *last == info.design_variable && info.levels.len() >= 2 {
                let levels = &info.levels;
                return Ok(ContrastSpec::simple(last, &levels[levels.len() - 1], &levels[0]));
            }
        }
    }
    // R's default: the last coefficient
    let _ = dds;
    names.last().cloned().map(ContrastSpec::Name).ok_or_else(|| DeseqError::InvalidContrast {
        reason: "the design has no coefficients".to_string(),
    })
}

fn describe_contrast(spec: &ContrastSpec) -> String {
    match spec {
        ContrastSpec::Simple { variable, numerator, denominator } => format!("{} {} vs {}", variable, numerator, denominator),
        ContrastSpec::Name(n) => format!("coefficient {}", n),
        ContrastSpec::Numeric(w) => format!("numeric contrast {:?}", w),
        ContrastSpec::List { numerator_coefs, denominator_coefs, list_values } => format!(
            "list contrast {} vs {} (listValues {}, {})",
            numerator_coefs.join("+"),
            denominator_coefs.join("+"),
            list_values.0,
            list_values.1
        ),
    }
}

/// The single coefficient (index, sign) a contrast reports, for shrinkage
/// R equivalent: lfcShrink(dds, coef = ...) needs one coefficient; a character contrast
/// whose denominator is the reference level is that level's coefficient, and the
/// reversed contrast is the coefficient negated
fn single_coefficient(info: &rust_deseq2::glm::DesignInfo, spec: &ContrastSpec) -> Result<(usize, f64)> {
    let vector = match spec {
        ContrastSpec::Simple { variable, numerator, denominator } => {
            rust_deseq2::glm::contrast_vector(info, variable, numerator, denominator)?
        }
        ContrastSpec::Name(n) => info.coef_names.iter().map(|c| if c == n { 1.0 } else { 0.0 }).collect(),
        ContrastSpec::Numeric(w) => w.clone(),
        ContrastSpec::List { numerator_coefs, denominator_coefs, list_values } => info
            .coef_names
            .iter()
            .map(|c| {
                if numerator_coefs.contains(c) {
                    list_values.0
                } else if denominator_coefs.contains(c) {
                    list_values.1
                } else {
                    0.0
                }
            })
            .collect(),
    };
    let nonzero: Vec<usize> = (0..vector.len()).filter(|&j| vector[j] != 0.0).collect();
    match nonzero.as_slice() {
        [j] if vector[*j].abs() == 1.0 => Ok((*j, vector[*j])),
        _ => Err(DeseqError::InvalidContrast {
            reason: format!(
                "--shrinkage-method normal/apeglm shrinks a single coefficient (R: lfcShrink(coef = ...)), \
                 but the contrast ({}) is not one; use --name, or a contrast against the reference level, or ashr",
                describe_contrast(spec)
            ),
        }),
    }
}

/// LRT results with the log2FoldChange/lfcSE of a contrast
/// R equivalent: results(dds, test = "LRT", contrast/name = ...): cleanContrast() computes
/// the contrast (with contrastAllZero setting LFC = 0) and then replaces stat and pvalue
/// with the LRT's
fn lrt_contrast_columns(
    dds: &DESeqDataSet,
    info: &rust_deseq2::glm::DesignInfo,
    spec: &ContrastSpec,
    lrt: &mut DESeqResults,
) -> Result<()> {
    let wald = wald_test_extended(dds, info, spec.clone(), 0.1, false, 0.0)?;
    lrt.log2_fold_changes = wald.log2_fold_changes;
    lrt.lfc_se = wald.lfc_se;
    lrt.contrast = wald.contrast;
    Ok(())
}

/// The reduced model matrix of the LRT from the --reduced formula (default `~ 1`)
/// R equivalent: nbinomLRT(): checkLRT(full, reduced), then
/// model.matrix(reduced, colData) and df = ncol(full) - ncol(reduced) >= 1
fn build_reduced_design(
    dds: &DESeqDataSet,
    info: &rust_deseq2::glm::DesignInfo,
    reduced_formula: Option<&str>,
) -> Result<(ndarray::Array2<f64>, bool)> {
    let reduced = rust_deseq2::glm::Formula::parse(reduced_formula.unwrap_or("~1"))?;
    info!("Reduced model: {}", reduced.display());
    let full_columns = dds.design_matrix().map(|m| m.ncols()).unwrap_or(info.coef_names.len());
    let matrix =
        rust_deseq2::glm::reduced_design(&reduced, &info.design_vars, full_columns, dds.sample_metadata(), dds.reference_levels())?;
    Ok((matrix, reduced.intercept))
}

fn run_normalize(
    counts_path: Option<&str>,
    htseq: Option<&str>,
    htseq_dir: &str,
    tx: &TximportArgs,
    output_path: &str,
    method: &str,
) -> Result<()> {
    let LoadedInput { counts, avg_tx_length, .. } = load_counts(counts_path, None, htseq, htseq_dir, tx)?;

    // Size factors do not depend on the design, but DESeqDataSet requires a
    // design variable that exists in the metadata, so supply a single-level one.
    let mut metadata = SampleMetadata::new(counts.sample_ids().to_vec());
    let n = counts.n_samples();
    metadata.add_condition("_intercept", vec!["1".to_string(); n])?;

    let mut dds = DESeqDataSet::new(counts, metadata, "_intercept")?;

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

    normalize_dataset(&mut dds, avg_tx_length, tx, size_method, method)?;

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

/// The dataset of the vst and rlog commands: design ~ design, or a --formula
fn transform_dataset(
    counts: CountMatrix,
    mut metadata: SampleMetadata,
    design: Option<&str>,
    formula: Option<&str>,
    continuous_vars: &[String],
) -> Result<DESeqDataSet> {
    register_continuous(&mut metadata, continuous_vars)?;
    match (formula, design) {
        (Some(text), _) => formula_dataset(counts, metadata, text, continuous_vars, &[]),
        (None, Some(design)) => {
            info!("Creating DESeqDataSet with design: ~ {}", design);
            DESeqDataSet::new(counts, metadata, design)
        }
        (None, None) => unreachable!("clap requires --design or --formula"),
    }
}

fn run_vst(
    input: &InputArgs,
    design: Option<&str>,
    formula: Option<&str>,
    continuous_vars: &[String],
    output_path: &str,
    method: &str,
    blind: bool,
    nsub: Option<usize>,
) -> Result<()> {
    use rust_deseq2::transform::{vst, vst_subsampled, VstMethod};

    let (counts, metadata, avg_tx_length) = load_input(input)?;

    let mut dds = transform_dataset(counts, metadata, design, formula, continuous_vars)?;

    normalize_dataset(&mut dds, avg_tx_length, &input.tx, SizeFactorMethod::Ratio, "ratio")?;

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
    let vst_result = match nsub {
        // R: vst(dds, blind, nsub, fitType) fits the trend on a gene subset
        Some(nsub) => vst_subsampled(&dds, vst_method, nsub)?,
        None => {
            // R: varianceStabilizingTransformation(dds, fitType=...) estimates the
            // dispersion trend with that fitType; the transformation follows it.
            let trend_method = match vst_method {
                VstMethod::Parametric => TrendFitMethod::Parametric,
                VstMethod::Mean => TrendFitMethod::Mean,
                VstMethod::Local => TrendFitMethod::Local,
            };
            estimate_dispersions(&mut dds, trend_method, &DispersionParams::default())?;
            vst(&dds, vst_method, blind)?
        }
    };

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
    input: &InputArgs,
    design: Option<&str>,
    formula: Option<&str>,
    continuous_vars: &[String],
    output_path: &str,
    blind: bool,
) -> Result<()> {
    use rust_deseq2::transform::rlog;

    let (counts, metadata, avg_tx_length) = load_input(input)?;

    let mut dds = if blind {
        let mut blind_meta = metadata.clone();
        let n = counts.n_samples();
        blind_meta.add_condition("_blind_intercept", vec!["1".to_string(); n])?;
        info!("Creating DESeqDataSet with blind design: ~1");
        DESeqDataSet::new(counts, blind_meta, "_blind_intercept")?
    } else {
        transform_dataset(counts, metadata, design, formula, continuous_vars)?
    };

    normalize_dataset(&mut dds, avg_tx_length, &input.tx, SizeFactorMethod::Ratio, "ratio")?;

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
