//! Command-line interface for rust_deseq2

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rust_deseq2")]
#[command(author = "SunJu Kim")]
#[command(version)]
#[command(about = "DESeq2 differential expression analysis in Rust")]
#[command(disable_help_flag = true)]
#[command(disable_version_flag = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run full DESeq2 analysis
    #[command(
        about = "Run full DESeq2 analysis",
        long_about = "Run full DESeq2 analysis\n\n\
            Performs the complete DESeq2 pipeline: size factor estimation, dispersion\n\
            estimation, GLM fitting, and statistical testing (Wald or LRT).\n\n\
            Supports multi-factor designs with batch effect correction, three LFC\n\
            shrinkage methods, and Cook's distance outlier detection/replacement.",
        after_long_help = "\
Examples:
  # Basic two-condition comparison (Wald test)
  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\
    --numerator treated --denominator control -o results.csv

  # With batch effect correction
  rust_deseq2 run -c counts.csv -m metadata.csv -d treatment \\
    --covariate batch --numerator drug --denominator placebo

  # Likelihood Ratio Test
  rust_deseq2 run -c counts.csv -m metadata.csv -d treatment \\
    --test lrt --reduced \"~1\" -o lrt_results.csv

  # With LFC shrinkage (apeglm)
  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\
    --numerator treated --denominator control --shrinkage --shrinkage-method apeglm

  # With local dispersion fit and t-distribution p-values
  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\
    --numerator treated --denominator control --fit-type local --use-t"
    )]
    Run {
        /// Path to count matrix CSV file
        #[arg(short, long,
            long_help = "Path to count matrix CSV file.\n\
                Format: first column = gene IDs, remaining columns = raw counts per sample.\n\
                Supports both CSV (comma) and TSV (tab) delimiters (auto-detected).")]
        counts: String,

        /// Path to sample metadata CSV file
        #[arg(short, long,
            long_help = "Path to sample metadata CSV file.\n\
                Format: first column = sample IDs (matching count matrix columns),\n\
                remaining columns = experimental variables.")]
        metadata: String,

        /// Design variable (variable of interest)
        #[arg(short, long,
            long_help = "Main design variable for differential expression.\n\
                Must match a column name in the metadata file.\n\
                The reference level is chosen alphabetically (override with --reference).")]
        design: String,

        /// Additional categorical covariate
        #[arg(long, value_name = "VAR",
            long_help = "Additional categorical covariates for batch effect correction.\n\
                Can be specified multiple times: --covariate batch --covariate experiment")]
        covariate: Vec<String>,

        /// Continuous covariate
        #[arg(long, value_name = "VAR",
            long_help = "Continuous covariates.\n\
                Can be specified multiple times: --continuous age --continuous weight")]
        continuous: Vec<String>,

        /// Reference level (format: factor=level)
        #[arg(long, value_name = "FACTOR=LEVEL",
            long_help = "Reference level for a factor.\n\
                Format: factor=level (e.g., --reference condition=control)\n\
                Without this, the alphabetically first level is used as reference.")]
        reference: Vec<String>,

        /// Statistical test [default: wald]
        #[arg(long, default_value = "wald",
            long_help = "Statistical test to use.\n\
                wald: Wald test — tests individual coefficients, requires --numerator/--denominator\n\
                lrt:  Likelihood Ratio Test — compares full vs reduced model, requires --reduced")]
        test: String,

        /// Reduced model formula for LRT
        #[arg(long,
            long_help = "Reduced model formula for LRT.\n\
                Examples: \"~1\" (intercept only), \"~batch\" (keep batch, test treatment)")]
        reduced: Option<String>,

        /// Numerator level for contrast
        #[arg(long,
            long_help = "Numerator level for the contrast (required for Wald test).\n\
                This is the \"treatment\" or \"case\" condition.")]
        numerator: Option<String>,

        /// Denominator level for contrast
        #[arg(long,
            long_help = "Denominator level for the contrast (required for Wald test).\n\
                This is the \"control\" or \"baseline\" condition.")]
        denominator: Option<String>,

        /// Output file path [default: deseq2_results.csv]
        #[arg(short, long, default_value = "deseq2_results.csv")]
        output: String,

        /// Significance threshold [default: 0.1]
        #[arg(short, long, default_value = "0.1")]
        alpha: f64,

        /// Apply LFC shrinkage
        #[arg(long,
            long_help = "Apply log2 fold-change shrinkage after testing.\n\
                Reduces noisy LFC estimates for low-count genes.\n\
                Use with --shrinkage-method to select the algorithm.")]
        shrinkage: bool,

        /// Shrinkage method [default: normal]
        #[arg(long, default_value = "normal",
            long_help = "LFC shrinkage method.\n\
                normal:  Normal prior (DESeq2 default, fastest)\n\
                apeglm:  Approximate posterior estimation (recommended for most uses)\n\
                ashr:    Adaptive shrinkage with unimodal assumption")]
        shrinkage_method: String,

        /// Size factor estimation method [default: ratio]
        #[arg(long, default_value = "ratio",
            long_help = "Size factor estimation method.\n\
                ratio:     Median of ratios (DESeq2 default)\n\
                poscounts: Modified ratio using positive counts only\n\
                iterate:   Iterative estimation for sparse data")]
        sf_type: String,

        /// Minimum replicates for outlier replacement [default: 7]
        #[arg(long, default_value = "7",
            long_help = "Minimum number of replicates in a cell for Cook's distance\n\
                outlier replacement to be applied. Set to a large value (e.g., 9999)\n\
                to effectively disable replacement.")]
        min_replicates: usize,

        /// Replace outliers via Cook's distance [default: true]
        #[arg(long, default_value_t = true,
            long_help = "Replace outlier counts detected by Cook's distance.\n\
                Enabled by default, matching R DESeq2 behavior.\n\
                Only applies when a condition group has >= min-replicates replicates.")]
        replace_outliers: bool,

        /// Log2 fold change threshold for hypothesis testing [default: 0]
        #[arg(long, default_value = "0",
            long_help = "Log2 fold change threshold for testing H0: |LFC| <= threshold.\n\
                When > 0, uses greaterAbs alternative hypothesis.\n\
                R equivalent: results(dds, lfcThreshold=..., altHypothesis=\"greaterAbs\")")]
        lfc_threshold: f64,

        /// Dispersion trend fit method [default: parametric]
        #[arg(long, default_value = "parametric",
            long_help = "Dispersion-mean trend fitting method.\n\
                parametric: Gamma GLM with identity link (DESeq2 default)\n\
                local:      Local regression (locfit)\n\
                mean:       Mean of gene-wise dispersions")]
        fit_type: String,

        /// Use t-distribution for Wald test p-values
        #[arg(long,
            long_help = "Use t-distribution instead of normal distribution for Wald test p-values.\n\
                More conservative for small sample sizes (< 10).\n\
                Degrees of freedom = n_samples - n_coefficients.")]
        use_t: bool,

        /// Maximum IRLS iterations for GLM fitting and dispersion estimation [default: 100]
        #[arg(long, default_value = "100")]
        maxit: usize,

        /// Beta convergence tolerance for GLM fitting [default: 1e-8]
        #[arg(long, default_value = "1e-8")]
        beta_tol: f64,

        /// Minimum dispersion value [default: 1e-8]
        #[arg(long, default_value = "1e-8")]
        min_disp: f64,

        /// Dispersion convergence tolerance [default: 1e-6]
        #[arg(long, default_value = "1e-6")]
        disp_tol: f64,

        /// Initial step size for dispersion line search [default: 1.0]
        #[arg(long, default_value = "1.0")]
        kappa_0: f64,

        /// Outlier SD threshold for MAP dispersion shrinkage [default: 2.0]
        #[arg(long, default_value = "2.0")]
        outlier_sd: f64,

        /// Trim fraction for outlier replacement trimmed mean [default: 0.2]
        #[arg(long, default_value = "0.2")]
        trim: f64,

        /// Upper quantile for beta prior variance estimation [default: 0.05]
        #[arg(long, default_value = "0.05")]
        upper_quantile: f64,

        /// Number of threads (0 = auto) [default: 0]
        #[arg(short = 't', long, default_value = "0")]
        threads: usize,

        /// Disable independent filtering (just apply BH to all genes)
        #[arg(long,
            long_help = "Disable independent filtering.\n\
                By default, DESeq2 uses independent filtering to increase power by\n\
                removing low-count genes before p-value adjustment. When this flag\n\
                is set, BH adjustment is applied directly to all genes.\n\
                R equivalent: results(dds, independentFiltering=FALSE)")]
        no_independent_filtering: bool,

        /// Cook's distance cutoff (use 'Inf' or a very large number to disable)
        #[arg(long,
            long_help = "Cook's distance cutoff for flagging outlier genes.\n\
                Genes with any sample exceeding this cutoff get pvalue/padj set to NA.\n\
                Default: F(0.99, p, m-p) where p = number of coefficients, m = number of samples.\n\
                Set to 'Inf' or a very large number (e.g., 1e99) to disable Cook's filtering.\n\
                R equivalent: results(dds, cooksCutoff=...)")]
        cooks_cutoff: Option<f64>,
    },

    /// Normalize count data only
    #[command(
        long_about = "Normalize count data using DESeq2's median-of-ratios method.\n\n\
            Outputs a matrix of normalized counts (raw counts / size factors).",
        after_long_help = "\
Examples:
  rust_deseq2 normalize -c counts.csv -o normalized.tsv
  rust_deseq2 normalize -c counts.csv -o normalized.tsv -m poscounts"
    )]
    Normalize {
        /// Path to count matrix CSV file
        #[arg(short, long)]
        counts: String,

        /// Output file path
        #[arg(short, long)]
        output: String,

        /// Normalization method [default: ratio]
        #[arg(short, long, default_value = "ratio",
            long_help = "Normalization method.\n\
                ratio:     Median of ratios (DESeq2 default)\n\
                poscounts: Modified ratio using positive counts only\n\
                iterate:   Iterative estimation for sparse data")]
        method: String,
    },

    /// Apply Variance Stabilizing Transformation (VST)
    #[command(
        long_about = "Apply Variance Stabilizing Transformation.\n\n\
            Produces transformed values suitable for visualization (PCA, heatmaps)\n\
            and distance-based analyses. Faster than rlog for large datasets.",
        after_long_help = "\
Examples:
  rust_deseq2 vst -c counts.csv -m metadata.csv -d condition -o vst.tsv
  rust_deseq2 vst -c counts.csv -m metadata.csv -d condition --blind"
    )]
    Vst {
        /// Path to count matrix CSV file
        #[arg(short, long)]
        counts: String,

        /// Path to sample metadata CSV file
        #[arg(short, long)]
        metadata: String,

        /// Design variable
        #[arg(short, long)]
        design: String,

        /// Output file path [default: vst_transformed.tsv]
        #[arg(short, long, default_value = "vst_transformed.tsv")]
        output: String,

        /// VST method [default: parametric]
        #[arg(long, default_value = "parametric",
            long_help = "Dispersion-mean trend fitting method for VST.\n\
                parametric: Parametric fit (default)\n\
                mean:       Mean dispersion\n\
                local:      Local regression")]
        method: String,

        /// Blind to experimental design
        #[arg(long,
            long_help = "Ignore experimental design during transformation.\n\
                Use blind=true for QC (PCA, sample clustering).\n\
                Use blind=false when design is expected to explain variance.")]
        blind: bool,
    },

    /// Apply Regularized Log Transformation (rlog)
    #[command(
        long_about = "Apply Regularized Log Transformation.\n\n\
            Produces transformed values with approximately homoscedastic variance.\n\
            Better than VST for small datasets (< 30 samples) but slower.",
        after_long_help = "\
Examples:
  rust_deseq2 rlog -c counts.csv -m metadata.csv -d condition -o rlog.tsv
  rust_deseq2 rlog -c counts.csv -m metadata.csv -d condition --blind"
    )]
    Rlog {
        /// Path to count matrix CSV file
        #[arg(short, long)]
        counts: String,

        /// Path to sample metadata CSV file
        #[arg(short, long)]
        metadata: String,

        /// Design variable
        #[arg(short, long)]
        design: String,

        /// Output file path [default: rlog_transformed.tsv]
        #[arg(short, long, default_value = "rlog_transformed.tsv")]
        output: String,

        /// Blind to experimental design
        #[arg(long,
            long_help = "Ignore experimental design during transformation.\n\
                Use blind=true for QC (PCA, sample clustering).\n\
                Use blind=false when design is expected to explain variance.")]
        blind: bool,
    },
}
