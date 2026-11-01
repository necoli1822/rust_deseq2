//! Command-line interface for rust_deseq2

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rust_deseq2")]
#[command(author = "SunJu Kim")]
#[command(version)]
#[command(about = "DESeq2 differential expression analysis in Rust")]
// Top-level -h/--help/--version are printed by main.rs before parsing. The help
// flag must stay enabled here: disable_help_flag propagates to every subcommand,
// which made `rust_deseq2 run --help` an "unexpected argument" error.
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

  # Any R design formula, e.g. an interaction (its coefficient by --name)
  rust_deseq2 run -c counts.csv -m metadata.csv --formula \"~ genotype * treatment\" \\
    --name genotypeKO.treatmenttrt

  # LRT of the interaction
  rust_deseq2 run -c counts.csv -m metadata.csv --formula \"~ genotype * treatment\" \\
    --test lrt --reduced \"~ genotype + treatment\"

  # Likelihood Ratio Test
  rust_deseq2 run -c counts.csv -m metadata.csv -d treatment \\
    --test lrt --reduced \"~1\" -o lrt_results.csv

  # With LFC shrinkage (apeglm)
  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\
    --numerator treated --denominator control --shrinkage --shrinkage-method apeglm

  # With local dispersion fit and t-distribution p-values
  rust_deseq2 run -c counts.csv -m metadata.csv -d condition \\
    --numerator treated --denominator control --fit-type local --use-t

  # salmon/kallisto/RSEM/StringTie quantifications summarized to genes (--quant-type)
  rust_deseq2 run --quant samples.csv --quant-dir quants --tx2gene tx2gene.csv \\
    -d condition --numerator treated --denominator control"
    )]
    Run {
        #[command(flatten)]
        input: InputArgs,

        /// Design variable (variable of interest)
        #[arg(short, long, required_unless_present = "formula", conflicts_with = "formula",
            long_help = "Main design variable for differential expression.\n\
                Must match a column name in the metadata file. The design is\n\
                ~ <--covariate ...> + <--continuous ...> + <design>.\n\
                The reference level is chosen alphabetically (override with --reference).\n\
                Not used with --formula.")]
        design: Option<String>,

        /// R design formula, e.g. "~ batch + condition + batch:condition"
        #[arg(long, value_name = "FORMULA",
            long_help = "The design as an R formula (R: DESeqDataSetFromMatrix(design = ...)), instead of\n\
                -d/--covariate: terms a + b, interactions a:b, a*b (= a + b + a:b), a*b*c,\n\
                (a + b + c)^2, a/b, a %in% b, and 0 + / - 1 for no intercept. Variables are\n\
                factors (treatment contrasts, levels sorted, --reference to change the\n\
                reference) unless listed with --continuous, or logical if all values are\n\
                TRUE/FALSE; R makes numeric metadata columns numeric, here --continuous\n\
                says which are. Variables can be R expressions, evaluated as model.frame()\n\
                does: factor(), as.factor(), relevel(x, ref = ), ordered() (contr.poly),\n\
                as.numeric(), as.integer(), log(), log2(), log10(), log1p(), exp(), expm1(),\n\
                sqrt(), abs(), scale(), I() with arithmetic, poly(x, degree[, raw = TRUE]),\n\
                ns()/bs() (also splines::ns/bs) and offset() (ignored, as by DESeq2); other\n\
                R functions are an error. The model matrix, its column order and\n\
                resultsNames follow R's model.matrix() and DESeq2: \"Intercept\",\n\
                \"<factor>_<level>_vs_<ref>\" (factor columns only), interaction columns as\n\
                \"batchb2.conditionB\", functions by make.names() (\"poly.x..2.1\", \"log.age.\"),\n\
                no renaming without an intercept (\"conditionB\").\n\
                --numerator/--denominator then need --contrast-var.\n\
                Example: --formula \"~ genotype * treatment\", --formula \"~ batch + ns(age, df = 3)\"")]
        formula: Option<String>,

        /// Additional categorical covariate
        #[arg(long, value_name = "VAR", conflicts_with = "formula",
            long_help = "Additional categorical covariates for batch effect correction.\n\
                Can be specified multiple times: --covariate batch --covariate experiment")]
        covariate: Vec<String>,

        /// Continuous covariate
        #[arg(long, value_name = "VAR",
            long_help = "Continuous covariates.\n\
                Can be specified multiple times: --continuous age --continuous weight\n\
                With --formula: the metadata columns that are numeric (all others are factors\n\
                or logical); functions such as log(), poly(), ns() need numeric columns.\n\
                Centre and scale them, as R recommends.")]
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
                wald: Wald test of the contrast (see --numerator, --name, --contrast-vector,\n\
                      --contrast-list; without any, R's default contrast)\n\
                lrt:  Likelihood Ratio Test — compares full vs reduced model, requires --reduced;\n\
                      a contrast option then only chooses the reported log2FoldChange/lfcSE")]
        test: String,

        /// Reduced model formula for LRT
        #[arg(long,
            long_help = "Reduced model formula for LRT, in the syntax of --formula; its variables must\n\
                be in the full design and it must have fewer columns (R: checkLRT()).\n\
                Examples: \"~1\" (intercept only), \"~batch\" (keep batch, test treatment),\n\
                \"~ genotype + treatment\" (test the genotype:treatment interaction)")]
        reduced: Option<String>,

        /// Numerator level for contrast
        #[arg(long, requires = "denominator",
            long_help = "Numerator level of the contrast (the \"treatment\" or \"case\" condition),\n\
                a level of --design or of --contrast-var (needed with --formula).\n\
                R equivalent: results(dds, contrast = c(factor, numerator, denominator))")]
        numerator: Option<String>,

        /// Denominator level for contrast
        #[arg(long, requires = "numerator",
            long_help = "Denominator level of the contrast (the \"control\" or \"baseline\" condition).")]
        denominator: Option<String>,

        #[command(flatten)]
        contrast: ContrastArgs,

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
        // ArgAction::Set: a bool defaulting to true is otherwise a SetTrue flag that
        // can never be turned off.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL",
            long_help = "Replace outlier counts detected by Cook's distance (true/false).\n\
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
                Degrees of freedom = n_samples - n_coefficients (with --weights, the sum of\n\
                the gene's normalised weights - n_coefficients).")]
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

        /// Observation weights matrix (genes x samples)
        #[arg(long, value_name = "FILE",
            long_help = "Observation weights, a CSV/TSV matrix in the layout of the count matrix\n\
                (first column = gene IDs, then one column per sample). Gene and sample IDs\n\
                must match the count matrix; values must be non-negative and may be\n\
                non-integer. Weights are normalised per gene by their maximum and enter\n\
                the dispersion and GLM fits; baseMean uses the raw weights.\n\
                R equivalent: assays(dds)[[\"weights\"]] <- w before DESeq()")]
        weights: Option<String>,

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
        #[arg(short, long, required_unless_present_any = ["htseq", "quant"], conflicts_with = "htseq")]
        counts: Option<String>,

        /// HTSeq sample table (sample name, file name, ...); counts are read from the files
        #[arg(long, value_name = "SAMPLE_TABLE",
            long_help = "Read counts from htseq-count output files instead of --counts.\n\
                SAMPLE_TABLE is a CSV/TSV with a header: column 1 = sample name, column 2 =\n\
                file name (relative to --htseq-dir). Rows of htseq-count counters\n\
                (__no_feature, __ambiguous, ...) are dropped.\n\
                R equivalent: DESeqDataSetFromHTSeqCount(sampleTable, directory)")]
        htseq: Option<String>,

        /// Directory holding the files named in the HTSeq sample table
        #[arg(long, value_name = "DIR", default_value = ".", requires = "htseq")]
        htseq_dir: String,

        #[command(flatten)]
        tx: TximportArgs,

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
        #[command(flatten)]
        input: InputArgs,

        /// Design variable
        #[arg(short, long, required_unless_present = "formula", conflicts_with = "formula")]
        design: Option<String>,

        /// R design formula used for the dispersions unless --blind (see `run --formula`)
        #[arg(long, value_name = "FORMULA")]
        formula: Option<String>,

        /// Numeric variables of --formula
        #[arg(long, value_name = "VAR", requires = "formula")]
        continuous: Vec<String>,

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

        /// Fit the dispersion trend on N genes (R's vst()); omit for the full VST
        #[arg(long, value_name = "N",
            long_help = "Estimate the dispersion trend on a subset of N genes, as R's\n\
                vst(object, nsub = N) does (R's default N is 1000), then transform all genes.\n\
                Omit to fit the trend on all genes, as varianceStabilizingTransformation() does.")]
        nsub: Option<usize>,
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
        #[command(flatten)]
        input: InputArgs,

        /// Design variable
        #[arg(short, long, required_unless_present = "formula", conflicts_with = "formula")]
        design: Option<String>,

        /// R design formula used for the dispersions unless --blind (see `run --formula`)
        #[arg(long, value_name = "FORMULA")]
        formula: Option<String>,

        /// Numeric variables of --formula
        #[arg(long, value_name = "VAR", requires = "formula")]
        continuous: Vec<String>,

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

/// Count and sample inputs: a count matrix with a metadata table, htseq-count files,
/// or transcript quantifications (tximport)
#[derive(Args, Clone, Debug)]
pub struct InputArgs {
    /// Path to count matrix CSV file
    #[arg(short, long, required_unless_present_any = ["htseq", "quant"], conflicts_with = "htseq",
        requires = "metadata",
        long_help = "Path to count matrix CSV file.\n\
            Format: first column = gene IDs, remaining columns = raw counts per sample.\n\
            Supports both CSV (comma) and TSV (tab) delimiters (auto-detected).")]
    pub counts: Option<String>,

    /// Path to sample metadata CSV file
    #[arg(short, long, conflicts_with = "htseq",
        long_help = "Path to sample metadata CSV file.\n\
            Format: first column = sample IDs (matching count matrix columns),\n\
            remaining columns = experimental variables.")]
    pub metadata: Option<String>,

    /// HTSeq sample table (sample name, file name, variables); replaces -c and -m
    #[arg(long, value_name = "SAMPLE_TABLE",
        long_help = "Read counts from htseq-count output files instead of -c/-m.\n\
            SAMPLE_TABLE is a CSV/TSV with a header: column 1 = sample name, column 2 =\n\
            file name (relative to --htseq-dir), remaining columns = sample variables\n\
            used by the design. In each file the first column is the gene ID and the\n\
            last the count; rows of htseq-count counters (__no_feature, __ambiguous,\n\
            ...) are dropped, and the gene IDs must be the same in every file.\n\
            R equivalent: DESeqDataSetFromHTSeqCount(sampleTable, directory, design)")]
    pub htseq: Option<String>,

    /// Directory holding the files named in the HTSeq sample table
    #[arg(long, value_name = "DIR", default_value = ".", requires = "htseq")]
    pub htseq_dir: String,

    #[command(flatten)]
    pub tx: TximportArgs,
}

/// tximport input and normalization factors
#[derive(Args, Clone, Debug)]
pub struct TximportArgs {
    /// Quantification sample table (sample name, file path, variables); replaces -c and -m
    #[arg(long, visible_alias = "salmon", value_name = "SAMPLE_TABLE", conflicts_with_all = ["counts", "metadata", "htseq"],
        long_help = "Read transcript quantifications (salmon, kallisto, RSEM, StringTie, ...) and\n\
            summarize them to genes, as tximport(files, type = <--quant-type>, tx2gene)\n\
            does with its defaults (countsFromAbundance = \"no\"), then build the dataset as\n\
            DESeqDataSetFromTximport() does: counts are rounded and the average\n\
            transcript lengths become normalization factors.\n\
            SAMPLE_TABLE is a CSV/TSV with a header: column 1 = sample name, column 2 =\n\
            path of the sample's file (relative to --quant-dir), remaining columns =\n\
            sample variables used by the design. Needs --tx2gene, except for RSEM\n\
            genes.results files. `--salmon` is an alias.")]
    pub quant: Option<String>,

    /// Quantifier that wrote the --quant files
    #[arg(long, value_name = "TYPE", default_value = "salmon", requires = "quant",
        value_parser = ["salmon", "sailfish", "kallisto", "rsem", "stringtie", "piscem", "oarfish"],
        long_help = "Quantifier that wrote the --quant files (R: tximport(type = ...)):\n\
            salmon, sailfish: quant.sf (Name, TPM, NumReads, EffectiveLength)\n\
            kallisto:  abundance.tsv (target_id, tpm, est_counts, eff_length); abundance.h5\n\
                       is not read\n\
            rsem:      *.isoforms.results (transcript_id, TPM, expected_count,\n\
                       effective_length; lengths below 1 set to 1), or, if the first\n\
                       file's path contains \"genes\", *.genes.results read at gene level\n\
                       without --tx2gene, as tximport does\n\
            stringtie: t_data.ctab (t_name, FPKM, cov, length); counts = cov * length /\n\
                       --read-length\n\
            piscem:    .quant (target_name, tpm, ecount, eelen)\n\
            oarfish:   .quant (tname, num_reads, len)")]
    pub quant_type: String,

    /// Directory the paths of the --quant sample table are relative to
    #[arg(long, visible_alias = "salmon-dir", value_name = "DIR", default_value = ".", requires = "quant")]
    pub quant_dir: String,

    /// Transcript-to-gene table for --quant (columns 1-2: transcript, gene)
    #[arg(long, value_name = "FILE", requires = "quant",
        long_help = "Transcript-to-gene table (CSV/TSV) for --quant: column 1 = transcript ID as in\n\
            the quantification files, column 2 = gene ID; further columns are ignored. A\n\
            header row is optional. Transcripts missing from the table are dropped, as\n\
            tximport does.")]
    pub tx2gene: Option<String>,

    /// Read length for --quant-type stringtie (R: tximport(readLength = 75))
    #[arg(long, value_name = "N", default_value_t = 75.0, requires = "quant")]
    pub read_length: f64,

    /// tximport average transcript lengths (txi$length) for the -c count matrix
    #[arg(long, value_name = "FILE", conflicts_with_all = ["htseq", "quant"],
        long_help = "Average transcript lengths from tximport (txi$length), a CSV/TSV matrix in the\n\
            layout of the count matrix, whose -c holds txi$counts. As in\n\
            DESeqDataSetFromTximport(), the counts are rounded and size factor estimation\n\
            produces normalization factors from the lengths (sf-type ratio or poscounts;\n\
            iterate ignores them, as in R). Implies --tximport.")]
    pub avg_tx_length: Option<String>,

    /// The -c counts are tximport estimated counts: round them
    #[arg(long, conflicts_with_all = ["htseq", "quant"],
        long_help = "The -c matrix holds tximport counts (txi$counts), which may be non-integer:\n\
            round them (halves to even) as DESeqDataSetFromTximport() does. Without\n\
            --avg-tx-length this is the countsFromAbundance = \"scaledTPM\" /\n\
            \"lengthScaledTPM\" case, where R uses just the counts. Without this flag\n\
            (or --avg-tx-length, --quant) non-integer counts are an error, as in R.")]
    pub tximport: bool,

    /// Normalization factors matrix (genes x samples), used instead of size factors
    #[arg(long, value_name = "FILE",
        long_help = "Gene- and sample-specific normalization factors, a CSV/TSV matrix in the layout\n\
            of the count matrix. They replace the size factors in every fit (the offset is\n\
            log(factor)) and in the normalized counts; size factor estimation is skipped\n\
            (--sf-type is ignored). They are used as given: like R, no centering is applied.\n\
            R equivalent: normalizationFactors(dds) <- nf before DESeq()")]
    pub norm_factors: Option<String>,
}

/// Contrast choices besides --numerator/--denominator
/// R equivalent: the name / contrast / listValues arguments of results()
#[derive(Args, Clone, Debug, Default)]
pub struct ContrastArgs {
    /// Factor of the --numerator/--denominator contrast [default: --design]
    #[arg(long, value_name = "FACTOR", requires = "numerator",
        long_help = "Factor whose levels --numerator and --denominator name, e.g. a --covariate\n\
            (default: the --design variable; required with --formula).\n\
            R equivalent: results(dds, contrast = c(\"batch\", \"b2\", \"b1\"))")]
    pub contrast_var: Option<String>,

    /// Report one coefficient by its name (R: results(name = ...))
    #[arg(long, value_name = "COEF",
        conflicts_with_all = ["numerator", "contrast_vector", "contrast_list"],
        long_help = "Report the coefficient with this name, as listed by resultsNames(dds):\n\
            \"Intercept\", a continuous variable's name (its slope),\n\
            \"<factor>_<level>_vs_<reference>\", or with --formula an interaction such as\n\
            \"genotypeKO.treatmenttrt\". The names are logged at the start of the run.\n\
            R equivalent: results(dds, name = COEF)")]
    pub name: Option<String>,

    /// Numeric contrast: one weight per coefficient, comma-separated
    #[arg(long, value_name = "W1,W2,...", value_delimiter = ',', allow_hyphen_values = true,
        conflicts_with_all = ["numerator", "contrast_list"],
        long_help = "A weight for every coefficient, in resultsNames(dds) order, e.g. \"0,1,-1\".\n\
            The log2 fold change is the weighted sum of coefficients and its SE comes from\n\
            the coefficient covariance.\n\
            R equivalent: results(dds, contrast = c(0, 1, -1))")]
    pub contrast_vector: Option<Vec<f64>>,

    /// List contrast: "NUM1,NUM2;DEN1,DEN2" of coefficient names
    #[arg(long, value_name = "NUM;DEN", allow_hyphen_values = true,
        conflicts_with_all = ["numerator"],
        long_help = "Coefficient names for the numerator and the denominator, separated by ';'\n\
            (names within each separated by ','); either side may be empty, e.g.\n\
            \"condition_B_vs_A;condition_C_vs_A\" or \"genotype_KO_vs_WT\".\n\
            Numerator coefficients get the first --list-values weight, denominator ones the second.\n\
            R equivalent: results(dds, contrast = list(c(...), c(...)), listValues = c(1, -1))")]
    pub contrast_list: Option<String>,

    /// Weights of the --contrast-list numerator and denominator [default: 1,-1]
    #[arg(long, value_name = "POS,NEG", value_delimiter = ',', allow_hyphen_values = true,
        requires = "contrast_list")]
    pub list_values: Option<Vec<f64>>,
}
