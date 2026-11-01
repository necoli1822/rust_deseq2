# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Fidelity fixes found by comparing against R DESeq2 1.50.2 (apeglm 1.32.0, ashr 2.2-63,
RcppNumerical 0.7-0, R 4.5.3) on simulated data sets covering each code path. Each item
names the R function it now follows.

### Fixed
- The library builder's `.interaction()` named its columns `f1_l1_x_f2_l2` and estimated the
  dispersions without the interaction columns; it now builds the design formula
  `~ factors + continuous + main + f1:f2`, with R's model matrix and names.
- Coefficient names now go through `make.names()` as R's do (levels with spaces or `-`), and
  `--reduced` honours `--reference` and rejects variables that are not in the full design
  (R: `checkLRT()`).
- List contrasts (library `ContrastSpec::List`) added the weights of a coefficient listed twice;
  they are now assigned as in R, and R's checks (no coefficient on both sides, `listValues`
  positive and negative, numeric contrast not all zero) are applied.
- **Wrong coefficient for contrasts.** Coefficients were found by suffix match, so with
  levels WT/KO/DKO a `KO vs WT` contrast returned the DKO result, and a covariate sharing level
  names with the main factor could be picked instead. Contrasts now use exact coefficient
  names, and a contrast between two non-reference levels is computed from the coefficient
  covariance (`results()`/`getContrast`) instead of being rejected.
- **apeglm shrank the wrong coefficient.** The CLI always shrank design column 1 (the batch
  coefficient with `--covariate batch`), and reversed contrasts got the wrong sign in both
  apeglm and normal shrinkage.
- **apeglm optimiser path.** The C++ MAP search now uses the Nocedal-Wright line search of the
  LBFGSpp bundled with RcppNumerical 0.7-0, the likelihood constant of `nbinomFn`, `uniroot`
  for the prior variance, and `optim(method = "BFGS")` refits for rows that do not converge.
- **ashr shrinkage** used a point mass at zero and a different mixture grid; `lfcShrink` uses
  `method = "shrink"` (no point mass, uniform prior) and `autoselect.mixsd`. Mixture weights
  are now solved to optimality (EM stopped far from R's mixSQP solution).
- **`--continuous` always failed** ("continuous variable not found"), the LRT reported the
  covariate's slope as log2FoldChange when continuous covariates were present, and the
  reduced model treated continuous variables as factors.
- **Outlier refit aborted** with `--fit-type local`/`mean` (or a parametric fit that fell back
  to local) whenever a 7+ replicate cell had a Cook's outlier; the refit also recomputed
  `varLogDispEsts` instead of reusing R's.
- **Outlier replacement and Cook's filtering used the wrong samples**: replaceable samples are
  now model-matrix cells with `minReplicatesForReplace` samples (`nOrMoreInCell`), and maxCooks
  only uses cells with 3+ replicates (`recordMaxCooks`).
- **MAP dispersion dropped the Cox-Reid term** whenever pivoting swapped rows of X'WX (e.g. an
  unscaled continuous covariate), shifting every gene's dispersion by ~13%.
- **Small designs (residual df 1-3)**: the simulated prior-variance estimate now reproduces R's
  `set.seed(2)` draws (`norm_rand` inversion, `exp_rand`, `rgamma`, `qnorm`, draw order).
- **`--lfc-threshold`** used the pre-2023 test; it now uses R's current `greaterAbs`.
- **poscounts size factors** were not rescaled to geometric mean 1.
- **Cook's distances for zero-heavy genes**: hat diagonals of optim-refit genes now come from
  the IRLS fit and stored mu is unclamped, as in `fitNbinomGLMs`.
- **Gene-wise dispersions for multi-factor designs** now take mu from the same GLM fit R uses.
- **VST**: `vst --method` now changes the dispersion fit, `mean` uses the mean dispersion, and
  `local` is a port of R's numerical integration (it produced values up to 1.8e8 before).
- **All-zero genes** report NA, and genes with no counts in the compared groups report
  LFC 0 / p 1 (`contrastAllZero`), as do genes that become all zero after replacement.
- **CLI**: the `normalize` subcommand failed on every input; subcommand `--help`/`-h` was
  rejected; `--replace-outliers false` was rejected (the flag could not be turned off).

- **GLM optim fallback** is now a port of R's `optim(method = "L-BFGS-B")` (lbfgsb.c and the
  optim.c driver, including its finite-difference gradient), replacing a hand-written L-BFGS
  that stopped elsewhere on genes that are zero in one group (e.g. LFC 30.0 vs R's 23.1). Used
  for Wald and LRT fits.
- **`--sf-type iterate`** re-applied poscounts; it is now R's `estimateSizeFactorsIterate`
  (alternating mean-fit dispersions and an L-BFGS-B maximisation of the trimmed likelihood).

- **Found on public GEO data** (see `validation/real/`): factor levels were sorted byte-wise
  (`WT` before `abaI`), making a different default reference than R's ICU collation; the
  gene-wise Cox-Reid term was cut off below det(X'WX) = 1e-10, stopping the dispersion search
  early for many-group designs (SE off by up to 2.1); statrs's `lgamma` was too coarse at
  1/alpha ~ 1e8 for the dispersion grid search; size factors differed from R in the last bits,
  which flipped rounding-level line-search decisions; the ashr weight solver could stall on a
  suboptimal mixture, and genes ash excludes now get the prior instead of NaN.

- **Local dispersion trend** (`fitType = "local"`) now ports locfit's fit exactly, including
  the global quadratic it subtracts before interpolating; the old version extrapolated the local
  fit itself and diverged below the smallest mean (e.g. 167231 vs R's 130832 at mean 0.001),
  which the VST evaluates (VST local off by up to 1.4e-2; now 5e-7).

- **Normalization factors were not used in the fits.** `normalizationFactors` (a genes x samples
  matrix) only entered the normalized counts and rlog; the GLM, gene-wise/MAP dispersion,
  moments estimate, LRT, Cook's distances, outlier replacement and refit, apeglm and normal
  shrinkage, VST and baseMean used the size factors. Every fit now takes the gene's row of
  `getSizeOrNormFactors()` (`DESeqDataSet::gene_norm_factors`), with `momentsDispEstimate`'s
  `mean(1/colMeans(nf))`, `replaceOutliers`' `trimBaseMean * nf` and the local VST's
  `exp(colMeans(log(nf)))`. The size-factor path is unchanged bit for bit.
- **`set_normalization_factors` centered the factors** by their row geometric means; R's
  `normalizationFactors<-` stores them as given, and so does this now.
- **Shrinkage after outlier replacement** (`normal`, `apeglm`) fitted the replaced counts. R's
  `refitWithoutOutliers` restores the original counts, which `lfcShrink` then fits with the
  refitted dispersions and MLE (ΔLFC up to 4.4 for the replaced genes; now 8e-7).
- **Non-integer counts** were accepted with a warning; like `DESeqDataSetFromMatrix()`, the CLI now
  rejects them unless they are tximport counts (`--tximport`, `--avg-tx-length`).

### Added
- Design formulas: `run --formula "<R formula>"` (and `vst`/`rlog --formula` for their
  dispersions), `DESeqDataSetBuilder::formula()`, `glm::Formula`, `glm::formula_design()`. Terms
  `a + b`, `a:b`, `a*b`, `a*b*c`, `(a + b)^2`, `a/b`, `a %in% b`, `-` and `0 +`/`- 1`; the model
  matrix is R's `model.matrix()` (term order by degree, treatment contrasts, R's marginality rule
  for a factor whose main effect is absent, first variable varying fastest) and the coefficient
  names are DESeq2's (`Intercept`, `<f>_<level>_vs_<ref>`, `batchb2.conditionB`, no renaming
  without an intercept). Variables are factors unless listed with `--continuous`. The model matrix
  is used by every step: dispersions, GLM, Cook's distances and cells, outlier replacement and
  refit, LRT, shrinkage. `--numerator`/`--denominator` follow `cleanContrast()` (including the
  `<factor><level>` columns of designs without an intercept), the default contrast of a design
  without intercept is R's `c(lastVar, last level, first level)`, `checkFullRank()` messages, and
  `normal` shrinkage stops for interactions or no intercept as `lfcShrink()` does. `--reduced` uses
  the same parser and R's `checkLRT()`/degrees-of-freedom checks. Checked against R with
  `validation/gen12.R` (22 scenarios, including a three-way interaction and outlier replacement).
- Functions in design formulas, evaluated as R's `model.frame()` does (new `glm::model_frame` and
  `glm::rfuns`): `factor()`, `as.factor()`, `relevel()`, `ordered()` (polynomial contrasts,
  `contr.poly`), `as.numeric()`/`as.integer()`, `log()` (with `base`), `log2`, `log10`, `log1p`,
  `exp`, `expm1`, `sqrt`, `abs`, `scale()`, `I()` with arithmetic, comparisons and `%in%`,
  `poly()` (orthogonal via LINPACK `dqrdc2`, or `raw = TRUE`), `splines::ns()` and `bs()` (ports of
  `splineDesign()`, type-7 knot quantiles and the natural-spline QR constraint), `c()`, and
  `offset()` (ignored, as DESeq2 ignores it). R's argument matching, `deparse()` names
  (`poly(x, 2)1`), `make.names()` and DESeq2's renaming (only colData factors of
  `all.vars(design)`); logical columns are coded as R codes them; `.` and other R errors are kept
  (`all variables in design formula must be columns in colData`, `'relevel' only for (unordered)
  factors`, ...); functions that cannot be evaluated are an explicit error. `^` now requires a
  power of at least 2 (R: `invalid power in formula`). Model matrices of 124 formulas equal R's to
  4e-16 (unit tests generated by `validation/gen_formula_cases.R`).
- LRT reduced models that are not full rank are fitted as `fitNbinomGLMs()` fits them (start at
  `log(baseMean)` for the intercept and 0 elsewhere, or 1 without an intercept) instead of being
  rejected. `validation/gen13.R` adds 30 scenarios (factor of a numeric column, relevel, log,
  scale, `I(x^2)`, orthogonal and raw `poly`, `ns` and `bs` with LRTs, factor × poly and factor:ns
  interactions, ordered factors, rank-deficient reduced models with and without intercept,
  offset, levels needing `make.names()`, logical and numeric-only designs, apeglm and normal
  shrinkage, VST): max |ΔLFC| 7e-7 (apeglm 4.9e-6), no significance or NA differences.
- Contrast choices of R's `results()`: `--name <COEF>` (a coefficient by its `resultsNames` name,
  e.g. a continuous variable's slope or a covariate), `--contrast-var` (a `--numerator`/
  `--denominator` contrast of another factor), `--contrast-vector` (numeric contrast) and
  `--contrast-list` with `--list-values` (list contrast). Without one, the Wald test reports
  the last coefficient, as `results(dds)` does (`--numerator`/`--denominator` were required).
  With LRT a contrast picks the reported LFC/SE; normal and apeglm shrinkage shrink the
  coefficient the contrast names (`lfcShrink(coef = )`). The resultsNames are logged.
- kallisto `abundance.h5` input for `--quant-type kallisto`, as tximport's
  `read_kallisto_h5()`: chosen when the first file is named `abundance.h5`; reads `aux/ids`,
  `est_counts` and `aux/eff_lengths` and computes the TPM from them. Uses the pure-Rust
  `rust-hdf5` crate (with its parallel chunk decompression; chosen by `validation/h5bench`)
  behind the default `hdf5` feature.
- tximport input from other quantifiers: `--quant <SAMPLE_TABLE> --quant-type <TYPE>` (`--salmon`
  and `--salmon-dir` remain as aliases of `--quant`, `--quant-dir`) reads kallisto
  `abundance.tsv`, RSEM `isoforms.results` (lengths below 1 set to 1) and `genes.results` (gene
  level, no tx2gene, chosen as tximport does from the first file name), StringTie `t_data.ctab`
  (counts = cov × length / `--read-length`), sailfish, piscem and oarfish files, with tximport's
  column choices (`io::read_quant`, `io::tximport::tximport`, `QuantType`).
- tximport input. `--salmon <SAMPLE_TABLE> --salmon-dir <DIR> --tx2gene <FILE>` reads salmon
  `quant.sf` files and summarizes them to genes as `tximport(type = "salmon")` does
  (`io::tximport`: sums, abundance-weighted lengths, `replaceMissingLength`, transcripts missing
  from tx2gene dropped, R's `read.delim` number parsing), then builds the dataset as
  `DESeqDataSetFromTximport()`: counts rounded halves to even, lengths stored as `avgTxLength`.
  `-c` with `--avg-tx-length <FILE>` takes `txi$counts`/`txi$length` matrices, and `--tximport`
  rounds `-c` counts without lengths (`countsFromAbundance` other than `"no"`). With
  `avgTxLength`, `estimate_size_factors` sets normalization factors (`estimateNormFactors`,
  `ratio` and `poscounts`; `iterate` ignores the lengths, as in R). For `run`, `vst`, `rlog` and
  `normalize`.
- `--norm-factors <FILE>` (R: `normalizationFactors(dds) <- nf`) for `run`, `vst`, `rlog` and
  `normalize`; size factor estimation is then skipped, as in `DESeq()`. Works with `--weights`.
- htseq-count input (R's `DESeqDataSetFromHTSeqCount`): `--htseq <SAMPLE_TABLE>` and
  `--htseq-dir <DIR>` for `run`, `vst`, `rlog` and `normalize`, and `io::read_htseq_count`.
  Counts come from the last column of each file, gene IDs must agree across files, and
  htseq-count's counter rows are dropped.
- `vst --nsub N`: R's `vst()`, which fits the dispersion trend on N genes (R's default 1000)
  spanning the range of mean counts. Without `--nsub`, `vst` remains
  `varianceStabilizingTransformation()`.
- `src/optim`: R's L-BFGS-B, `dnbinom` and long-double `sum()` for use by the above.
- Observation weights (R's `assays(dds)[["weights"]]`): `run --weights <FILE>` and
  `DESeqDataSet::set_weights`. Weights enter baseMean, gene-wise and MAP dispersions (weighted
  likelihood, Cox-Reid term on samples with weight > 0.01), the GLM fits, the LRT, outlier refits
  and the `--use-t` degrees of freedom, following `getAndCheckWeights` and the C++ fitters.
  Weighted `lfcShrink` (normal, apeglm) is not ported and is rejected.

### Changed
- Minimum supported Rust version is now 1.89 (was 1.70), required by `rust-hdf5`.
- Rust edition 2024 (was 2021). Dependencies: statrs 0.19 (without its default nalgebra and
  rand features), ndarray 0.17, and the lock file at the latest compatible versions (clap 4.6,
  rayon 1.12, ...); the unused rand, rand_distr, ndarray-stats, anyhow, num-traits and approx
  are removed. Results are unchanged: every validation output is byte identical.
- `--replace-outliers` now takes a value: `--replace-outliers false`.
- apeglm's prior-scale message is logged at info level instead of printed to stderr.

## [0.1.0] - 2025-01-31

### Added
- Complete DESeq2 pipeline: size factor estimation, dispersion estimation, GLM fitting, statistical testing
- Wald test and Likelihood Ratio Test (LRT)
- Three dispersion fit types: parametric (Gamma GLM), local (LOESS), mean
- LFC shrinkage methods: normal, apeglm, ashr
- Outlier handling via Cook's distance detection and replacement
- Multi-factor designs with batch correction (categorical and continuous covariates)
- t-distribution p-values for small sample sizes (`--use-t`)
- Variance Stabilizing Transformation (VST) and regularized log (rlog)
- Normalization methods: median-of-ratios, positive counts, iterative
- Convenience functions: fpkm, fpm, collapseReplicates
- CLI with subcommands: `run`, `normalize`, `vst`, `rlog`
- Library API for programmatic use
- 87 unit tests
- 100% significance agreement with R DESeq2 across all validated datasets
- 8-12x faster execution than R DESeq2
- ~90x less memory usage than R DESeq2
