# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Fidelity fixes found by comparing against R DESeq2 1.50.2 (apeglm 1.32.0, ashr 2.2-63,
RcppNumerical 0.7-0, R 4.5.3) on simulated data sets covering each code path. Each item
names the R function it now follows.

### Fixed
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

### Added
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
