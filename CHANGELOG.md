# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
