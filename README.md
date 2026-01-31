# RustDESeq2

[![Crates.io](https://img.shields.io/crates/v/rust_deseq2.svg)](https://crates.io/crates/rust_deseq2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/necoli1822/rust_deseq2/actions/workflows/ci.yml/badge.svg)](https://github.com/necoli1822/rust_deseq2/actions/workflows/ci.yml)

A pure Rust implementation of the [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) algorithm for differential expression analysis of RNA-seq count data. Produces results identical to R DESeq2 while running **8-12x faster** with **~90x less memory**.

## Features

- **Complete DESeq2 pipeline**: size factor estimation → dispersion estimation → GLM fitting → statistical testing
- **Statistical tests**: Wald test, Likelihood Ratio Test (LRT)
- **Three dispersion fit types**: parametric (Gamma GLM), local (LOESS), mean
- **LFC shrinkage**: normal, apeglm, ashr methods
- **Outlier handling**: Cook's distance detection and replacement
- **Multi-factor designs**: batch correction with categorical and continuous covariates
- **t-distribution p-values**: more conservative testing for small samples (`--use-t`)
- **Transformations**: VST (Variance Stabilizing Transformation), rlog (Regularized Log)
- **Normalization**: median-of-ratios, positive counts, iterative methods
- **Convenience functions**: fpkm, fpm, collapseReplicates
- **100% significance agreement** with R DESeq2 across all validated datasets
- **8-12x faster**, **~90x less memory** than R DESeq2

## Installation

```bash
# From crates.io
cargo install rust_deseq2

# From source
git clone https://github.com/necoli1822/rust_deseq2.git
cd rust_deseq2
cargo build --release
# Binary will be at target/release/rust_deseq2
```

## Quick Start

```bash
# Basic differential expression analysis
rust_deseq2 run -c counts.csv -m metadata.csv -d condition \
  --numerator treated --denominator control -o results.csv

# With batch correction
rust_deseq2 run -c counts.csv -m metadata.csv -d treatment \
  --covariate batch --numerator drug --denominator placebo

# Likelihood Ratio Test
rust_deseq2 run -c counts.csv -m metadata.csv -d treatment \
  --test lrt --reduced "~1" -o results.csv

# With local dispersion fit and t-distribution
rust_deseq2 run -c counts.csv -m metadata.csv -d condition \
  --numerator treated --denominator control --fit-type local --use-t

# LFC shrinkage (apeglm)
rust_deseq2 run -c counts.csv -m metadata.csv -d condition \
  --numerator treated --denominator control --shrinkage --shrinkage-method apeglm

# Variance Stabilizing Transformation (for PCA, heatmaps)
rust_deseq2 vst -c counts.csv -m metadata.csv -d condition -o vst.tsv --blind

# Regularized log transformation
rust_deseq2 rlog -c counts.csv -m metadata.csv -d condition -o rlog.tsv

# Normalize counts only
rust_deseq2 normalize -c counts.csv -o normalized.tsv
```

## CLI Reference

### `run` — Full DESeq2 analysis

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --counts` | Count matrix CSV/TSV file | *required* |
| `-m, --metadata` | Sample metadata CSV file | *required* |
| `-d, --design` | Design variable (column in metadata) | *required* |
| `--covariate` | Categorical covariate for batch correction | — |
| `--continuous` | Continuous covariate | — |
| `--reference` | Reference level (format: factor=level) | alphabetical |
| `--test` | Statistical test: `wald` or `lrt` | `wald` |
| `--numerator` | Numerator level for Wald contrast | — |
| `--denominator` | Denominator level for Wald contrast | — |
| `--reduced` | Reduced model formula for LRT (e.g., `~1`) | — |
| `--fit-type` | Dispersion fit: `parametric`, `local`, `mean` | `parametric` |
| `--use-t` | Use t-distribution for Wald test p-values | `false` |
| `--shrinkage` | Apply LFC shrinkage | `false` |
| `--shrinkage-method` | Shrinkage: `normal`, `apeglm`, `ashr` | `normal` |
| `--replace-outliers` | Replace Cook's distance outliers | `true` |
| `-o, --output` | Output file path | `deseq2_results.csv` |
| `-a, --alpha` | Significance threshold | `0.1` |
| `--maxit` | Max IRLS iterations (GLM + dispersion) | `100` |
| `--beta-tol` | Beta convergence tolerance for GLM | `1e-8` |
| `--min-disp` | Minimum dispersion value | `1e-8` |
| `--disp-tol` | Dispersion convergence tolerance | `1e-6` |
| `--kappa-0` | Initial step size for dispersion line search | `1.0` |
| `--outlier-sd` | Outlier SD threshold for MAP shrinkage | `2.0` |
| `--trim` | Trim fraction for outlier replacement | `0.2` |
| `--upper-quantile` | Upper quantile for beta prior variance | `0.05` |
| `-t, --threads` | Number of threads (0 = auto) | `0` |

### `normalize` — Normalize counts

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --counts` | Count matrix CSV/TSV file | *required* |
| `-o, --output` | Output file path | *required* |
| `-m, --method` | Method: `ratio`, `poscounts`, `iterate` | `ratio` |

### `vst` — Variance Stabilizing Transformation

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --counts` | Count matrix CSV/TSV file | *required* |
| `-m, --metadata` | Sample metadata CSV file | *required* |
| `-d, --design` | Design variable | *required* |
| `-o, --output` | Output file path | `vst_transformed.tsv` |
| `--method` | Fit method: `parametric`, `mean`, `local` | `parametric` |
| `--blind` | Blind to experimental design | `false` |

### `rlog` — Regularized Log Transformation

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --counts` | Count matrix CSV/TSV file | *required* |
| `-m, --metadata` | Sample metadata CSV file | *required* |
| `-d, --design` | Design variable | *required* |
| `-o, --output` | Output file path | `rlog_transformed.tsv` |
| `--blind` | Blind to experimental design | `false` |

## Input File Format

### Count matrix (CSV/TSV, auto-detected)
```
gene_id,sample1,sample2,sample3,...
GENE001,100,200,150,...
GENE002,50,75,60,...
```

### Metadata (CSV)
```
sample,condition
sample1,control
sample2,control
sample3,treated
```

## Benchmark

### Environment

| | Version |
|---|---|
| CPU | Intel Core i9-14900 |
| RAM | 32 GB DDR5 |
| OS | Linux 6.6 (WSL2) |
| Rust | rustc 1.92.0 |
| R | 4.3.3 |
| R DESeq2 | 1.42.1 |

All benchmarks are median of 5 runs on Linux-native filesystem. R timings are `proc.time()` internal (exclude package loading overhead).

### Execution Time

| Dataset | Genes | Samples | R DESeq2 | RustDESeq2 | Speedup |
|---|---:|---:|---:|---:|---:|
| *A. baumannii* GSE151925 | 3,851 | 6 | 0.839 s | 0.090 s | **9.3x** |
| *M. tuberculosis* GSE100097 | 5,220 | 12 | 1.138 s | 0.130 s | **8.8x** |
| *Salmonella* GSE46391 | 4,418 | 6 | 1.167 s | 0.100 s | **11.7x** |

### Memory Usage

| Dataset | R DESeq2 | RustDESeq2 | Reduction |
|---|---:|---:|---:|
| *A. baumannii* GSE151925 | 797 MB | 8 MB | **~100x** |
| *M. tuberculosis* GSE100097 | 797 MB | 15 MB | **~53x** |
| *Salmonella* GSE46391 | 797 MB | 10 MB | **~80x** |

R memory is dominated by the R runtime, S4 object overhead, and garbage collector. Rust allocates only the data structures needed for computation.

## Accuracy

### Validation Against R DESeq2

All 8 validated datasets achieve **100% significance agreement** — every gene is classified identically (significant or not at padj < 0.05) between R and Rust.

| Dataset | Genes | Pearson *r* (log2FC) | Max |padj diff| | Significance Agreement |
|---|---:|---:|---:|---:|
| *A. baumannii* GSE151925 | 3,851 | 1.00000000 | ~1e-7 | 3,851 / 3,851 (100%) |
| *M. tuberculosis* GSE100097 | 5,220 | 1.00000000 | ~1e-7 | 5,220 / 5,220 (100%) |
| *Salmonella* GSE46391 | 4,418 | 1.00000000 | ~1e-7 | 4,418 / 4,418 (100%) |
| *P. aeruginosa* GSE55197 | 6,014 | 1.00000000 | 5.00e-7 | 6,014 / 6,014 (100%) |
| *E. coli* GSE220559 | 4,498 | 1.00000000 | 5.01e-7 | 4,498 / 4,498 (100%) |
| *S. aureus* GSE130777 | 2,648 | 1.00000000 | 3.27e-5 | 2,648 / 2,648 (100%) |
| *S. pneumoniae* GSE137447 | 2,280 | 0.99999200 | 3.80e-2 | 2,280 / 2,280 (100%) |
| *B. subtilis* (simulated) | 4,200 | 1.00000000 | 1.21e-5 | 4,200 / 4,200 (100%) |

Across all 33,129 genes tested, there are **zero discordant significance calls**.

### Why Are There Small Numerical Differences?

The minor differences in log2 fold-change values (typically < 1e-06) are inherent to floating-point arithmetic and do **not** indicate a bug. They arise because:

1. **Different linear algebra backends.** R DESeq2 calls LAPACK/BLAS routines written in Fortran. RustDESeq2 uses pure Rust implementations (ndarray). Even with identical algorithms, different compilers emit different instruction orderings and SIMD vectorizations.

2. **Floating-point non-associativity.** IEEE 754 arithmetic does not obey the associative law: `(a + b) + c` may differ from `a + (b + c)` by the least significant bit. Each matrix multiplication, QR decomposition, or dot product accumulates these rounding differences.

3. **IRLS convergence paths.** DESeq2 fits a negative binomial GLM via Iteratively Reweighted Least Squares. When two implementations reach the convergence threshold at slightly different iteration counts, the final coefficient estimates can diverge at the last few decimal places.

These differences are a fundamental property of numerical computing — even recompiling the *same* code with a different compiler version or on a different CPU architecture can change the last few bits of a floating-point result.

## R Function Mapping

Key equivalences between R DESeq2 and RustDESeq2:

| R DESeq2 | RustDESeq2 | Description |
|----------|------------|-------------|
| `DESeq()` | `run_deseq()` | Full pipeline |
| `estimateSizeFactors()` | `estimate_size_factors()` | Normalization |
| `estimateDispersions()` | `estimate_dispersions()` | Dispersion estimation |
| `nbinomWaldTest()` | `wald_test()` | Wald test |
| `nbinomLRT()` | `likelihood_ratio_test()` | LRT |
| `results()` | `results()` / `results_extended()` | Extract results |
| `resultsNames()` | `results_names()` | Coefficient names |
| `lfcShrink(type="normal")` | `shrink_lfc_normal()` | Normal shrinkage |
| `lfcShrink(type="apeglm")` | `shrink_lfc_apeglm()` | Apeglm shrinkage |
| `lfcShrink(type="ashr")` | `apply_ashr_shrinkage()` | Ashr shrinkage |
| `varianceStabilizingTransformation()` | `vst()` | VST |
| `rlog()` | `rlog()` | Regularized log |
| `fpkm()` | `fpkm()` | FPKM normalization |
| `fpm()` | `fpm()` | FPM normalization |
| `collapseReplicates()` | `collapse_replicates()` | Collapse technical replicates |

## Roadmap: Single-Cell RNA-seq Support

RustDESeq2 currently targets bulk RNA-seq. The following plan covers full single-cell support, implementing all scRNA-seq features from R DESeq2 in pure Rust (no R/glmGamPoi dependency).

### Phase 1: Parameter Exposure (Easy)

Already-implemented algorithms that just need CLI/API configurability.

| Feature | R Parameter | Current State | Work Required |
|---------|------------|---------------|---------------|
| Configurable minmu | `DESeq(minmu=1e-6)` | Hardcoded `0.5` | Propagate through pipeline, add `--min-mu` CLI flag |
| Disable outlier replacement | `minReplicatesForReplace=Inf` | Hardcoded `7` | Add `--min-replicates` CLI flag, support `Inf` |
| Non-integer counts | `skipIntegerMode=TRUE` | Already uses f64 | Add optional integer validation toggle |
| `--single-cell` preset | N/A | N/A | Convenience flag: sets `minmu=1e-6`, `minReplicatesForReplace=Inf`, `sfType=poscounts`, `useT=TRUE` |

### Phase 2: Observation Weights (Medium)

Per-gene-per-sample weight matrix for zero-inflation models (zinbwave integration).

| Feature | R Function | Work Required |
|---------|-----------|---------------|
| Weight storage | `assays(dds)[["weights"]]` | Add `Option<Array2<f64>>` to DESeqDataSet |
| Weighted base means | `getBaseMeansAndVariances()` | Multiply normalized counts by weights |
| Weighted GLM fitting | C++ `fitBeta(weightsSEXP=)` | Modify IRLS to use `W_ij * w_ij` in weight matrix |
| Weighted dispersion estimation | C++ `fitDisp(weightsSEXP=)` | Pass weights through dispersion IRLS |
| Weight-aware df for useT | `rowSums(weights)` | df = sum(weights_per_gene) - n_coefs |
| Degenerate design detection | `getAndCheckWeights()` | Flag genes where weights collapse design rank as allZero |

### Phase 3: glmGamPoi Equivalent (Hard)

Pure Rust reimplementation of the glmGamPoi algorithms, optimized for datasets with many cells and sparse counts.

| Feature | R (glmGamPoi) Function | Description |
|---------|----------------------|-------------|
| Overdispersion MLE | `overdispersion_mle()` | Gene-wise dispersion estimation via closed-form + Newton for NB with minmu=1e-6 |
| Local median trend | `loc_median_fit()` | Dispersion-mean trend via running median (replaces parametric/local/mean) |
| Quasi-likelihood shrinkage | `overdispersion_shrinkage()` | Empirical Bayes shrinkage producing qlDisp values (MLE, fit, MAP) |
| GLM fitting | `glm_gp()` | NB GLM fit using quasi-likelihood framework |
| Quasi-likelihood F-test | `test_de()` | F-statistic based LRT (replaces chi-squared) with quasi-likelihood df |
| New fitType variant | `fitType="glmGamPoi"` | Add `TrendFitMethod::GlmGamPoi` to route through the above |

### Implementation Notes

- All features will be implemented in pure Rust with no external R/Python dependencies
- Phase 1 can be released independently as it only requires parameter plumbing
- Phase 2 enables zinbwave-style workflows where weights are pre-computed externally
- Phase 3 is the most complex but provides the largest performance benefit for scRNA-seq
- glmGamPoi and observation weights are mutually exclusive (same as R)

## License

MIT

## Acknowledgments

This project is an independent Rust reimplementation of the DESeq2 algorithm originally developed by Michael Love, Wolfgang Huber, and Simon Anders.

> Love, M.I., Huber, W., Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15:550. https://doi.org/10.1186/s13059-014-0550-8

The original R implementation is available at https://bioconductor.org/packages/release/bioc/html/DESeq2.html (LGPL >= 3).

The LFC shrinkage methods reference:
- **apeglm**: Zhu, A., Ibrahim, J.G., Love, M.I. (2019). Heavy-tailed prior distributions for sequence count data. *Bioinformatics*, 35:2084-2092.
- **ashr**: Stephens, M. (2017). False discovery rates: a new deal. *Biostatistics*, 18:275-294.
