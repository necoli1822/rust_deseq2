# Validation against R DESeq2

Simulated scenarios, one per code path, each run through R DESeq2 and the
`rust_deseq2` CLI with equivalent settings.

```bash
validation/run.sh "Rscript" /tmp/deseq2-val
# or with a pixi environment providing R + DESeq2 + apeglm + ashr:
validation/run.sh "pixi run --manifest-path tool/deseq2/pixi.toml Rscript" /tmp/deseq2-val
```

| File | Role |
|---|---|
| `gen.R` | Base scenarios: fitTypes, `useT`, `lfcThreshold`, shrinkage, 3-level and LRT designs, batch and continuous covariates, outlier replacement/filtering, poscounts, WT/KO/DKO contrasts, 2 vs 2, VST/rlog/normalize |
| `gen2.R` | Model-matrix cells: 16 samples with batch 4/4 (no replacement in R), unbalanced 2 vs 5 |
| `gen3.R` | VST with `fitType = "local"` and `"mean"` |
| `gen4.R` | Residual df 2-3 with realistic dispersion scatter (simulated prior-variance path) |
| `gen5.R` | R's subsampling `vst(nsub = 1000)` for each fitType, blind and not |
| `gen6.R` | Observation weights (4 vs 4, `useT`, `~batch + condition`) and `sfType = "iterate"` |
| `gen7.R` | `--htseq` input: htseq-count files with counter rows and an `--additional-attr` column, read by `DESeqDataSetFromHTSeqCount` (`~batch + condition`, VST, normalized counts) |
| `gen8.R` | tximport and normalization factors: simulated salmon `quant.sf` files (isoforms, genes without reads in some or all samples, a transcript missing from tx2gene) through `tximport` + `DESeqDataSetFromTximport` (`--salmon`; the same txi as `-c --avg-tx-length`, with `poscounts`; `lengthScaledTPM` counts with `--tximport`), and a 7 vs 7 data set with Cook's outliers and a `normalizationFactors` matrix (`--norm-factors`: Wald with outlier replacement, LRT, apeglm and normal shrinkage, `local`, `--weights`, `~batch + condition`); VST (parametric, local, `nsub`), rlog and normalized counts for both |
| `gen9.R` | `--quant-type` kallisto, RSEM (isoforms with effective lengths below 1, and genes.results), StringTie (`readLength = 100`), piscem and oarfish files of one simulated data set through `tximport` + `DESeqDataSetFromTximport` |
| `run_rust.sh` | Runs the CLI on every scenario |
| `compare.py` | Max abs difference per column, NA-pattern and padj < 0.1 mismatches; max abs difference of every `r_<name>.tsv` matrix (VST, rlog, normalized counts) with its `rust_<name>_<tag>.tsv` |
| `worst.py` | Genes with the largest difference in one column of one scenario |

`real/` runs the same comparison on public GEO data (GSE151925, GSE46391, GSE130777,
GSE100097, GSE55197, GSE220559): `validation/real/run.sh "<Rscript command>" <work dir>`
downloads the data (`fetch.sh`), builds the R references (`realdata.R`) and runs the CLI
(`real_run.sh`).

Columns of the `compare.py` table: `dLFC`, `dSE`, `dpadj` are max absolute
differences; `dlog10p` is the max difference of log10 p-values; `NAmis` counts
genes whose padj is NA in only one of R and Rust; `sigDis` counts genes called
significant (padj < 0.1) by only one; `nSigR` is R's number of significant genes.

Reference versions used for the results in the top-level README: R 4.5.3,
DESeq2 1.50.2, apeglm 1.32.0, ashr 2.2-63, RcppNumerical 0.7-0, tximport 1.38.2 (without
readr, so `quant.sf` files are read with `read.delim`).

`h5bench/` compares pure-Rust HDF5 readers on kallisto `abundance.h5` files against rhdf5
(accuracy, speed, build cost); results and recommendation in `h5bench/README.md`.
