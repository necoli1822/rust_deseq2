# R references for tximport input and normalization factors
#
# usage: Rscript validation/gen8.R <data dir>
# Writes under <data dir>:
#   tximport_salmon     salmon quant.sf files (isoforms with different effective lengths,
#                       genes with no reads in some or all samples, a transcript missing
#                       from tx2gene), tx2gene and a sample table; R's tximport() ->
#                       DESeqDataSetFromTximport() -> DESeq() results, blind VST
#                       (parametric, local, nsub = 500), blind rlog and normalized counts
#   tximport_matrix     the same txi as -c txi$counts with --avg-tx-length txi$length
#   tximport_poscounts  the same with DESeq(sfType = "poscounts")
#   tximport_lstpm      countsFromAbundance = "lengthScaledTPM" counts (-c --tximport)
#   normfactors*        7 vs 7 with Cook's outliers and a gene x sample normalizationFactors
#                       matrix: Wald (with outlier replacement), LRT, apeglm and normal
#                       shrinkage, fitType local, observation weights, ~batch + condition,
#                       plus blind VST (parametric, local, nsub = 500), rlog and normalized counts
suppressMessages({library(DESeq2); library(tximport); library(apeglm)})
out <- commandArgs(TRUE)[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
q <- function(e) suppressMessages(suppressWarnings(e))

# genes x samples matrix in the count-matrix layout, every double written exactly
wmat <- function(path, m) {
  df <- data.frame(gene_id = rownames(m), check.names = FALSE)
  for (j in seq_len(ncol(m))) df[[colnames(m)[j]]] <- sprintf("%.17g", m[, j])
  write.csv(df, path, row.names = FALSE, quote = FALSE)
}
wmeta <- function(d, md) write.csv(data.frame(sample = rownames(md), md), file.path(d, "metadata.csv"),
                                   row.names = FALSE, quote = FALSE)
res_out <- function(d, res, dds) {
  df <- data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                   lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion)
  write.table(df, file.path(d, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}
tr_out <- function(d, name, m) write.table(m, file.path(d, paste0("r_", name, ".tsv")), sep = "\t", quote = FALSE)
transforms <- function(d, dd) {
  tr_out(d, "vst", assay(q(varianceStabilizingTransformation(dd, blind = TRUE))))
  tr_out(d, "vstlocal", assay(q(varianceStabilizingTransformation(dd, blind = TRUE, fitType = "local"))))
  tr_out(d, "fastvst", assay(q(vst(dd, blind = TRUE, nsub = 500))))
  tr_out(d, "rlog", assay(q(rlog(dd, blind = TRUE))))
  tr_out(d, "norm", counts(q(estimateSizeFactors(dd)), normalized = TRUE))
}
contrast <- c("condition", "trt", "ctrl")

## ---------------------------------------------------------------- salmon
set.seed(81)
ng <- 1500; m <- 8
cond <- rep(c("ctrl", "trt"), each = 4); samples <- sprintf("S%d", seq_len(m))
niso <- sample(1:4, ng, TRUE, prob = c(0.4, 0.3, 0.2, 0.1))
genes <- sprintf("ENSG%05d", seq_len(ng))
genes[1:6] <- c("geneB", "GeneA", "gene_c", "Gene10", "gene2", "abc")   # collation of gene factor levels
gid <- rep(genes, niso); ntx <- length(gid)
txid <- sprintf("ENST%06d", seq_len(ntx))
txlen <- round(runif(ntx, 400, 6000))
efflen <- round(txlen - 180 + runif(ntx, -20, 20), 3)
mu0 <- exp(rnorm(ng, 5.5, 2)); disp <- 0.05 + 3 / mu0
beta <- ifelse(runif(ng) < 0.15, rnorm(ng, 0, 1.5), 0)
sf <- runif(m, 0.6, 1.6)
gc <- sapply(seq_len(m), function(j) rnbinom(ng, mu = mu0 * 2^(beta * (cond[j] == "trt")) * sf[j], size = 1 / disp))
for (g in sample(ng, 60)) gc[g, sample(m, sample(1:3, 1))] <- 0      # no reads in some samples
gc[ng, ] <- 0                                                        # no reads at all
dtu <- runif(ng) < 0.2                                               # isoform switch in trt
numreads <- matrix(0, ntx, m)
for (g in seq_len(ng)) {
  idx <- which(gid == genes[g]); k <- length(idx)
  p0 <- rgamma(k, 1)
  for (j in seq_len(m)) {
    p <- p0 * exp(rnorm(k, 0, 0.3))
    if (dtu[g] && cond[j] == "trt") p <- rev(p)
    if (k > 1 && runif(1) < 0.1) p[sample(k, 1)] <- 0                # an isoform without reads
    numreads[idx, j] <- round(gc[g, j] * p * efflen[idx] / sum(p * efflen[idx]), 3)
  }
}
# a quantified transcript that is not in tx2gene
txid <- c(txid, "ENST_NOGENE"); efflen <- c(efflen, 812.5); txlen <- c(txlen, 1000)
numreads <- rbind(numreads, round(runif(m, 50, 500), 3))
d <- file.path(out, "tximport_salmon")
dir.create(file.path(d, "files"), showWarnings = FALSE, recursive = TRUE)
for (j in seq_len(m)) {
  rate <- numreads[, j] / efflen
  tpm <- rate / sum(rate) * 1e6
  dir.create(file.path(d, "files", samples[j]), showWarnings = FALSE)
  write.table(data.frame(Name = txid, Length = txlen, EffectiveLength = sprintf("%.3f", efflen),
                         TPM = sprintf("%.6f", tpm), NumReads = sprintf("%.3f", numreads[, j])),
              file.path(d, "files", samples[j], "quant.sf"), sep = "\t", quote = FALSE, row.names = FALSE)
}
# tx2gene with a header, a transcript that is not quantified and a duplicated transcript
# (cleanTx2Gene keeps its first row)
t2g <- data.frame(TXNAME = c(txid[seq_len(ntx)], "ENST_NOTQUANT", txid[3]),
                  GENEID = c(gid, "ENSG_NOTQUANT", genes[ng - 1]))
write.csv(t2g, file.path(d, "tx2gene.csv"), row.names = FALSE, quote = FALSE)
st <- data.frame(sample = samples, file = file.path(samples, "quant.sf"), condition = cond)
write.csv(st, file.path(d, "sample_table.csv"), row.names = FALSE, quote = FALSE)

files <- file.path(d, "files", st$file); names(files) <- st$sample
tx2gene <- read.csv(file.path(d, "tx2gene.csv"))
# dropInfReps: these files have no bootstraps (and the check needs jsonlite); counts,
# abundance and length do not depend on it
txi <- q(tximport(files, type = "salmon", tx2gene = tx2gene, dropInfReps = TRUE))
stopifnot(any(txi$abundance == 0), any(rowSums(txi$abundance) == 0), nrow(txi$counts) == ng)
coldata <- data.frame(condition = factor(cond), row.names = samples)
dd <- q(DESeqDataSetFromTximport(txi, coldata, ~condition))
dds <- q(DESeq(dd))
res_out(d, results(dds, contrast = contrast), dds)
transforms(d, dd)
tr_out(d, "nf", normalizationFactors(dds))

## the same txi as matrices: -c counts --avg-tx-length lengths
for (sc in c("tximport_matrix", "tximport_poscounts")) {
  d <- file.path(out, sc); dir.create(d, showWarnings = FALSE)
  wmat(file.path(d, "counts.csv"), txi$counts); wmat(file.path(d, "avg_tx_length.csv"), txi$length)
  wmeta(d, coldata)
}
res_out(file.path(out, "tximport_matrix"), results(dds, contrast = contrast), dds)
dds <- q(DESeq(dd, sfType = "poscounts"))
res_out(file.path(out, "tximport_poscounts"), results(dds, contrast = contrast), dds)

## countsFromAbundance = "lengthScaledTPM": DESeqDataSetFromTximport() uses just the counts
txl <- q(tximport(files, type = "salmon", tx2gene = tx2gene, dropInfReps = TRUE, countsFromAbundance = "lengthScaledTPM"))
d <- file.path(out, "tximport_lstpm"); dir.create(d, showWarnings = FALSE)
wmat(file.path(d, "counts.csv"), txl$counts); wmeta(d, coldata)
dds <- q(DESeq(q(DESeqDataSetFromTximport(txl, coldata, ~condition))))
stopifnot(is.null(normalizationFactors(dds)))
res_out(d, results(dds, contrast = contrast), dds)

## ---------------------------------------------------------------- normalizationFactors
set.seed(82)
n <- 2500; m <- 14
c7 <- rep(c("ctrl", "trt"), each = 7); batch <- rep(c("b1", "b2"), 7)
mu0 <- exp(rnorm(n, 5, 2)); disp <- 0.05 + 4 / mu0
beta <- ifelse(runif(n) < 0.15, rnorm(n, 0, 1.5), 0)
sf <- runif(m, 0.6, 1.6)
# gene- and sample-specific factors, not centered (normalizationFactors<- takes them as given)
nf <- signif(outer(exp(rnorm(n, 0, 0.3)), sf) * exp(matrix(rnorm(n * m, 0, 0.25), n)), 12)
x <- sapply(seq_len(m), function(j) rnbinom(n, mu = mu0 * 2^(beta * (c7[j] == "trt")) * nf[, j], size = 1 / disp))
idx <- sample(n, 60); x[cbind(idx, sample(m, 60, TRUE))] <- x[cbind(idx, sample(m, 60, TRUE))] * 50 + 100
dimnames(x) <- dimnames(nf) <- list(sprintf("g%04d", seq_len(n)), sprintf("s%02d", seq_len(m)))
set.seed(83)
w <- matrix(runif(n * m, 0.2, 1), n, dimnames = dimnames(x)); w[runif(n * m) < 0.05] <- 0.01
md <- data.frame(condition = factor(c7), batch = factor(batch), row.names = colnames(x))
mkdd <- function(design, weights = FALSE) {
  dd <- DESeqDataSetFromMatrix(x, md, design)
  normalizationFactors(dd) <- nf
  if (weights) assays(dd)[["weights"]] <- w
  dd
}
nfdir <- function(sc) {
  d <- file.path(out, sc); dir.create(d, showWarnings = FALSE)
  write.csv(data.frame(gene_id = rownames(x), x, check.names = FALSE), file.path(d, "counts.csv"),
            row.names = FALSE, quote = FALSE)
  wmeta(d, md); wmat(file.path(d, "nf.csv"), nf)
  d
}
d <- nfdir("normfactors")
dds <- q(DESeq(mkdd(~condition)))
stopifnot(sum(mcols(dds)$replace, na.rm = TRUE) > 0)
res_out(d, results(dds, contrast = contrast), dds)
transforms(d, mkdd(~condition))
for (ty in c("apeglm", "normal")) {
  d <- nfdir(paste0("normfactors_", ty))
  res_out(d, q(lfcShrink(dds, coef = "condition_trt_vs_ctrl", type = ty)), dds)
}
d <- nfdir("normfactors_lrt")
dds <- q(DESeq(mkdd(~condition), test = "LRT", reduced = ~1)); res_out(d, results(dds), dds)
d <- nfdir("normfactors_local")
dds <- q(DESeq(mkdd(~condition), fitType = "local")); res_out(d, results(dds, contrast = contrast), dds)
d <- nfdir("normfactors_weights"); wmat(file.path(d, "weights.csv"), w)
dds <- q(DESeq(mkdd(~condition, weights = TRUE))); res_out(d, results(dds, contrast = contrast), dds)
d <- nfdir("normfactors_batch")
dds <- q(DESeq(mkdd(~batch + condition))); res_out(d, results(dds, contrast = contrast), dds)
cat("written", out, "\n")
