# R references for --quant input from quantifiers other than salmon
#
# usage: Rscript validation/gen9.R <data dir>
# One simulated transcript-level data set (as in gen8.R: isoforms with different
# effective lengths, genes without reads in some or all samples, a transcript missing
# from tx2gene) written in each quantifier's format, then R's
# tximport(type = ...) -> DESeqDataSetFromTximport() -> DESeq() results:
#   quant_kallisto      abundance.tsv
#   quant_rsem          *.isoforms.results, with effective lengths below 1
#   quant_rsem_genes    *.genes.results (gene level, no tx2gene)
#   quant_stringtie     t_data.ctab (counts = cov * length / readLength, readLength 100)
#   quant_piscem        .quant
#   quant_oarfish       .quant (abundance = num_reads)
suppressMessages({library(DESeq2); library(tximport)})
out <- commandArgs(TRUE)[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
q <- function(e) suppressMessages(suppressWarnings(e))
res_out <- function(d, res, dds) {
  df <- data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                   lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion)
  write.table(df, file.path(d, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

set.seed(91)
ng <- 1200; m <- 8
cond <- rep(c("ctrl", "trt"), each = 4); samples <- sprintf("S%d", seq_len(m))
niso <- sample(1:4, ng, TRUE, prob = c(0.4, 0.3, 0.2, 0.1))
genes <- sprintf("ENSG%05d", seq_len(ng))
genes[1:4] <- c("geneB", "GeneA", "gene_c", "abc")
gid <- rep(genes, niso); ntx <- length(gid)
txid <- sprintf("ENST%06d", seq_len(ntx))
txlen <- round(runif(ntx, 400, 6000))
efflen <- round(txlen - 180 + runif(ntx, -20, 20), 2)
mu0 <- exp(rnorm(ng, 5.5, 2)); disp <- 0.05 + 3 / mu0
beta <- ifelse(runif(ng) < 0.15, rnorm(ng, 0, 1.5), 0)
sf <- runif(m, 0.6, 1.6)
gc <- sapply(seq_len(m), function(j) rnbinom(ng, mu = mu0 * 2^(beta * (cond[j] == "trt")) * sf[j], size = 1 / disp))
for (g in sample(ng, 50)) gc[g, sample(m, sample(1:3, 1))] <- 0
gc[ng, ] <- 0
numreads <- matrix(0, ntx, m)
for (g in seq_len(ng)) {
  idx <- which(gid == genes[g]); k <- length(idx)
  p0 <- rgamma(k, 1)
  for (j in seq_len(m)) {
    p <- p0 * exp(rnorm(k, 0, 0.3))
    if (k > 1 && runif(1) < 0.1) p[sample(k, 1)] <- 0
    numreads[idx, j] <- round(gc[g, j] * p * efflen[idx] / sum(p * efflen[idx]), 2)
  }
}
txid <- c(txid, "ENST_NOGENE"); efflen <- c(efflen, 812.5); txlen <- c(txlen, 1000)
numreads <- rbind(numreads, round(runif(m, 50, 500), 2))
# three isoforms of multi-isoform genes without reads (given effective lengths < 1 for RSEM)
low <- which(gid %in% genes[niso >= 2])[c(1, 5, 9)]; numreads[low, ] <- 0
tpm <- apply(numreads / efflen, 2, function(r) r / sum(r) * 1e6)
t2g <- data.frame(TXNAME = txid[seq_len(ntx)], GENEID = gid)
st <- function(f) data.frame(sample = samples, file = f, condition = cond)
coldata <- data.frame(condition = factor(cond), row.names = samples)
num <- function(x, d = 6) sprintf(paste0("%.", d, "f"), x)

run <- function(sc, type, fname, write_one, tx2gene = t2g, ...) {
  d <- file.path(out, sc); dir.create(file.path(d, "files"), showWarnings = FALSE, recursive = TRUE)
  for (j in seq_len(m)) {
    f <- file.path(d, "files", fname[j]); dir.create(dirname(f), showWarnings = FALSE, recursive = TRUE)
    write.table(write_one(j), f, sep = "\t", quote = FALSE, row.names = FALSE)
  }
  if (!is.null(tx2gene)) write.csv(tx2gene, file.path(d, "tx2gene.csv"), row.names = FALSE, quote = FALSE)
  write.csv(st(fname), file.path(d, "sample_table.csv"), row.names = FALSE, quote = FALSE)
  files <- file.path(d, "files", fname); names(files) <- samples
  txi <- q(tximport(files, type = type, tx2gene = tx2gene, dropInfReps = TRUE, ...))
  dds <- q(DESeq(q(DESeqDataSetFromTximport(txi, coldata, ~condition))))
  res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
  invisible(txi)
}

run("quant_kallisto", "kallisto", file.path(samples, "abundance.tsv"), function(j)
  data.frame(target_id = txid, length = txlen, eff_length = num(efflen, 2),
             est_counts = num(numreads[, j], 2), tpm = num(tpm[, j])))

# RSEM: a few transcripts with effective length below 1 (no reads), set to 1 by tximport
rl <- efflen; rl[low] <- c(0, 0.4, 0.95)
rtpm <- tpm; rtpm[low, ] <- 0
run("quant_rsem", "rsem", sprintf("%s.isoforms.results", samples), function(j)
  data.frame(transcript_id = txid, gene_id = c(gid, "NA"), length = txlen, effective_length = num(rl, 2),
             expected_count = num(numreads[, j], 2), TPM = num(rtpm[, j], 2), FPKM = num(rtpm[, j] / 2, 2),
             IsoPct = "0.00"))

# RSEM genes.results: gene-level rows, positive effective lengths (DESeqDataSetFromTximport
# requires lengths > 0)
gsum <- function(x) rowsum(x[seq_len(ntx), , drop = FALSE], factor(gid, levels = genes))
gcount <- gsum(numreads); gtpm <- gsum(tpm)
glen <- gsum(tpm * efflen) / gtpm
glen[!is.finite(glen)] <- rep(tapply(efflen[seq_len(ntx)], factor(gid, levels = genes), mean), m)[!is.finite(glen)]
run("quant_rsem_genes", "rsem", sprintf("%s.genes.results", samples), function(j)
  data.frame(gene_id = genes, `transcript_id(s)` = "tx", length = num(glen[, j] + 180, 2),
             effective_length = num(glen[, j], 2), expected_count = num(gcount[, j], 2),
             TPM = num(gtpm[, j], 2), FPKM = num(gtpm[, j] / 2, 2), check.names = FALSE), tx2gene = NULL)

# StringTie: cov = reads * readLength / length, so tximport's counts are cov * length / readLength
run("quant_stringtie", "stringtie", file.path(samples, "t_data.ctab"), function(j)
  data.frame(t_id = seq_along(txid), chr = "chr1", strand = "+", start = 1, end = txlen, t_name = txid,
             num_exons = 3, length = txlen, gene_id = c(gid, "."), gene_name = ".",
             cov = num(numreads[, j] * 100 / txlen), FPKM = num(tpm[, j] / 3)), readLength = 100)

run("quant_piscem", "piscem", sprintf("%s.quant", samples), function(j)
  data.frame(target_name = txid, len = txlen, eelen = num(efflen, 2), tpm = num(tpm[, j]),
             ecount = num(numreads[, j], 2)))

run("quant_oarfish", "oarfish", sprintf("%s.quant", samples), function(j)
  data.frame(tname = txid, len = txlen, num_reads = num(numreads[, j], 2)))
cat("written", out, "\n")
