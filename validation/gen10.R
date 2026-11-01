# R reference for --quant-type kallisto with abundance.h5 (HDF5) files
#
# usage: Rscript validation/gen10.R <data dir>
# Writes <data dir>/quant_kallisto_h5: one simulated transcript-level data set (as in gen9.R)
# as kallisto abundance.h5 files written with rhdf5 in kallisto's layout (aux/ids as
# fixed-length strings, est_counts, aux/eff_lengths, aux/lengths, aux/num_bootstrap; chunked,
# deflate), a tx2gene and a sample table, and R's tximport(type = "kallisto") ->
# DESeqDataSetFromTximport() -> DESeq() results. tximport reads HDF5 because the files are
# named abundance.h5, and computes the TPM itself from est_counts / eff_lengths.
suppressMessages({library(DESeq2); library(tximport); library(rhdf5)})
out <- file.path(commandArgs(TRUE)[1], "quant_kallisto_h5")
dir.create(file.path(out, "files"), showWarnings = FALSE, recursive = TRUE)
q <- function(e) suppressMessages(suppressWarnings(e))

set.seed(101)
ng <- 1200; m <- 8
cond <- rep(c("ctrl", "trt"), each = 4); samples <- sprintf("S%d", seq_len(m))
niso <- sample(1:4, ng, TRUE, prob = c(0.4, 0.3, 0.2, 0.1))
genes <- sprintf("ENSG%05d", seq_len(ng)); genes[1:3] <- c("geneB", "GeneA", "abc")
gid <- rep(genes, niso); ntx <- length(gid)
txid <- sprintf("ENST%06d.%d", seq_len(ntx), sample(1:9, ntx, TRUE))
txlen <- round(runif(ntx, 400, 6000))
efflen <- txlen - 180 + runif(ntx, -20, 20)          # full doubles, as kallisto writes them
mu0 <- exp(rnorm(ng, 5.5, 2)); disp <- 0.05 + 3 / mu0
beta <- ifelse(runif(ng) < 0.15, rnorm(ng, 0, 1.5), 0)
sf <- runif(m, 0.6, 1.6)
gc <- sapply(seq_len(m), function(j) rnbinom(ng, mu = mu0 * 2^(beta * (cond[j] == "trt")) * sf[j], size = 1 / disp))
for (g in sample(ng, 50)) gc[g, sample(m, sample(1:3, 1))] <- 0
gc[ng, ] <- 0
est <- matrix(0, ntx, m)
for (g in seq_len(ng)) {
  idx <- which(gid == genes[g]); k <- length(idx); p0 <- rgamma(k, 1)
  for (j in seq_len(m)) {
    p <- p0 * exp(rnorm(k, 0, 0.3))
    if (k > 1 && runif(1) < 0.1) p[sample(k, 1)] <- 0
    est[idx, j] <- gc[g, j] * p * efflen[idx] / sum(p * efflen[idx])   # EM-like non-integer counts
  }
}
txid <- c(txid, "ENST_NOGENE"); efflen <- c(efflen, 812.5); txlen <- c(txlen, 1000)
est <- rbind(est, runif(m, 50, 500))

for (j in seq_len(m)) {
  dir.create(file.path(out, "files", samples[j]), showWarnings = FALSE)
  f <- file.path(out, "files", samples[j], "abundance.h5")
  unlink(f); h5createFile(f); h5createGroup(f, "aux")
  n <- length(txid)
  h5createDataset(f, "aux/ids", n, storage.mode = "character", size = max(nchar(txid)) + 1,
                  chunk = 512, level = 6)
  h5write(txid, f, "aux/ids")
  for (nm in c("est_counts", "aux/eff_lengths")) {
    h5createDataset(f, nm, n, storage.mode = "double", chunk = 512, level = 6)
  }
  h5write(est[, j], f, "est_counts"); h5write(efflen, f, "aux/eff_lengths")
  h5createDataset(f, "aux/lengths", n, storage.mode = "integer", chunk = 512, level = 6)
  h5write(as.integer(txlen), f, "aux/lengths")
  h5write(0L, f, "aux/num_bootstrap")
  h5closeAll()
}
write.csv(data.frame(TXNAME = txid[seq_len(ntx)], GENEID = gid), file.path(out, "tx2gene.csv"),
          row.names = FALSE, quote = FALSE)
st <- data.frame(sample = samples, file = file.path(samples, "abundance.h5"), condition = cond)
write.csv(st, file.path(out, "sample_table.csv"), row.names = FALSE, quote = FALSE)

files <- file.path(out, "files", st$file); names(files) <- samples
txi <- q(tximport(files, type = "kallisto", tx2gene = read.csv(file.path(out, "tx2gene.csv")), dropInfReps = TRUE))
stopifnot(nrow(txi$counts) == ng, any(txi$abundance == 0))
cd <- data.frame(condition = factor(cond), row.names = samples)
dds <- q(DESeq(q(DESeqDataSetFromTximport(txi, cd, ~condition))))
res <- results(dds, contrast = c("condition", "trt", "ctrl"))
write.table(data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                       lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion),
            file.path(out, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat("written", out, "\n")
