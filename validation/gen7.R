# R reference for `--htseq` input: DESeqDataSetFromHTSeqCount
#
# usage: Rscript validation/gen7.R <data dir>
# Writes <data dir>/htseq: htseq-count files (with the __no_feature/... counter rows,
# one sample in --additional-attr layout), a sample table, and R's results from
# DESeqDataSetFromHTSeqCount(~ batch + condition) + DESeq, plus its blind VST and
# normalized counts.
suppressMessages(library(DESeq2))
out <- file.path(commandArgs(TRUE)[1], "htseq")
dir.create(file.path(out, "files"), showWarnings = FALSE, recursive = TRUE)

set.seed(11)
n <- 2000; m <- 8
mu0 <- exp(rnorm(n, 5, 2)); disp <- 0.05 + 4 / mu0
beta <- ifelse(runif(n) < 0.15, rnorm(n, 0, 1.5), 0)
cond <- rep(c("ctrl", "trt"), each = 4); batch <- rep(c("b1", "b2"), 4)
sf <- runif(m, 0.6, 1.6)
x <- sapply(seq_len(m), function(j) rnbinom(n, mu = mu0 * 2^(beta * (cond[j] == "trt")) *
                                              2^(0.5 * (batch[j] == "b2")) * sf[j], size = 1 / disp))
genes <- sprintf("ENSG%06d", 1:n); samples <- sprintf("S%d", 1:m)

counters <- c("__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique")
files <- sprintf("%s.htseq.txt", samples)
for (j in seq_len(m)) {
  f <- file.path(out, "files", files[j])
  if (j == 2) {
    # htseq-count --additional-attr gene_name: the count is the last column
    tab <- data.frame(c(genes, counters), c(paste0("name", 1:n), rep("", 5)), c(x[, j], sample(1e5, 5)))
    tab[n + 1:5, 2] <- NA
    write.table(tab, f, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE, na = "")
    next
  } else {
    tab <- data.frame(c(genes, counters), c(x[, j], sample(1e5, 5)))
  }
  write.table(tab, f, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}
st <- data.frame(sampleName = samples, fileName = files, condition = cond, batch = batch)
write.csv(st, file.path(out, "sample_table.csv"), row.names = FALSE, quote = FALSE)

dds <- suppressWarnings(DESeqDataSetFromHTSeqCount(st, file.path(out, "files"), ~ batch + condition))
stopifnot(nrow(dds) == n, identical(rownames(dds), genes), all(counts(dds) == x))
dds <- suppressMessages(DESeq(dds))
res <- results(dds)
df <- data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                 lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion)
write.table(df, file.path(out, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
v <- assay(varianceStabilizingTransformation(dds, blind = TRUE))
write.table(data.frame(gene_id = rownames(v), v, check.names = FALSE), file.path(out, "r_vst.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(gene_id = rownames(dds), counts(dds, normalized = TRUE), check.names = FALSE),
            file.path(out, "r_norm.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat("written", out, "\n")
