# Small designs (residual df 2-3) with extra dispersion variability, so the
# simulated prior-variance path gives dispPriorVar above its 0.25 floor.
suppressMessages(library(DESeq2))
out <- commandArgs(TRUE)[1]
simv <- function(cond, seed, n = 3000) {
  set.seed(seed)
  m <- length(cond)
  mu0 <- exp(rnorm(n, 5, 2)); disp <- (0.05 + 4 / mu0) * exp(rnorm(n, 0, 1.2))
  beta <- ifelse(runif(n) < 0.15, rnorm(n, 0, 1.5), 0)
  sf <- runif(m, 0.6, 1.6)
  x <- sapply(seq_len(m), function(j) rnbinom(n, mu = mu0 * 2^(beta * (cond[j] == "trt")) * sf[j], size = 1 / disp))
  dimnames(x) <- list(sprintf("g%04d", 1:n), sprintf("s%02d", 1:m)); x
}
for (sc in list(list("small_var_2v2", rep(c("ctrl", "trt"), each = 2), 31),
                list("small_var_2v3", c("ctrl", "ctrl", "trt", "trt", "trt"), 32))) {
  d <- file.path(out, sc[[1]]); dir.create(d, showWarnings = FALSE)
  x <- simv(sc[[2]], sc[[3]]); md <- data.frame(condition = factor(sc[[2]]))
  write.csv(data.frame(gene_id = rownames(x), x, check.names = FALSE), file.path(d, "counts.csv"), row.names = FALSE, quote = FALSE)
  write.csv(data.frame(sample = colnames(x), condition = sc[[2]]), file.path(d, "metadata.csv"), row.names = FALSE, quote = FALSE)
  dds <- suppressMessages(DESeq(DESeqDataSetFromMatrix(x, md, ~condition)))
  res <- results(dds, contrast = c("condition", "trt", "ctrl"))
  write.table(data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                         lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion),
              file.path(d, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  cat(sc[[1]], "dispPriorVar", attr(dispersionFunction(dds), "dispPriorVar"), "\n")
}
