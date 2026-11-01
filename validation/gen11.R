# R references for contrast choices: results(name = ), numeric and list contrasts,
# contrasts on a covariate, R's default contrast, and shrinkage of a chosen coefficient
#
# usage: Rscript validation/gen11.R <data dir>
# One data set, design ~ batch + age + condition (condition A/B/C, 4 samples each,
# batch b1/b2, continuous age), written to every scenario directory:
#   name_age, name_batch          results(name = "age" / "batch_b2_vs_b1")
#   cvar_batch                    contrast = c("batch", "b1", "b2")
#   numeric_CvsB                  contrast = c(0, 0, 0, -1, 1)
#   list_CvsB                     contrast = list("condition_C_vs_A", "condition_B_vs_A")
#   list_avg                      list(c(B_vs_A, C_vs_A), character()), listValues = c(0.5, -0.5)
#   default                       results(dds): the last coefficient, condition_C_vs_A (R applies
#                                 c(lastVar, last, first) only without an intercept or with expanded matrices)
#   numeric_useT                  DESeq(useT = TRUE), numeric C vs B
#   name_lfcthr                   name = "batch_b2_vs_b1", lfcThreshold = 0.5
#   lrt_list                      LRT ~batch + age, LFC from list C vs B
#   shrink_normal_batch           lfcShrink(coef = "batch_b2_vs_b1", type = "normal")
#   shrink_apeglm_age             lfcShrink(coef = "age", type = "apeglm")
suppressMessages({library(DESeq2); library(apeglm)})
out <- commandArgs(TRUE)[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
q <- function(e) suppressMessages(suppressWarnings(e))

set.seed(111)
n <- 3000; m <- 12
cond <- rep(c("A", "B", "C"), each = 4); batch <- rep(c("b1", "b2"), 6)
age <- round(rnorm(m, 0, 1), 3)    # centered and scaled, as R recommends for numeric design variables
mu0 <- exp(rnorm(n, 5, 2)); disp <- 0.05 + 4 / mu0
bB <- ifelse(runif(n) < 0.15, rnorm(n, 0, 1.2), 0); bC <- bB + ifelse(runif(n) < 0.15, rnorm(n, 0, 1), 0)
bBatch <- ifelse(runif(n) < 0.2, rnorm(n, 0, 0.8), 0); bAge <- ifelse(runif(n) < 0.1, rnorm(n, 0, 0.5), 0)
sf <- runif(m, 0.6, 1.6)
x <- sapply(seq_len(m), function(j) {
  eff <- bB * (cond[j] == "B") + bC * (cond[j] == "C") + bBatch * (batch[j] == "b2") + bAge * age[j]
  rnbinom(n, mu = mu0 * 2^eff * sf[j], size = 1 / disp)
})
dimnames(x) <- list(sprintf("g%04d", 1:n), sprintf("s%02d", 1:m))
md <- data.frame(batch = batch, age = age, condition = cond, row.names = colnames(x))

res_out <- function(sc, res, dds) {
  d <- file.path(out, sc); dir.create(d, showWarnings = FALSE)
  write.csv(data.frame(gene_id = rownames(x), x, check.names = FALSE), file.path(d, "counts.csv"),
            row.names = FALSE, quote = FALSE)
  write.csv(data.frame(sample = rownames(md), md), file.path(d, "metadata.csv"), row.names = FALSE, quote = FALSE)
  df <- data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                   lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion)
  write.table(df, file.path(d, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

mk <- function() {
  cd <- md; cd$batch <- factor(cd$batch); cd$condition <- factor(cd$condition)
  DESeqDataSetFromMatrix(x, cd, ~ batch + age + condition)
}
dds <- q(DESeq(mk()))
stopifnot(identical(resultsNames(dds), c("Intercept", "batch_b2_vs_b1", "age", "condition_B_vs_A", "condition_C_vs_A")))
res_out("name_age", results(dds, name = "age"), dds)
res_out("name_batch", results(dds, name = "batch_b2_vs_b1"), dds)
res_out("cvar_batch", results(dds, contrast = c("batch", "b1", "b2")), dds)
res_out("numeric_CvsB", results(dds, contrast = c(0, 0, 0, -1, 1)), dds)
res_out("list_CvsB", results(dds, contrast = list("condition_C_vs_A", "condition_B_vs_A")), dds)
res_out("list_avg", results(dds, contrast = list(c("condition_B_vs_A", "condition_C_vs_A"), character()),
                            listValues = c(0.5, -0.5)), dds)
res_out("default", results(dds), dds)
res_out("name_lfcthr", results(dds, name = "batch_b2_vs_b1", lfcThreshold = 0.5), dds)
res_out("shrink_normal_batch", q(lfcShrink(dds, coef = "batch_b2_vs_b1", type = "normal")), dds)
res_out("shrink_apeglm_age", q(lfcShrink(dds, coef = "age", type = "apeglm")), dds)

ddt <- q(DESeq(mk(), useT = TRUE))
res_out("numeric_useT", results(ddt, contrast = c(0, 0, 0, -1, 1)), ddt)

ddl <- q(DESeq(mk(), test = "LRT", reduced = ~ batch + age))
res_out("lrt_list", results(ddl, contrast = list("condition_C_vs_A", "condition_B_vs_A")), ddl)
cat("written", out, "\n")
