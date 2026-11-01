# R references for design formulas (run --formula): interactions, no-intercept designs,
# factor x continuous interactions, a three-way interaction and LRTs of interactions
#
# usage: Rscript validation/gen12.R <data dir>
# Data sets (one per block below), each written to its scenario directories:
#   ~ batch + condition + batch:condition (batch b1/b2 x condition A/B/C, 4 per cell)
#     inter_default        results(dds): the last coefficient, batchb2.conditionC
#     inter_name           name = "batchb2.conditionB"
#     inter_cvar           contrast = c("condition", "C", "B") (main-effect levels)
#     inter_lrt            LRT, reduced ~ batch + condition
#     inter_apeglm         lfcShrink(coef = "batchb2.conditionB", type = "apeglm")
#     inter_default also holds r_vst.tsv / r_rlog.tsv: varianceStabilizingTransformation()
#     and rlog() with blind = FALSE, whose dispersions use the interaction design
#   ~ condition + condition:batch (no batch main effect: batch nested in condition)
#     nested_name          name = "conditionB.batchb2"
#   ~ genotype * treatment (WT/KO x ctrl/trt, 7 per cell; outliers replaced and refit)
#     gt_default           results(dds): genotypeKO.treatmenttrt
#     gt_list              list(c("treatment_trt_vs_ctrl", "genotypeKO.treatmenttrt")):
#                          the treatment effect in KO
#     gt_cvar_genotype     contrast = c("genotype", "KO", "WT") (genotype effect in ctrl)
#     gt_lrt               LRT, reduced ~ genotype + treatment
#   ~ x + condition + x:condition (condition A/B, x centred and scaled)
#     fxc_name             name = "x.conditionB"
#     fxc_slope            name = "x"
#     fxc_list             list(c("x", "x.conditionB")): the slope in condition B
#   ~ 0 + condition (A/B/C)
#     noint_default        results(dds): R's default c("condition", "C", "A") without intercept
#     noint_BvsA           contrast = c("condition", "B", "A")
#     noint_name           name = "conditionB"
#   ~ 0 + batch + condition
#     noint2_default       results(dds): c("condition", "C", "A") -> coefficient conditionC
#     noint2_CvsB          contrast = c("condition", "C", "B")
#     noint2_AvsC          contrast = c("condition", "A", "C") (-conditionC)
#     noint2_batch         contrast = c("batch", "b2", "b1") (first variable: both columns)
#   ~ batch * condition * genotype (2 x 2 x 2, 3 per cell)
#     threeway_default     results(dds): batchb2.conditionB.genotypeWT (KO is the reference)
#     threeway_lrt         LRT, reduced ~ (batch + condition + genotype)^2
suppressMessages({library(DESeq2); library(apeglm)})
out <- commandArgs(TRUE)[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
q <- function(e) suppressMessages(suppressWarnings(e))

# counts with gene effects b[, k] on the columns of a model matrix mm (log2 scale)
simulate <- function(mm, n, seed, effect_frac = 0.15) {
  set.seed(seed)
  m <- nrow(mm)
  mu0 <- exp(rnorm(n, 5, 2)); disp <- 0.05 + 4 / mu0
  b <- sapply(seq_len(ncol(mm)), function(k) ifelse(runif(n) < effect_frac, rnorm(n, 0, 1), 0))
  b[, 1] <- 0
  sf <- runif(m, 0.6, 1.6)
  x <- sapply(seq_len(m), function(j) rnbinom(n, mu = mu0 * 2^(b %*% mm[j, ]) * sf[j], size = 1 / disp))
  dimnames(x) <- list(sprintf("g%04d", 1:n), sprintf("s%02d", 1:m))
  x
}

res_out <- function(sc, res, dds, x, md) {
  d <- file.path(out, sc); dir.create(d, showWarnings = FALSE)
  write.csv(data.frame(gene_id = rownames(x), x, check.names = FALSE), file.path(d, "counts.csv"),
            row.names = FALSE, quote = FALSE)
  write.csv(data.frame(sample = colnames(x), md), file.path(d, "metadata.csv"), row.names = FALSE, quote = FALSE)
  df <- data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                   lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj, dispersion = mcols(dds)$dispersion)
  write.table(df, file.path(d, "r_results.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}
mat_out <- function(sc, name, m) {
  write.table(data.frame(gene_id = rownames(m), m, check.names = FALSE), file.path(out, sc, paste0("r_", name, ".tsv")),
              sep = "\t", row.names = FALSE, quote = FALSE)
}
factors <- function(md, levels = list()) {
  for (v in names(md)) if (is.character(md[[v]])) md[[v]] <- factor(md[[v]], levels = levels[[v]] %||% sort(unique(md[[v]])))
  md
}
`%||%` <- function(a, b) if (is.null(a)) b else a

# --- ~ batch + condition + batch:condition ---------------------------------------------
md <- data.frame(batch = rep(c("b1", "b2"), 12), condition = rep(c("A", "B", "C"), each = 8))
cd <- factors(md)
x <- simulate(model.matrix(~ batch * condition, cd), 2000, 121)
dds <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ batch + condition + batch:condition)))
stopifnot(identical(resultsNames(dds), c("Intercept", "batch_b2_vs_b1", "condition_B_vs_A", "condition_C_vs_A",
                                         "batchb2.conditionB", "batchb2.conditionC")))
res_out("inter_default", results(dds), dds, x, md)
res_out("inter_name", results(dds, name = "batchb2.conditionB"), dds, x, md)
res_out("inter_cvar", results(dds, contrast = c("condition", "C", "B")), dds, x, md)
res_out("inter_apeglm", q(lfcShrink(dds, coef = "batchb2.conditionB", type = "apeglm")), dds, x, md)
mat_out("inter_default", "vst", assay(varianceStabilizingTransformation(dds, blind = FALSE)))
mat_out("inter_default", "rlog", assay(q(rlog(dds, blind = FALSE))))
ddl <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ batch + condition + batch:condition), test = "LRT",
               reduced = ~ batch + condition))
res_out("inter_lrt", results(ddl), ddl, x, md)

ddn <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ condition + condition:batch)))
stopifnot(identical(resultsNames(ddn), c("Intercept", "condition_B_vs_A", "condition_C_vs_A", "conditionA.batchb2",
                                         "conditionB.batchb2", "conditionC.batchb2")))
res_out("nested_name", results(ddn, name = "conditionB.batchb2"), ddn, x, md)

# --- ~ genotype * treatment, 7 per cell, with outliers ---------------------------------
md <- data.frame(genotype = rep(c("WT", "KO"), each = 14), treatment = rep(rep(c("ctrl", "trt"), each = 7), 2))
cd <- factors(md, list(genotype = c("WT", "KO")))
x <- simulate(model.matrix(~ genotype * treatment, cd), 2000, 122)
set.seed(1220)
og <- sample(nrow(x), 40); os <- sample(ncol(x), 40, replace = TRUE)
x[cbind(og, os)] <- x[cbind(og, os)] * 20L + 200L
dds <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ genotype * treatment)))
stopifnot(identical(resultsNames(dds), c("Intercept", "genotype_KO_vs_WT", "treatment_trt_vs_ctrl", "genotypeKO.treatmenttrt")))
cat("genotype * treatment: outlier counts replaced in", sum(mcols(dds)$replace, na.rm = TRUE), "genes\n")
stopifnot(sum(mcols(dds)$replace, na.rm = TRUE) > 0)
res_out("gt_default", results(dds), dds, x, md)
res_out("gt_list", results(dds, contrast = list(c("treatment_trt_vs_ctrl", "genotypeKO.treatmenttrt"))), dds, x, md)
res_out("gt_cvar_genotype", results(dds, contrast = c("genotype", "KO", "WT")), dds, x, md)
ddl <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ genotype * treatment), test = "LRT", reduced = ~ genotype + treatment))
cat("genotype * treatment LRT: outlier counts replaced in", sum(mcols(ddl)$replace, na.rm = TRUE), "genes\n")
res_out("gt_lrt", results(ddl), ddl, x, md)

# --- ~ x + condition + x:condition -----------------------------------------------------
set.seed(123)
md <- data.frame(x = round(as.vector(scale(rnorm(16))), 3), condition = rep(c("A", "B"), each = 8))
cd <- factors(md)
x <- simulate(model.matrix(~ x * condition, cd), 2000, 124)
dds <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ x + condition + x:condition)))
stopifnot(identical(resultsNames(dds), c("Intercept", "x", "condition_B_vs_A", "x.conditionB")))
res_out("fxc_name", results(dds, name = "x.conditionB"), dds, x, md)
res_out("fxc_slope", results(dds, name = "x"), dds, x, md)
res_out("fxc_list", results(dds, contrast = list(c("x", "x.conditionB"))), dds, x, md)

# --- ~ 0 + condition ---------------------------------------------------------------------
md <- data.frame(condition = rep(c("A", "B", "C"), each = 4))
cd <- factors(md)
x <- simulate(model.matrix(~ condition, cd), 2000, 125)
dds <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ 0 + condition)))
stopifnot(identical(resultsNames(dds), c("conditionA", "conditionB", "conditionC")))
res_out("noint_default", results(dds), dds, x, md)
res_out("noint_BvsA", results(dds, contrast = c("condition", "B", "A")), dds, x, md)
res_out("noint_name", results(dds, name = "conditionB"), dds, x, md)

# --- ~ 0 + batch + condition -----------------------------------------------------------
md <- data.frame(batch = rep(c("b1", "b2"), 6), condition = rep(c("A", "B", "C"), each = 4))
cd <- factors(md)
x <- simulate(model.matrix(~ batch + condition, cd), 2000, 126)
dds <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ 0 + batch + condition)))
stopifnot(identical(resultsNames(dds), c("batchb1", "batchb2", "conditionB", "conditionC")))
res_out("noint2_default", results(dds), dds, x, md)
res_out("noint2_CvsB", results(dds, contrast = c("condition", "C", "B")), dds, x, md)
res_out("noint2_AvsC", results(dds, contrast = c("condition", "A", "C")), dds, x, md)
res_out("noint2_batch", results(dds, contrast = c("batch", "b2", "b1")), dds, x, md)

# --- ~ batch * condition * genotype ------------------------------------------------------
md <- data.frame(batch = rep(c("b1", "b2"), 12), condition = rep(rep(c("A", "B"), each = 2), 6),
                 genotype = rep(c("KO", "WT"), each = 12))
cd <- factors(md)
x <- simulate(model.matrix(~ batch * condition * genotype, cd), 2000, 127)
dds <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ batch * condition * genotype)))
stopifnot(identical(resultsNames(dds)[8], "batchb2.conditionB.genotypeWT"))
res_out("threeway_default", results(dds), dds, x, md)
ddl <- q(DESeq(DESeqDataSetFromMatrix(x, cd, ~ batch * condition * genotype), test = "LRT",
               reduced = ~ (batch + condition + genotype)^2))
res_out("threeway_lrt", results(ddl), ddl, x, md)
cat("written", out, "\n")
