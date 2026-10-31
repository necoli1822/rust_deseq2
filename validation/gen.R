suppressMessages({library(DESeq2); library(apeglm); library(ashr)})
args <- commandArgs(TRUE); out <- args[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

sim <- function(m, cond, seed, n = 3000, extra = NULL, outlier = FALSE, sparse = FALSE) {
  set.seed(seed)
  mu0 <- exp(rnorm(n, 5, 2)); disp <- 0.05 + 4 / mu0
  beta <- ifelse(runif(n) < 0.15, rnorm(n, 0, 1.5), 0)
  lv <- sort(unique(cond)); eff <- matrix(0, n, length(lv))
  for (k in seq_along(lv)[-1]) eff[, k] <- beta * (k - 1) + ifelse(runif(n) < .05, rnorm(n), 0)
  sf <- runif(m, 0.6, 1.6)
  x <- sapply(seq_len(m), function(j) {
    mu <- mu0 * 2^eff[, match(cond[j], lv)] * sf[j]
    if (!is.null(extra)) mu <- mu * 2^(extra$coef * extra$x[j])
    rnbinom(n, mu = mu, size = 1 / disp)
  })
  if (outlier) { idx <- sample(n, 60); x[cbind(idx, sample(m, 60, TRUE))] <- x[cbind(idx, sample(m, 60, TRUE))] * 50 + 100 }
  if (sparse) x[matrix(runif(n * m) < 0.3, n)] <- 0
  dimnames(x) <- list(sprintf("g%04d", 1:n), sprintf("s%02d", 1:m)); x
}
wr <- function(d, x, md) {
  dir.create(d, showWarnings = FALSE)
  write.csv(data.frame(gene_id = rownames(x), x, check.names = FALSE), file.path(d, "counts.csv"), row.names = FALSE, quote = FALSE)
  write.csv(data.frame(sample = colnames(x), md), file.path(d, "metadata.csv"), row.names = FALSE, quote = FALSE)
}
res_out <- function(d, res, dds, name = "r_results.tsv") {
  df <- data.frame(gene_id = rownames(res), baseMean = res$baseMean, log2FoldChange = res$log2FoldChange,
                   lfcSE = res$lfcSE, pvalue = res$pvalue, padj = res$padj)
  if (!is.null(mcols(dds)$dispersion)) df$dispersion <- mcols(dds)$dispersion
  write.table(df, file.path(d, name), sep = "\t", row.names = FALSE, quote = FALSE)
}
mk <- function(x, md, design) {
  for (v in names(md)) if (is.character(md[[v]])) md[[v]] <- factor(md[[v]], levels = sort(unique(md[[v]])))
  DESeqDataSetFromMatrix(x, md, design)
}
q <- function(e) suppressMessages(suppressWarnings(e))

two <- rep(c("ctrl", "trt"), each = 3)
## S01-S03 basic, fitTypes
x <- sim(6, two, 1); md <- data.frame(condition = two)
for (ft in c("parametric", "local", "mean")) {
  d <- file.path(out, paste0("basic_", ft)); wr(d, x, md)
  dds <- q(DESeq(mk(x, md, ~condition), fitType = ft)); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
}
## use-t, lfcThreshold
d <- file.path(out, "use_t"); wr(d, x, md)
dds <- q(DESeq(mk(x, md, ~condition), useT = TRUE)); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
d <- file.path(out, "lfcthr"); wr(d, x, md)
dds <- q(DESeq(mk(x, md, ~condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl"), lfcThreshold = 1), dds)
## shrinkage
for (ty in c("normal", "apeglm", "ashr")) {
  d <- file.path(out, paste0("shrink_", ty)); wr(d, x, md)
  r <- q(lfcShrink(dds, coef = "condition_trt_vs_ctrl", type = ty)); res_out(d, r, dds)
}
## 3 levels, non-reference contrast
three <- rep(c("A", "B", "C"), each = 3); x3 <- sim(9, three, 2); md3 <- data.frame(condition = three)
d <- file.path(out, "three_CvsB"); wr(d, x3, md3)
dds <- q(DESeq(mk(x3, md3, ~condition))); res_out(d, results(dds, contrast = c("condition", "C", "B")), dds)
d <- file.path(out, "lrt_three"); wr(d, x3, md3)
dds <- q(DESeq(mk(x3, md3, ~condition), test = "LRT", reduced = ~1)); res_out(d, results(dds), dds)
## batch
cb <- rep(c("ctrl", "trt"), 4); bb <- rep(c("b1", "b2"), each = 4)
xb <- sim(8, cb, 3, extra = list(x = (bb == "b2") * 1, coef = 0.8)); mdb <- data.frame(batch = bb, condition = cb)
d <- file.path(out, "batch"); wr(d, xb, mdb)
dds <- q(DESeq(mk(xb, mdb, ~batch + condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
d <- file.path(out, "shrink_apeglm_batch"); wr(d, xb, mdb)
r <- q(lfcShrink(dds, coef = "condition_trt_vs_ctrl", type = "apeglm")); res_out(d, r, dds)
d <- file.path(out, "lrt_batch"); wr(d, xb, mdb)
dds <- q(DESeq(mk(xb, mdb, ~batch + condition), test = "LRT", reduced = ~batch)); res_out(d, results(dds), dds)
## continuous covariate
set.seed(9); age <- round(runif(8, 20, 60))
xc <- sim(8, cb, 4, extra = list(x = (age - 40) / 10, coef = 0.3)); mdc <- data.frame(age = age, condition = cb)
d <- file.path(out, "continuous"); wr(d, xc, mdc)
dds <- q(DESeq(mk(xc, mdc, ~age + condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
d <- file.path(out, "lrt_continuous"); wr(d, xc, mdc)
dds <- q(DESeq(mk(xc, mdc, ~age + condition), test = "LRT", reduced = ~age)); res_out(d, results(dds), dds)
## outliers: 7 vs 7 (replacement) and 4 vs 4 (filtering only)
c7 <- rep(c("ctrl", "trt"), each = 7); xo <- sim(14, c7, 5, outlier = TRUE); mdo <- data.frame(condition = c7)
d <- file.path(out, "outlier_replace"); wr(d, xo, mdo)
dds <- q(DESeq(mk(xo, mdo, ~condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
c4 <- rep(c("ctrl", "trt"), each = 4); xo4 <- sim(8, c4, 6, outlier = TRUE); mdo4 <- data.frame(condition = c4)
d <- file.path(out, "outlier_filter"); wr(d, xo4, mdo4)
dds <- q(DESeq(mk(xo4, mdo4, ~condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
## sparse + poscounts
xs <- sim(6, two, 7, sparse = TRUE)
d <- file.path(out, "poscounts"); wr(d, xs, md)
dds <- q(DESeq(mk(xs, md, ~condition), sfType = "poscounts")); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
write.table(data.frame(sf = sizeFactors(dds)), file.path(d, "r_sf.tsv"), sep = "\t", quote = FALSE)
## transforms (on basic data, 3000 genes -> vst uses all genes)
d <- file.path(out, "transform"); wr(d, x, md)
dd <- mk(x, md, ~condition)
for (bl in c(TRUE, FALSE)) {
  write.table(assay(q(varianceStabilizingTransformation(dd, blind = bl))), file.path(d, sprintf("r_vst_%s.tsv", bl)), sep = "\t", quote = FALSE)
  write.table(assay(q(rlog(dd, blind = bl))), file.path(d, sprintf("r_rlog_%s.tsv", bl)), sep = "\t", quote = FALSE)
}
write.table(counts(q(estimateSizeFactors(dd)), normalized = TRUE), file.path(d, "r_norm.tsv"), sep = "\t", quote = FALSE)
## genotype WT/KO/DKO, reference WT (contrast-name suffix collision)
geno <- rep(c("WT","KO","DKO"), each = 3); xg <- sim(9, geno, 11); mdg <- data.frame(genotype = geno)
d <- file.path(out, "geno_KOvsWT"); wr(d, xg, mdg)
mdg2 <- mdg; mdg2$genotype <- relevel(factor(mdg2$genotype, levels = sort(unique(geno))), "WT")
dds <- q(DESeq(DESeqDataSetFromMatrix(xg, mdg2, ~genotype))); res_out(d, results(dds, contrast = c("genotype", "KO", "WT")), dds)
## 2 vs 2 (df = 2, simulated prior variance path)
c2 <- rep(c("ctrl","trt"), each = 2); x2 <- sim(4, c2, 12); md2 <- data.frame(condition = c2)
d <- file.path(out, "small_2v2"); wr(d, x2, md2)
dds <- q(DESeq(mk(x2, md2, ~condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
## 7 vs 7 outliers with local fit
d <- file.path(out, "outlier_replace_local"); wr(d, xo, mdo)
dds <- q(DESeq(mk(xo, mdo, ~condition), fitType = "local")); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
cat("done\n")
