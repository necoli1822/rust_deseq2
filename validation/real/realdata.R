# Real-data references: GEO count tables -> counts.csv / metadata.csv / R DESeq2 results
# usage: realdata.R <geo download dir> <output dir> <validation dir with gen.R>
# Run fetch.sh first to download the GEO supplementary files into <geo download dir>.
suppressMessages({library(DESeq2); library(apeglm); library(ashr)})
a <- commandArgs(TRUE); G <- a[1]; out <- a[2]
src <- readLines(file.path(a[3], "gen.R"))
eval(parse(text = src[seq_len(grep("^two <- ", src)[1] - 1)]))   # wr, res_out, mk, q
out <- a[2]   # the helpers above reassign `out`
dir.create(out, showWarnings = FALSE, recursive = TRUE)
clean <- function(x) { x <- as.matrix(x); storage.mode(x) <- "integer"; x[rowSums(x) >= 0, , drop = FALSE] }
ids <- function(v) gsub("[,\t ]", "|", v)

## A. baumannii GSE151925
t <- read.delim(gzfile(file.path(G, "GSE151925_readcount.txt.gz")), check.names = FALSE)
aba <- clean(t[, -1]); rownames(aba) <- ids(t[, 1])
aba_md <- data.frame(condition = sub("_[0-9]+$", "", colnames(aba)))
## Salmonella GSE46391 (strains UK, G1, BC8 are high-pathogenic)
t <- read.delim(gzfile(file.path(G, "GSE46391_HP_and_LP_Total_gene_reads.txt.gz")), check.names = FALSE)
sal <- clean(t[, c("UK", "G1", "BC8", "C45", "C19", "G45")]); rownames(sal) <- ids(t[, 1])
sal_md <- data.frame(condition = c("HP", "HP", "HP", "LP", "LP", "LP"))
## S. aureus GSE130777 (Total counts columns)
t <- read.csv(gzfile(file.path(G, "GSE130777_mgrA_vs_LAC.csv.gz")), check.names = FALSE)
cc <- grep("Total counts", colnames(t))
sau <- clean(round(t[, cc])); rownames(sau) <- make.unique(ids(t$Name)); colnames(sau) <- c("LAC1", "LAC2", "LAC3", "mgrA1", "mgrA2", "mgrA3")
sau_md <- data.frame(condition = c("WT", "WT", "WT", "mgrA", "mgrA", "mgrA"))
## M. tuberculosis GSE100097 (12 per-sample files; Total gene reads)
f <- sort(list.files(file.path(G, "GSE100097"), pattern = "GE.csv.gz$", full.names = TRUE))
f <- f[order(as.integer(sub(".*TBC([0-9]+)_GE.*", "\\1", f)))]
l <- lapply(f, function(p) read.csv(gzfile(p)))
mtb <- clean(sapply(l, function(d) d[["Total.gene.reads"]])); rownames(mtb) <- make.unique(ids(l[[1]]$Name))
colnames(mtb) <- sprintf("TBC%02d", seq_along(f))
# GEO order: dextrose exp, exp, stat, stat, NRP1, NRP1, then the same for lipid
mtb_md <- data.frame(carbon = rep(c("dextrose", "lipid"), each = 6),
                     phase = rep(c("exp", "exp", "stat", "stat", "nrp1", "nrp1"), 2))
## P. aeruginosa GSE55197 (headerless per-sample files)
rd55 <- function(cond) {
  f <- sort(list.files(file.path(G, "GSE55197"), pattern = paste0("_", cond, "\\.[0-9]+\\.txt\\.gz$"), full.names = TRUE))
  setNames(lapply(f, function(p) read.delim(gzfile(p), header = FALSE)), paste0(cond, "_", seq_along(f)))
}
pae_tab <- function(conds) {
  l <- unlist(lapply(conds, rd55), recursive = FALSE)
  x <- clean(sapply(l, function(d) d[[2]])); rownames(x) <- make.unique(ids(l[[1]][[1]])); x
}
## E. coli GSE220559 (11 groups x 3)
f <- sort(list.files(file.path(G, "GSE220559"), pattern = "txt.gz$", full.names = TRUE))
l <- lapply(f, function(p) read.delim(gzfile(p)))
eco <- clean(sapply(l, function(d) d[[2]])); rownames(eco) <- ids(l[[1]][[1]])
colnames(eco) <- sub("^GSM[0-9]+_(.*)\\.txt\\.gz$", "\\1", basename(f))
eco_md <- data.frame(condition = sub("_[0-9]+$", "", colnames(eco)))

run <- function(name, x, md, design, contrast = NULL, shrink = NULL, ref = NULL, test = "Wald",
                reduced = NULL, fitType = "parametric", sfType = "ratio", weights = NULL) {
  d <- file.path(out, name); wr(d, x, md)
  dd <- mk(x, md, design)
  if (!is.null(ref)) for (v in names(ref)) dd[[v]] <- relevel(dd[[v]], ref[[v]])
  if (!is.null(weights)) {
    assays(dd)[["weights"]] <- weights
    write.csv(data.frame(gene_id = rownames(weights), weights, check.names = FALSE), file.path(d, "weights.csv"), row.names = FALSE, quote = FALSE)
  }
  dds <- if (test == "LRT") q(DESeq(dd, test = "LRT", reduced = reduced, fitType = fitType, sfType = sfType))
         else q(DESeq(dd, fitType = fitType, sfType = sfType))
  res <- if (test == "LRT") results(dds) else results(dds, contrast = contrast)
  if (!is.null(shrink)) {
    # The CLI's --numerator/--denominator is a contrast, so ashr is compared with
    # lfcShrink(contrast = ...); apeglm only accepts coef = in R.
    coef <- paste0(contrast[1], "_", contrast[2], "_vs_", contrast[3])
    res <- if (shrink == "ashr") q(lfcShrink(dds, contrast = contrast, type = shrink))
           else q(lfcShrink(dds, coef = coef, type = shrink))
  }
  res_out(d, res, dds)
  cat(name, ":", sum(res$padj < 0.1, na.rm = TRUE), "significant of", nrow(res), "\n")
}

run("aba_abaI_vs_WT", aba, aba_md, ~condition, c("condition", "abaI", "WT"), ref = list(condition = "WT"))
run("aba_abaI_vs_abaM", aba, aba_md, ~condition, c("condition", "abaI", "abaM"), ref = list(condition = "WT"))
run("aba_lrt", aba, aba_md, ~condition, test = "LRT", reduced = ~1)
run("aba_apeglm", aba, aba_md, ~condition, c("condition", "abaI", "WT"), ref = list(condition = "WT"), shrink = "apeglm")
run("aba_ashr", aba, aba_md, ~condition, c("condition", "abaI", "WT"), ref = list(condition = "WT"), shrink = "ashr")
set.seed(7); w <- matrix(runif(length(aba), 0.3, 1), nrow(aba), dimnames = dimnames(aba))
run("aba_weights", aba, aba_md, ~condition, c("condition", "abaI", "WT"), ref = list(condition = "WT"), weights = w)
run("sal_HP_vs_LP", sal, sal_md, ~condition, c("condition", "LP", "HP"))
run("sal_local", sal, sal_md, ~condition, c("condition", "LP", "HP"), fitType = "local")
run("sal_iterate", sal, sal_md, ~condition, c("condition", "LP", "HP"), sfType = "iterate")
run("sau_mgrA_vs_WT", sau, sau_md, ~condition, c("condition", "mgrA", "WT"), ref = list(condition = "WT"))
run("sau_poscounts", sau, sau_md, ~condition, c("condition", "mgrA", "WT"), ref = list(condition = "WT"), sfType = "poscounts")
run("sau_apeglm", sau, sau_md, ~condition, c("condition", "mgrA", "WT"), ref = list(condition = "WT"), shrink = "apeglm")
run("mtb_carbon_batch", mtb, mtb_md, ~phase + carbon, c("carbon", "lipid", "dextrose"))
run("mtb_lrt_phase", mtb, mtb_md, ~phase + carbon, test = "LRT", reduced = ~phase)
pae2 <- pae_tab(c("anox", "att_c")); run("pae_2v2", pae2, data.frame(condition = sub("_[0-9]+$", "", colnames(pae2))), ~condition, c("condition", "att_c", "anox"))
pae3 <- pae_tab(c("exp", "heat50")); run("pae_6v4", pae3, data.frame(condition = sub("_[0-9]+$", "", colnames(pae3))), ~condition, c("condition", "heat50", "exp"))
run("eco_CIP_vs_CAZ", eco, eco_md, ~condition, c("condition", "CIP", "CAZ"))
run("eco_lrt", eco, eco_md, ~condition, test = "LRT", reduced = ~1)

## transforms on Salmonella
d <- file.path(out, "sal_transform"); wr(d, sal, sal_md)
dd <- mk(sal, sal_md, ~condition)
wt <- function(m, n) write.table(m, file.path(d, n), sep = "\t", quote = FALSE)
wt(assay(q(varianceStabilizingTransformation(dd, blind = TRUE))), "r_T_vst.tsv")
wt(assay(q(vst(dd, blind = TRUE, nsub = 1000))), "r_T_vstnsub.tsv")
wt(assay(q(rlog(dd, blind = TRUE))), "r_T_rlog.tsv")
cat("done\n")
