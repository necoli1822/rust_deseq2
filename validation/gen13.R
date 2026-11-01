# R references for functions in design formulas (run --formula): factor(), relevel(),
# ordered(), log(), scale(), I(), poly(), ns(), bs(), offset(), logical columns, levels
# that need make.names(), and LRTs with polynomial/spline terms, including a reduced
# model matrix that is not full rank
#
# usage: Rscript validation/gen13.R <data dir>
# One data set of 24 samples (cond A/B/C x 8, batch b1/b2, x centred and scaled, conc
# positive with log(conc) ~ N(0, 0.6), age 20-60, dose 1/2/3 numeric, geno with levels
# "wild type", "KO-2", "1x", flag TRUE/FALSE), written to every scenario directory:
#   fdose_default       ~ batch + factor(dose)            results(dds): factor.dose.3
#   fdose_name          ~ batch + factor(dose)            name = "factor.dose.2"
#   relevel_default     ~ batch + relevel(cond, ref = "B") results(dds): relevel.cond..ref....B..C
#   relevel_fdose_lrt   ~ batch + relevel(factor(dose), "2"), LRT reduced ~ batch
#   logconc_name        ~ cond + log(conc)                name = "log.conc."
#   scaleage_name       ~ cond + scale(age)               name = "scale.age."
#   isq_name            ~ cond + x + I(x^2)               name = "I.x.2."
#   poly_name           ~ cond + poly(x, 2)               name = "poly.x..2.2"
#   poly_lrt            ~ cond + poly(x, 2), LRT reduced ~ cond
#   polyraw_lrt         ~ cond + poly(x, 2, raw = TRUE), LRT reduced ~ cond
#   ns_name             ~ cond + ns(x, df = 3)            name = "ns.x..df...3.2"; also r_vst.tsv
#                       (varianceStabilizingTransformation(blind = FALSE), dispersions of this design)
#   ns_lrt              ~ cond + ns(x, df = 3), LRT reduced ~ cond
#   bs_name             ~ cond + bs(x, df = 4)            name = "bs.x..df...4.3"
#   bs_lrt              ~ cond + bs(x, df = 4), LRT reduced ~ cond
#   condpoly_name       ~ cond * poly(x, 2)               name = "condB.poly.x..2.1"
#   condpoly_lrt        ~ cond * poly(x, 2), LRT reduced ~ cond + poly(x, 2)
#   condns_name         ~ cond + cond:ns(x, df = 2)       name = "condC.ns.x..df...2.2"
#   ordered_default     ~ batch + ordered(cond)           results(dds): ordered.cond..Q (contr.poly)
#   ordered_L           ~ batch + ordered(cond)           name = "ordered.cond..L"
#   rankdef_lrt         ~ cond + poly(x, 2), LRT reduced ~ x + poly(x, 2): 4 columns of rank 3
#                       (R fits it from log(baseMean) for the intercept and 0 elsewhere; df = 1)
#   rankdef0_lrt        ~ 0 + cond + poly(x, 3), LRT reduced ~ 0 + cond + x + scale(x): 5 columns
#                       of rank 4, no intercept (R starts every coefficient at 1)
#   offset_default      ~ cond + offset(conc)             results(dds); R ignores the offset
#   mknames_cvar        ~ batch + geno                    contrast = c("geno", "KO-2", "wild type")
#   mknames_default     ~ batch + geno                    results(dds): geno_wild.type_vs_1x
#   logical_name        ~ flag + cond                     name = "flagTRUE" (logical: not renamed)
#   logical_default     ~ flag + cond                     results(dds): cond_C_vs_A
#   numeric_only        ~ x + scale(age)                  results(dds): scale.age.
#   noint_fdose         ~ 0 + factor(dose)                results(dds): factor.dose.3 (dose is
#                       numeric, so R's no-intercept default contrast does not apply)
#   shr_normal_logconc  ~ log(conc) + cond                lfcShrink(coef = "cond_C_vs_A", type = "normal")
#   shr_apeglm_poly     ~ cond + poly(x, 2)               lfcShrink(coef = "poly.x..2.1", type = "apeglm")
suppressMessages({library(DESeq2); library(apeglm); library(splines)})
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

set.seed(130)
m <- 24
md <- data.frame(cond = rep(c("A", "B", "C"), each = 8), batch = rep(c("b1", "b2"), 12),
                 x = round(as.vector(scale(rnorm(m))), 3), conc = round(exp(rnorm(m, 0, 0.6)), 3),
                 age = round(runif(m, 20, 60), 1), dose = rep(c(1, 2, 3), 8),
                 geno = rep(c("wild type", "KO-2", "1x"), each = 2, times = 4), flag = rep(c(TRUE, FALSE), each = 12))
cd <- md
for (v in c("cond", "batch", "geno")) cd[[v]] <- factor(cd[[v]])
x <- simulate(model.matrix(~ batch + cond + factor(dose) + poly(x, 2) + log(conc) + ns(age, df = 2) + flag + geno, cd),
              2000, 131)
dds_of <- function(design, ...) q(DESeq(DESeqDataSetFromMatrix(x, cd, design), ...))
check_names <- function(dds, expected) {
  if (!identical(resultsNames(dds), expected)) stop("resultsNames: ", paste(resultsNames(dds), collapse = ", "))
}

dds <- dds_of(~ batch + factor(dose))
check_names(dds, c("Intercept", "batch_b2_vs_b1", "factor.dose.2", "factor.dose.3"))
res_out("fdose_default", results(dds), dds, x, md)
res_out("fdose_name", results(dds, name = "factor.dose.2"), dds, x, md)

dds <- dds_of(~ batch + relevel(cond, ref = "B"))
check_names(dds, c("Intercept", "batch_b2_vs_b1", "relevel.cond..ref....B..A", "relevel.cond..ref....B..C"))
res_out("relevel_default", results(dds), dds, x, md)
ddl <- dds_of(~ batch + relevel(factor(dose), "2"), test = "LRT", reduced = ~ batch)
res_out("relevel_fdose_lrt", results(ddl), ddl, x, md)

dds <- dds_of(~ cond + log(conc))
check_names(dds, c("Intercept", "cond_B_vs_A", "cond_C_vs_A", "log.conc."))
res_out("logconc_name", results(dds, name = "log.conc."), dds, x, md)
dds <- dds_of(~ cond + scale(age))
res_out("scaleage_name", results(dds, name = "scale.age."), dds, x, md)
dds <- dds_of(~ cond + x + I(x^2))
res_out("isq_name", results(dds, name = "I.x.2."), dds, x, md)

dds <- dds_of(~ cond + poly(x, 2))
check_names(dds, c("Intercept", "cond_B_vs_A", "cond_C_vs_A", "poly.x..2.1", "poly.x..2.2"))
res_out("poly_name", results(dds, name = "poly.x..2.2"), dds, x, md)
res_out("shr_apeglm_poly", q(lfcShrink(dds, coef = "poly.x..2.1", type = "apeglm")), dds, x, md)
ddl <- dds_of(~ cond + poly(x, 2), test = "LRT", reduced = ~ cond)
res_out("poly_lrt", results(ddl), ddl, x, md)
ddl <- dds_of(~ cond + poly(x, 2, raw = TRUE), test = "LRT", reduced = ~ cond)
res_out("polyraw_lrt", results(ddl), ddl, x, md)

dds <- dds_of(~ cond + ns(x, df = 3))
check_names(dds, c("Intercept", "cond_B_vs_A", "cond_C_vs_A", "ns.x..df...3.1", "ns.x..df...3.2", "ns.x..df...3.3"))
res_out("ns_name", results(dds, name = "ns.x..df...3.2"), dds, x, md)
mat_out("ns_name", "vst", assay(varianceStabilizingTransformation(dds, blind = FALSE)))
ddl <- dds_of(~ cond + ns(x, df = 3), test = "LRT", reduced = ~ cond)
res_out("ns_lrt", results(ddl), ddl, x, md)

dds <- dds_of(~ cond + bs(x, df = 4))
res_out("bs_name", results(dds, name = "bs.x..df...4.3"), dds, x, md)
ddl <- dds_of(~ cond + bs(x, df = 4), test = "LRT", reduced = ~ cond)
res_out("bs_lrt", results(ddl), ddl, x, md)

dds <- dds_of(~ cond * poly(x, 2))
check_names(dds, c("Intercept", "cond_B_vs_A", "cond_C_vs_A", "poly.x..2.1", "poly.x..2.2", "condB.poly.x..2.1",
                   "condC.poly.x..2.1", "condB.poly.x..2.2", "condC.poly.x..2.2"))
res_out("condpoly_name", results(dds, name = "condB.poly.x..2.1"), dds, x, md)
ddl <- dds_of(~ cond * poly(x, 2), test = "LRT", reduced = ~ cond + poly(x, 2))
res_out("condpoly_lrt", results(ddl), ddl, x, md)
dds <- dds_of(~ cond + cond:ns(x, df = 2))
res_out("condns_name", results(dds, name = "condC.ns.x..df...2.2"), dds, x, md)

dds <- dds_of(~ batch + ordered(cond))
check_names(dds, c("Intercept", "batch_b2_vs_b1", "ordered.cond..L", "ordered.cond..Q"))
res_out("ordered_default", results(dds), dds, x, md)
res_out("ordered_L", results(dds, name = "ordered.cond..L"), dds, x, md)

ddl <- dds_of(~ cond + poly(x, 2), test = "LRT", reduced = ~ x + poly(x, 2))
stopifnot(qr(attr(ddl, "reducedModelMatrix"))$rank == 3, ncol(attr(ddl, "reducedModelMatrix")) == 4)
cat("rankdef_lrt: reduced fits not converged:", sum(!mcols(ddl)$reducedBetaConv, na.rm = TRUE), "\n")
res_out("rankdef_lrt", results(ddl), ddl, x, md)
ddl <- dds_of(~ 0 + cond + poly(x, 3), test = "LRT", reduced = ~ 0 + cond + x + scale(x))
stopifnot(qr(attr(ddl, "reducedModelMatrix"))$rank == 4, ncol(attr(ddl, "reducedModelMatrix")) == 5)
cat("rankdef0_lrt: reduced fits not converged:", sum(!mcols(ddl)$reducedBetaConv, na.rm = TRUE), "\n")
res_out("rankdef0_lrt", results(ddl), ddl, x, md)

dds <- dds_of(~ cond + offset(conc))
check_names(dds, c("Intercept", "cond_B_vs_A", "cond_C_vs_A"))
res_out("offset_default", results(dds), dds, x, md)

dds <- dds_of(~ batch + geno)
check_names(dds, c("Intercept", "batch_b2_vs_b1", "geno_KO.2_vs_1x", "geno_wild.type_vs_1x"))
res_out("mknames_cvar", results(dds, contrast = c("geno", "KO-2", "wild type")), dds, x, md)
res_out("mknames_default", results(dds), dds, x, md)

dds <- dds_of(~ flag + cond)
check_names(dds, c("Intercept", "flagTRUE", "cond_B_vs_A", "cond_C_vs_A"))
res_out("logical_name", results(dds, name = "flagTRUE"), dds, x, md)
res_out("logical_default", results(dds), dds, x, md)

dds <- dds_of(~ x + scale(age))
check_names(dds, c("Intercept", "x", "scale.age."))
res_out("numeric_only", results(dds), dds, x, md)
dds <- dds_of(~ 0 + factor(dose))
check_names(dds, c("factor.dose.1", "factor.dose.2", "factor.dose.3"))
res_out("noint_fdose", results(dds), dds, x, md)

dds <- dds_of(~ log(conc) + cond)
res_out("shr_normal_logconc", q(lfcShrink(dds, coef = "cond_C_vs_A", type = "normal")), dds, x, md)
cat("written", out, "\n")
