# References for observation weights and sfType = "iterate"
suppressMessages(library(DESeq2))
args <- commandArgs(TRUE); out <- args[1]
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
src <- readLines(file.path(here, "gen.R"))
eval(parse(text = src[seq_len(grep("^two <- ", src)[1] - 1)]))

wout <- function(d, w) write.csv(data.frame(gene_id = rownames(w), w, check.names = FALSE),
                                  file.path(d, "weights.csv"), row.names = FALSE, quote = FALSE)
mkw <- function(x, seed) {
  set.seed(seed)
  w <- matrix(runif(length(x), 0.2, 1), nrow(x)); w[runif(length(x)) < 0.05] <- runif(1, 0.005, 0.02)
  dimnames(w) <- dimnames(x); w
}

## weights, 4 vs 4
c4 <- rep(c("ctrl", "trt"), each = 4); x <- sim(8, c4, 41); md <- data.frame(condition = c4); w <- mkw(x, 42)
for (sc in c("weights", "weights_use_t")) {
  d <- file.path(out, sc); wr(d, x, md); wout(d, w)
  dd <- mk(x, md, ~condition); assays(dd)[["weights"]] <- w
  dds <- q(DESeq(dd, useT = sc == "weights_use_t")); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
}
## weights with a batch covariate
cb <- rep(c("ctrl", "trt"), 4); bb <- rep(c("b1", "b2"), each = 4)
xb <- sim(8, cb, 43, extra = list(x = (bb == "b2") * 1, coef = 0.8)); mdb <- data.frame(batch = bb, condition = cb); wb <- mkw(xb, 44)
d <- file.path(out, "weights_batch"); wr(d, xb, mdb); wout(d, wb)
dd <- mk(xb, mdb, ~batch + condition); assays(dd)[["weights"]] <- wb
dds <- q(DESeq(dd)); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)

## sfType = "iterate" on sparse data
two <- rep(c("ctrl", "trt"), each = 3)
xs <- sim(6, two, 45, sparse = TRUE); md2 <- data.frame(condition = two)
d <- file.path(out, "iterate"); wr(d, xs, md2)
dds <- q(DESeq(mk(xs, md2, ~condition), sfType = "iterate")); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
write.table(data.frame(sf = sizeFactors(dds)), file.path(d, "r_sf.tsv"), sep = "\t", quote = FALSE)
cat("done\n")
