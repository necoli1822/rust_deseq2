# Extra scenarios for model-matrix cell handling (sourced after gen.R's helpers)
suppressMessages({library(DESeq2)})
args <- commandArgs(TRUE); out <- args[1]
args_all <- commandArgs(FALSE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", args_all, value = TRUE))))
src <- readLines(file.path(here, "gen.R"))
# reuse helper definitions only (everything up to the first scenario)
eval(parse(text = src[seq_len(grep("^two <- ", src)[1] - 1)]))

## 16 samples, condition 8/8, batch 4/4 inside: no cell reaches 7 -> R replaces nothing
cb <- rep(rep(c("ctrl", "trt"), each = 4), 2); bb <- rep(c("b1", "b2"), each = 8)
xb <- sim(16, cb, 21, extra = list(x = (bb == "b2") * 1, coef = 0.5), outlier = TRUE)
mdb <- data.frame(batch = bb, condition = cb)
d <- file.path(out, "outlier_batch16"); wr(d, xb, mdb)
dds <- q(DESeq(mk(xb, mdb, ~batch + condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)

## unbalanced 2 vs 5 with outliers: maxCooks only over the 5-sample cell
cu <- c(rep("ctrl", 2), rep("trt", 5)); xu <- sim(7, cu, 22, outlier = TRUE); mdu <- data.frame(condition = cu)
d <- file.path(out, "unbal_2v5"); wr(d, xu, mdu)
dds <- q(DESeq(mk(xu, mdu, ~condition))); res_out(d, results(dds, contrast = c("condition", "trt", "ctrl")), dds)
cat("done\n")
