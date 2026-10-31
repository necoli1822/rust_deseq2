# VST references for fitType local / mean
suppressMessages(library(DESeq2))
d <- file.path(commandArgs(TRUE)[1], "transform")
x <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1))
md <- read.csv(file.path(d, "metadata.csv"), row.names = 1); md$condition <- factor(md$condition)
dd <- DESeqDataSetFromMatrix(x, md, ~condition)
for (ft in c("local", "mean")) for (bl in c(TRUE, FALSE)) {
  v <- suppressMessages(varianceStabilizingTransformation(dd, blind = bl, fitType = ft))
  write.table(assay(v), file.path(d, sprintf("r_vst%s_%s.tsv", ft, bl)), sep = "\t", quote = FALSE)
}
cat("done\n")
