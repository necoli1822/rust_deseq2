# References for R's fast vst() (trend fitted on nsub = 1000 genes)
suppressMessages(library(DESeq2))
d <- file.path(commandArgs(TRUE)[1], "transform")
x <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1))
md <- read.csv(file.path(d, "metadata.csv"), row.names = 1); md$condition <- factor(md$condition)
dd <- DESeqDataSetFromMatrix(x, md, ~condition)
for (ft in c("parametric", "local", "mean")) for (bl in c(TRUE, FALSE)) {
  v <- suppressMessages(vst(dd, blind = bl, nsub = 1000, fitType = ft))
  write.table(assay(v), file.path(d, sprintf("r_fastvst%s_%s.tsv", ft, bl)), sep = "\t", quote = FALSE)
}
cat("done\n")
