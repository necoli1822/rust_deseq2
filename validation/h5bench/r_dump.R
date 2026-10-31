# R reference for h5bench: what rhdf5 reads from a kallisto abundance.h5
# usage: Rscript r_dump.R <abundance.h5> <out.tsv> <repeats>
suppressMessages(library(rhdf5))
a <- commandArgs(TRUE); f <- a[1]
ids <- as.character(h5read(f, "aux/ids"))
counts <- as.numeric(h5read(f, "est_counts"))
eff <- as.numeric(h5read(f, "aux/eff_lengths"))
nb <- as.integer(h5read(f, "aux/num_bootstrap"))
bs <- lapply(seq_len(nb) - 1, function(k) as.numeric(h5read(f, sprintf("bootstrap/bs%d", k))))
df <- data.frame(target_id = ids, est_counts = sprintf("%.17g", counts), eff_length = sprintf("%.17g", eff))
for (k in seq_len(nb)) df[[sprintf("bs%d", k - 1)]] <- sprintf("%.17g", bs[[k]])
write.table(df, a[2], sep = "\t", quote = FALSE, row.names = FALSE)
# timing: the same reads, repeated
t <- sapply(seq_len(as.integer(a[3])), function(i) system.time({
  x <- list(h5read(f, "aux/ids"), h5read(f, "est_counts"), h5read(f, "aux/eff_lengths"))
  for (k in seq_len(nb) - 1) x[[length(x) + 1]] <- h5read(f, sprintf("bootstrap/bs%d", k))
})[["elapsed"]] * 1000)
cat(sprintf("rhdf5\tok\ttranscripts=%d\tbootstraps=%d\tmedian_ms=%.2f\tmin_ms=%.2f\n", length(ids), nb, median(t), min(t)))
