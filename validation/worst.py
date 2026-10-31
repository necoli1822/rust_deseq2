"""Show the genes with the largest |R - Rust| in a column. usage: worst.py <scenario_dir> <tag> [column] [n]"""
import csv, math, sys

d, tag = sys.argv[1], sys.argv[2]
col = sys.argv[3] if len(sys.argv) > 3 else "log2FoldChange"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8


def load(p):
    rows = list(csv.reader(open(p), delimiter="\t"))
    return {r[0]: dict(zip(rows[0][1:], r[1:])) for r in rows[1:]}


def f(v):
    try:
        x = float(v)
        return x
    except ValueError:
        return math.nan


R, U = load(f"{d}/r_results.tsv"), load(f"{d}/rust_{tag}.tsv")
counts = {r[0]: r[1:] for r in list(csv.reader(open(f"{d}/counts.csv")))[1:]}
diffs = []
for g in R:
    a, b = f(R[g][col]), f(U[g][col])
    key = abs(a - b) if not (math.isnan(a) or math.isnan(b)) else (0 if math.isnan(a) and math.isnan(b) else math.inf)
    diffs.append((key, g, a, b))
for k, g, a, b in sorted(diffs, reverse=True)[:n]:
    print(f"{g} R={a:.6g} rust={b:.6g} | R p={R[g]['pvalue']} rust p={U[g]['pvalue']} | counts={' '.join(counts[g])}")
