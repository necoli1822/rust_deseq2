"""Compare Rust outputs against R DESeq2 references. usage: compare.py <data_dir> <tag>"""
import csv, math, os, sys

D, T = sys.argv[1], sys.argv[2]


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except ValueError:
        return None


def table(path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    h = rows[0]
    return {r[0]: dict(zip(h[1:], r[1:])) for r in rows[1:]}


def matrix(path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return {r[0]: [float(x) for x in r[1:]] for r in rows[1:]}


print(f"{'scenario':24s} {'dLFC':>9s} {'dSE':>9s} {'dlog10p':>9s} {'dpadj':>9s} {'NAmis':>6s} {'sigDis':>6s} {'nSigR':>6s}")
for s in sorted(os.listdir(D)):
    rp, up = os.path.join(D, s, "r_results.tsv"), os.path.join(D, s, f"rust_{T}.tsv")
    if not os.path.exists(rp):
        continue
    if not os.path.exists(up):
        print(f"{s:24s} {'(no rust output)':>9s}")
        continue
    R, U = table(rp), table(up)
    dl = dse = dp = dq = 0.0
    namis = sigdis = nsig = 0
    for g, r in R.items():
        u = U.get(g, {})
        for key, acc in (("log2FoldChange", "dl"), ("lfcSE", "dse"), ("padj", "dq")):
            a, b = num(r.get(key, "NA")), num(u.get(key, "NA"))
            if (a is None) != (b is None):
                if key == "padj":
                    namis += 1
                continue
            if a is None:
                continue
            d = abs(a - b)
            if acc == "dl": dl = max(dl, d)
            elif acc == "dse": dse = max(dse, d)
            else: dq = max(dq, d)
        a, b = num(r["pvalue"]), num(u.get("pvalue", "NA"))
        if a is not None and b is not None and a > 0 and b > 0:
            dp = max(dp, abs(math.log10(a) - math.log10(b)))
        qa, qb = num(r["padj"]), num(u.get("padj", "NA"))
        sa, sb = qa is not None and qa < 0.1, qb is not None and qb < 0.1
        nsig += sa
        sigdis += sa != sb
    print(f"{s:24s} {dl:9.2e} {dse:9.2e} {dp:9.2e} {dq:9.2e} {namis:6d} {sigdis:6d} {nsig:6d}")

TD = os.path.join(D, "transform")
for name in ["norm", "vst_TRUE", "vst_FALSE", "vstlocal_TRUE", "vstlocal_FALSE", "vstmean_TRUE", "vstmean_FALSE", "rlog_TRUE", "rlog_FALSE"]:
    rp, up = os.path.join(TD, f"r_{name}.tsv"), os.path.join(TD, f"rust_{name}_{T}.tsv")
    if not os.path.exists(up):
        print(f"transform/{name:14s} (no rust output)")
        continue
    R, U = matrix(rp), matrix(up)
    d = max(abs(a - b) for g in R for a, b in zip(R[g], U[g]))
    print(f"transform/{name:14s} max|diff| = {d:.3e}")
