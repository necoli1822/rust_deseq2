"""Exact comparison of an h5bench TSV with the rhdf5 TSV. usage: compare.py <r.tsv> <rust.tsv>"""
import csv, sys


def load(p):
    rows = list(csv.reader(open(p), delimiter="\t"))
    return rows[0], rows[1:]


hr, R = load(sys.argv[1])
hu, U = load(sys.argv[2])
problems = []
if hr != hu:
    problems.append(f"columns differ: R {len(hr)} vs Rust {len(hu)}")
if len(R) != len(U):
    problems.append(f"rows differ: R {len(R)} vs Rust {len(U)}")
ids = sum(1 for r, u in zip(R, U) if r[0] != u[0])
vals = sum(1 for r, u in zip(R, U) for a, b in zip(r[1:], u[1:]) if float(a) != float(b))
n = sum(len(r) - 1 for r in R)
if ids:
    problems.append(f"{ids} IDs differ")
if vals:
    problems.append(f"{vals} of {n} values differ")
print("identical (%d IDs, %d values)" % (len(R), n) if not problems else "; ".join(problems))
