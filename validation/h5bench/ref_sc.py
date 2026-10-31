"""h5py reference for the single-cell part of h5bench.

usage: ref_sc.py <file> <10x|h5ad> <full|partial> <spec out | ->
Reads the datasets of the layout (all of them, or indptr plus the data/indices of 1000
cells from the middle), prints the elapsed seconds, and with a spec path writes one line
per dataset: path, kind, role, n, s1, s2 where s1 = sum(x) and s2 = sum((i+1) * x) over
the values as u64 bit patterns (ints sign-extended to 64 bits, floats as their IEEE
bits), mod 2^64; strings are hashed over the bytes of "\\n".join(strings).
"""
import sys
import time

import h5py
import numpy as np

LAYOUT = {
    "10x": [("matrix/barcodes", "str"), ("matrix/features/id", "str"), ("matrix/features/name", "str"),
            ("matrix/indptr", "indptr"), ("matrix/data", "nz"), ("matrix/indices", "nz")],
    "h5ad": [("obs/_index/values", "str"), ("var/_index/values", "str"),
             ("X/indptr", "indptr"), ("X/data", "nz"), ("X/indices", "nz")],
}
BLOCK = 1000


def kind(ds):
    if h5py.check_string_dtype(ds.dtype) is not None or ds.dtype.kind in "SO":
        return "str"
    return {("i", 4): "i32", ("i", 8): "i64", ("f", 4): "f32", ("f", 8): "f64"}[(ds.dtype.kind, ds.dtype.itemsize)]


def to_u64(x):
    if x.dtype.kind == "f":
        return x.view(np.uint32 if x.dtype.itemsize == 4 else np.uint64).astype(np.uint64)
    return x.astype(np.int64).view(np.uint64)


def sums(x):
    w = np.arange(1, len(x) + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(x.sum(dtype=np.uint64)), int((x * w).sum(dtype=np.uint64))


def main():
    path, layout, mode, spec = sys.argv[1:5]
    # timed: reading only (strings decoded to str/bytes objects, as the Rust readers
    # build Strings); checksum conversions happen afterwards
    t = time.perf_counter()
    got = []
    with h5py.File(path, "r") as f:
        indptr = None
        for p, role in LAYOUT[layout]:
            ds = f[p]
            k = kind(ds)
            if role == "str":
                v = ds[...]
            elif role == "indptr":
                v = indptr = ds[...]
            elif mode == "full":
                v = ds[...]
            else:
                s = (len(indptr) - 1) // 2
                lo, hi = int(indptr[s]), int(indptr[min(s + BLOCK, len(indptr) - 1)])
                v = ds[lo:hi]
            got.append((p, k, role, v))
    elapsed = time.perf_counter() - t
    if spec != "-":
        with open(spec, "w") as o:
            for p, k, role, v in got:
                if role == "str":
                    b = b"\n".join(s if isinstance(s, bytes) else s.encode() for s in v)
                    x = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
                else:
                    x = to_u64(v)
                s1, s2 = sums(x)
                o.write(f"{p}\t{k}\t{role}\t{len(v)}\t{s1}\t{s2}\n")
    print(f"h5py\t{mode}\tseconds={elapsed:.3f}")


main()
