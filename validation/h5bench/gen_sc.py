"""Write synthetic single-cell HDF5 files for h5bench.

usage: gen_sc.py <out dir> <n_cells> <genes per cell> <tag>
Writes <out>/<tag>.10x.h5 (Cell Ranger v3 layout: /matrix/{barcodes, data, indices,
indptr, shape, features/{id, name, feature_type, genome}}, gzip, CSC with one column
per cell) and <tag>.h5ad (AnnData, X as CSR float32, obs/var with variable-length
string indexes and a categorical, written by anndata with gzip).
"""
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

out, n_cells, per_cell, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
n_genes = 33538  # GRCh38 gene count used by Cell Ranger references
rng = np.random.default_rng(1)

nnz = rng.poisson(per_cell, n_cells).clip(50, n_genes)
indptr = np.zeros(n_cells + 1, dtype=np.int64)
np.cumsum(nnz, out=indptr[1:])
indices = np.empty(indptr[-1], dtype=np.int64)
gene_p = rng.gamma(0.3, 1.0, n_genes)
gene_p /= gene_p.sum()
for c in range(n_cells):
    idx = rng.choice(n_genes, nnz[c], replace=False, p=gene_p)
    idx.sort()
    indices[indptr[c]:indptr[c + 1]] = idx
data = (rng.geometric(0.45, indptr[-1])).astype(np.int32)
barcodes = np.array([f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(n_cells)], dtype="S18")
gene_ids = np.array([f"ENSG{i:011d}" for i in range(n_genes)], dtype="S15")
gene_names = np.array([f"GENE{i}" if i % 97 else f"MT-ND{i % 7}" for i in range(n_genes)])

kw = dict(compression="gzip", compression_opts=4)
with h5py.File(f"{out}/{tag}.10x.h5", "w") as f:
    m = f.create_group("matrix")
    m.create_dataset("barcodes", data=barcodes, **kw)
    m.create_dataset("data", data=data, chunks=(80000,), **kw)
    m.create_dataset("indices", data=indices, chunks=(80000,), **kw)
    m.create_dataset("indptr", data=indptr, chunks=True, **kw)
    m.create_dataset("shape", data=np.array([n_genes, n_cells], dtype=np.int32))
    g = m.create_group("features")
    g.create_dataset("id", data=gene_ids, **kw)
    g.create_dataset("name", data=gene_names.astype("S16"), **kw)
    g.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_genes, dtype="S15"), **kw)
    g.create_dataset("genome", data=np.array([b"GRCh38"] * n_genes, dtype="S6"), **kw)
    g.create_dataset("_all_tag_keys", data=np.array([b"genome"], dtype="S6"))

# AnnData: cells x genes CSR, i.e. the same arrays read row-wise
X = sp.csr_matrix((data.astype(np.float32), indices.astype(np.int32), indptr), shape=(n_cells, n_genes))
obs = pd.DataFrame({"cell_type": pd.Categorical(rng.choice(["T", "B", "NK", "Mono"], n_cells))},
                   index=pd.Index(barcodes.astype(str), name=None))
var = pd.DataFrame({"gene_name": gene_names}, index=pd.Index(gene_ids.astype(str)))
ad.AnnData(X=X, obs=obs, var=var).write_h5ad(f"{out}/{tag}.h5ad", compression="gzip", compression_opts=4)
print(tag, "cells", n_cells, "nnz", int(indptr[-1]))
