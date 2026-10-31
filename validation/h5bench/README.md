# Pure-Rust HDF5 readers: hdf5-pure vs rust-hdf5

Which crate should read HDF5 input: kallisto's `abundance.h5` for `--quant-type kallisto`
now, and single-cell matrices (10x Cell Ranger `.h5`, AnnData `.h5ad`) later? Two
pure-Rust readers (no libhdf5) are compared on accuracy, speed, peak memory and build
cost:

- `hdf5-pure` 0.47.0, opened two ways: `File::open` (reads the whole file into memory)
  and `File::open_streaming` (fetches the needed regions on demand);
- `rust-hdf5` 0.7.9, with and without its `parallel` feature (rayon chunk
  decompression; all threads, and limited to 8 with `RAYON_NUM_THREADS=8`).

The references are R `rhdf5` (what `tximport(type = "kallisto")` uses) for kallisto and
h5py (libhdf5) for single-cell files.

Date: 2026-10-31. rustc 1.92.0 (the system toolchain; both crates need 1.89), Intel Xeon
Gold 6430 (128 threads), 250 GB RAM. R 4.5.3 / rhdf5 2.56.1, Python h5py and anndata from
`tool/h5py/`.

## Results

### Accuracy

Every reader returned identical data in every case:
- kallisto: every ID and float64 is identical to rhdf5's values;
- single-cell: every dataset's length and two exact checksums equal h5py's, namely Σx and
  Σ(i+1)·x over the values as u64 bit patterns, mod 2^64; strings are checksummed over
  their bytes;
- this covers fixed-length strings (10x), variable-length strings (`.h5ad`), int32,
  int64 and float32 arrays, and range reads.

### Single-cell files

Files:
- `pbmc_10k_v3`: the public 10x PBMC 10k v3 `filtered_feature_bc_matrix.h5`, written by
  Cell Ranger 3.0 (11,769 cells, 33,538 genes).
- `medium`/`large`: synthetic data written by `gen_sc.py` with 20,000 and 150,000 cells,
  about 2,000 genes per cell (40 M and 300 M non-zeros), each as a Cell Ranger v3 layout
  `.10x.h5` and an anndata `.h5ad`. Both are gzip level 4.

What is read:
- **full**: barcodes and gene IDs/names (`.h5ad`: the obs/var indexes), `indptr`, `data`,
  `indices`.
- **partial**: the same strings and `indptr`, plus the `data`/`indices` of 1,000 cells from
  the middle (a range read).

The time is the median of 3 runs, file in the page cache, timing the read only. Peak RSS
is measured for runs without the verification step.

Seconds:

| File | Mode | h5py | hdf5-pure | hdf5-pure stream | rust-hdf5 | rust-hdf5 parallel (128 / 8 threads) |
|---|---|---:|---:|---:|---:|---:|
| pbmc_10k_v3 (36 MB) | full | 0.71 | 0.53 | 0.51 | 0.32 | **0.059 / 0.061** |
| medium.10x.h5 (90 MB) | full | 1.47 | 1.31 | 1.30 | 0.77 | **0.075 / 0.074** |
| medium.h5ad (98 MB) | full | 1.38 | 1.27 | 1.27 | 0.77 | **0.063 / 0.064** |
| large.10x.h5 (672 MB) | full | 12.6 | 11.8 | 10.7 | 6.86 | **0.58 / 0.58** |
| large.h5ad (722 MB) | full | 11.4 | 11.6 | 11.6 | 6.73 | **0.40 / 0.42** |
| pbmc_10k_v3 | partial | 0.084 | 0.071 | 0.054 | 0.035 | **0.024 / 0.025** |
| large.10x.h5 | partial | 0.129 | 0.365 | 0.121 | 0.081 | **0.050 / 0.051** |
| large.h5ad | partial | 0.158 | 0.617 | 0.212 | 0.094 | **0.080 / 0.082** |

Peak RSS (MB):

| File | Mode | h5py | hdf5-pure | hdf5-pure stream | rust-hdf5 | rust-hdf5 parallel (128 / 8) |
|---|---|---:|---:|---:|---:|---:|
| large.10x.h5 | full | 3,513 | 4,125 | 3,454 | 3,452 | 3,491 / 3,494 |
| large.h5ad | full | 2,401 | 3,032 | 2,311 | 2,316 | 2,346 / 2,343 |
| large.10x.h5 | partial | 87 | 730 | 58 | **45** | 50 / 50 |
| large.h5ad | partial | 93 | 766 | 44 | **43** | 47 / 47 |

Full reads hold the decoded arrays (3.4 GB for `large.10x.h5`), so every reader is close
to that. `hdf5-pure`'s `File::open` adds the whole compressed file (+0.7 GB) and keeps it
for partial reads too (730 MB to read 1,000 cells). Its streaming open and `rust-hdf5` read
only what is asked for.

### kallisto `abundance.h5`

kallisto 0.52.0 and 0.46.2 output on simulated reads: 544 or 98,941 transcripts, 5 or 30
bootstraps, files of 68 KB to 5.8 MB, superblock 0, deflate, fixed-length IDs. Times are the
median of 20 passes, in ms:

| File | rhdf5 | hdf5-pure | hdf5-pure stream | rust-hdf5 | rust-hdf5 parallel |
|---|---:|---:|---:|---:|---:|
| small 0.52 | 8 | 0.25 | 0.36 | 0.26 | 0.35 |
| large 0.52 | 121 | 70.7 | 64.9 | **36.4** | 38.4 |
| small 0.46 | 9 | 0.26 | 0.33 | 0.29 | 0.36 |
| large 0.46 | 121 | 69.6 | 71.6 | **45.2** | 44.0 |

`parallel` does not help here: the datasets are small.

### Build cost

| | Unique deps | Clean release build | rust-version | License |
|---|---:|---:|---|---|
| `hdf5-pure` | 8 | 12.0 s | 1.89 | MIT OR Apache-2.0 |
| `rust-hdf5` | 14 | 18.3 s | 1.89 | MIT |
| `rust-hdf5` + `parallel` | 20 | 18.6 s | 1.89 | MIT |

Both trees are pure Rust: no `-sys`, `cc` or `cmake`. Deflate comes from `miniz_oxide`
(and `zlib-rs` for `rust-hdf5`). `hdf5-pure`'s `fast-deflate` feature would add zlib-ng, which
is C, and was not used. The `parallel` feature adds rayon, which `rust_deseq2` already depends
on.

crates.io (2026-10-31):
- `hdf5-pure`: 140,522 downloads in 90 days, 59 versions since 2026-03.
- `rust-hdf5`: 10,856 downloads in 90 days, 50 versions since 2026-04.

Both are 0.x and release often, so either one must be pinned (`~0.x`) and re-checked with
this harness when upgraded.

## Recommendation: `rust-hdf5` with `parallel`

For single-cell-sized files the difference is large and consistent:
- **Speed:** single-threaded, `rust-hdf5` is 1.6-1.8x faster than `hdf5-pure`. With
  `parallel` it reads the 672 MB 10x file in 0.58 s, where `hdf5-pure` takes 10.7-11.8 s
  and h5py (libhdf5) 12.6 s. That is a 20x gain, the same with 8 threads as with 128.
- **Memory:** `rust-hdf5` reads only what is needed (45 MB for a 1,000-cell range read).
- **Cost:** the extra build cost is small: 6 s and 12 more dependencies, one of which
  (rayon) this crate already uses.

`hdf5-pure`'s advantages are lighter: fewer dependencies, a dual license and more users.
Its `File::open` loads the whole file. If it were chosen, `open_streaming` would be
needed, and that mode is still 1.3-2.3x slower than `rust-hdf5` single-threaded.

For kallisto files both are fast enough (≤ 72 ms for 99k transcripts vs rhdf5's 121 ms),
and `rust-hdf5` is again the faster of the two.

Not covered:
- Other compression filters: Blosc, LZ4 and zstd appear in some single-cell files;
  `rust-hdf5` supports them behind features, `hdf5-pure` only through custom code.
- Superblock 2/3 files.
- Cold-cache (disk-bound) reads.

## First round (4 crates, kallisto files only)

`hdf5-reader` 0.11 and `hdf5-rust` 2.0 were also tested on the kallisto files. Both read
them exactly, but they were dropped:
- `hdf5-reader`: 40 dependencies including a second ndarray (0.17), 86-89 ms on the large
  file;
- `hdf5-rust`: needs Rust 1.99, has no string-dataset reader, reads the whole file into
  memory, and took 201-232 ms on the large file.

Their code and builds were removed.

## Reproduce

```bash
# h5bench with and without rust-hdf5's parallel feature
cargo build --release --manifest-path validation/h5bench/Cargo.toml --target-dir /tmp/h5b
cargo build --release --manifest-path validation/h5bench/Cargo.toml --target-dir /tmp/h5b-par --features par
# single-cell files (tool/h5py: python, h5py, anndata, numpy, scipy)
pixi run --manifest-path tool/h5py/pixi.toml python validation/h5bench/gen_sc.py /tmp/sc 150000 2000 large
REPEATS=3 bash validation/h5bench/run_sc.sh /tmp/h5b/release/h5bench /tmp/h5b-par/release/h5bench /tmp/scout \
  /tmp/sc/large.10x.h5:10x /tmp/sc/large.h5ad:h5ad
# kallisto files (tool/kallisto 0.52.0, tool/kallisto-0.46 0.46.2; reads from sim_reads.py)
REPEATS=20 bash validation/h5bench/run.sh /tmp/h5b/release/h5bench /tmp/h5b-par/release/h5bench /tmp/kout \
  path/to/abundance.h5 ...
# build cost and native-dependency check
bash validation/h5bench/build_cost.sh /tmp/h5cost
bash validation/h5bench/native_deps.sh /tmp/h5cost
```
