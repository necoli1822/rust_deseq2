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

Both trees are pure Rust: no `-sys`, `cc` or `cmake`. flate2's `zlib-rs` backend for
`hdf5-pure` is pure Rust too and adds the `zlib-rs` crate that `rust-hdf5` already uses. Deflate comes from `miniz_oxide`
(and `zlib-rs` for `rust-hdf5`). `hdf5-pure`'s `fast-deflate` feature would add zlib-ng, which
is C, and was not used. The `parallel` feature adds rayon, which `rust_deseq2` already depends
on.

crates.io (2026-10-31):
- `hdf5-pure`: 140,522 downloads in 90 days, 59 versions since 2026-03.
- `rust-hdf5`: 10,856 downloads in 90 days, 50 versions since 2026-04.

Both are 0.x and release often, so either one must be pinned (`~0.x`) and re-checked with
this harness when upgraded.

### Fairness re-run: same decompressor, application-level threads

The runs above are not a like-for-like comparison:
- **Different inflaters:** `rust-hdf5` uses `zlib-rs`, while `hdf5-pure` used `flate2`'s
  default `miniz_oxide`.
- **No `parallel` in `hdf5-pure`:** it has no parallel feature. Its CHANGELOG says the old
  `parallel` feature gated a path no public function reached, so it was removed; the
  `open_streaming` doc comment that still mentions it is stale.

So both crates were re-run with every variant (`run_sc.sh`, 3 runs each; all 100 cases again
identical to h5py):
- `hdf5-pure` with `flate2`'s pure-Rust `zlib-rs` backend (the `zlibrs` feature of this
  harness), the same inflater as `rust-hdf5`;
- application-level threads for both: a whole dataset is read as 8 row ranges in 8 threads
  (`*-threads`). `hdf5_pure::File` is `Sync` and is shared. `rust_hdf5::H5File` is not
  (`Rc` inside unless its `threadsafe` feature is on), so each thread opens the file itself.

This run had another job on the machine (load average 47-51 of 128 threads, a `zstd` using
about 28 cores). Absolute times are higher than above (h5py most of all), but all variants
ran under the same conditions.

Full reads, seconds (peak RSS in MB):

| Reader | pbmc_10k_v3 | medium.10x.h5 | medium.h5ad | large.10x.h5 | large.h5ad |
|---|---:|---:|---:|---:|---:|
| `hdf5-pure`, miniz_oxide | 0.66 | 1.66 | 1.51 | 12.1 (4,125) | 13.3 (3,032) |
| `hdf5-pure`, zlib-rs | 0.57 | 1.17 | 1.08 | 8.52 (4,125) | 8.99 (3,034) |
| `hdf5-pure` stream, zlib-rs | 0.54 | 1.12 | 1.15 | 8.21 (3,454) | 8.51 (2,313) |
| `rust-hdf5` (1 thread) | 0.47 | 1.09 | 1.02 | 7.24 (3,452) | 7.22 (2,318) |
| `hdf5-pure` 8 threads, miniz_oxide | 0.41 | 0.71 | 0.43 | 3.93 (6,217) | 4.07 (4,082) |
| `hdf5-pure` 8 threads, zlib-rs | 0.33 | 0.62 | 0.47 | 4.28 (6,280) | 3.14 (4,169) |
| `rust-hdf5` 8 threads | 0.29 | 0.59 | 0.40 | 3.16 (3,758) | 1.98 (2,479) |
| `rust-hdf5` `parallel`, 8 threads | **0.078** | **0.138** | **0.081** | **0.57** (3,493) | **0.42** (2,344) |
| `rust-hdf5` `parallel`, all threads | 0.083 | 0.137 | 0.086 | 0.60 (3,492) | 0.42 (2,346) |
| h5py (loaded machine) | 3.68 | 3.77 | 1.62 | 20.2 (3,509) | 12.8 (2,402) |

Partial reads (1,000 cells) were unchanged in ranking:
- `rust-hdf5` single-threaded: 0.087-0.110 s, 45 MB.
- `rust-hdf5` `parallel`: 0.049-0.088 s.
- `hdf5-pure` streaming with zlib-rs: 0.089-0.216 s, 44-58 MB.
- `hdf5-pure` `File::open`: 0.38-0.45 s, 730-770 MB.

What the fair comparison changes:
- **Single-threaded, same inflater:** the gap shrinks from 1.6-1.8x to 1.1-1.2x. Most of
  it was miniz_oxide vs zlib-rs.
- **Application-level threading** helps both by 2-4x. `rust-hdf5` stays ahead (1.98-3.16
  s vs 3.14-4.28 s), and `hdf5-pure` threads on a buffered `File::open` need 1.6-1.8x the
  memory.
- **Built-in chunk-level parallelism** is what `hdf5-pure` lacks. With the same 8 threads,
  `rust-hdf5` `parallel` (0.42-0.57 s) is about 7x faster than the best `hdf5-pure` setup
  (3.1-4.3 s) and 4-5x faster than hand-split threads on `rust-hdf5` itself.

## Recommendation: `rust-hdf5` with `parallel`

For single-cell-sized files the difference holds up under a like-for-like comparison:
- **Speed:** single-threaded with the same inflater, `rust-hdf5` is only 1.1-1.2x faster.
  The decisive difference is its built-in chunk-level parallel decompression, which
  `hdf5-pure` does not have. With 8 threads it reads the 672 MB 10x file in 0.57 s, against
  3.9-4.3 s for the best `hdf5-pure` setup (zlib-rs + 8 application threads), about 7x
  faster. 8 threads is as fast as 128.
- **Memory:** `rust-hdf5` reads only what is needed (45 MB for a 1,000-cell range read).
- **Cost:** the extra build cost is small: 6 s and 12 more dependencies, one of which
  (rayon) this crate already uses.

`hdf5-pure`'s advantages are lighter: fewer dependencies, a dual license, more users, and a
`File` that can be shared between threads as is. Its `File::open` loads the whole file. If
it were chosen, it would need `open_streaming`, flate2's `zlib-rs` backend and threading in
the caller, and it would still be about 7x slower on large files.

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
cargo build --release --manifest-path validation/h5bench/Cargo.toml --target-dir /tmp/h5b-zr --features zlibrs
# single-cell files (tool/h5py: python, h5py, anndata, numpy, scipy)
pixi run --manifest-path tool/h5py/pixi.toml python validation/h5bench/gen_sc.py /tmp/sc 150000 2000 large
REPEATS=3 bash validation/h5bench/run_sc.sh /tmp/h5b/release/h5bench /tmp/h5b-zr/release/h5bench /tmp/h5b-par/release/h5bench /tmp/scout \
  /tmp/sc/large.10x.h5:10x /tmp/sc/large.h5ad:h5ad
# kallisto files (tool/kallisto 0.52.0, tool/kallisto-0.46 0.46.2; reads from sim_reads.py)
REPEATS=20 bash validation/h5bench/run.sh /tmp/h5b/release/h5bench /tmp/h5b-par/release/h5bench /tmp/kout \
  path/to/abundance.h5 ...
# build cost and native-dependency check
bash validation/h5bench/build_cost.sh /tmp/h5cost
bash validation/h5bench/native_deps.sh /tmp/h5cost
```
