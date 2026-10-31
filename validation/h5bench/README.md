# Pure-Rust HDF5 readers on kallisto `abundance.h5`

Which crate should read kallisto's `abundance.h5` for `--quant-type kallisto`? Four
pure-Rust candidates (no libhdf5) are compared here on real kallisto output against R's
`rhdf5`, which is what `tximport(type = "kallisto")` uses (`read_kallisto_h5`). The
libhdf5 bindings `hdf5` and `hdf5-metno` are excluded because they are not pure Rust.

Date: 2026-10-31. rustc 1.99.1 (conda-forge, `tool/rust/`; the system default 1.92 is
untouched), Intel Xeon Gold 6430. R 4.5.3, rhdf5 2.56.1.

## Test files

kallisto `quant` on simulated paired-end reads (`sim_reads.py`, e.g. `sim_reads.py out 55000 1 400000`;
transcripts are random sequences, genes have 1-3 isoforms sharing exons):

| File | kallisto | Transcripts | Bootstraps | Size |
|---|---|---:|---:|---:|
| small | 0.52.0 | 544 | 5 | 68 KB |
| large | 0.52.0 | 98,941 | 30 | 5.7 MB |
| small | 0.46.2 | 544 | 5 | 68 KB |
| large | 0.46.2 | 98,941 | 30 | 5.8 MB |

Layout of both versions (rhdf5): `aux/ids` fixed-length strings (12 bytes, not
variable-length), `est_counts`, `aux/eff_lengths` and `bootstrap/bs<k>` float64, all
chunked with the deflate filter; superblock version 0 in both.

## Results

What was read: `aux/ids`, `est_counts`, `aux/eff_lengths` (the three datasets tximport
reads) and every `bootstrap/bs<k>`. Accuracy is an exact comparison of every ID and
every float64 with rhdf5's values. Times are the median of 20 full open-and-read passes
(file in page cache).

| Crate | Version | All 4 files vs rhdf5 | small (ms) | large 0.52 (ms) | large 0.46 (ms) |
|---|---|---|---:|---:|---:|
| `hdf5-pure` | 0.47.0 | identical | **0.24** | 77 | 79 |
| `rust-hdf5` | 0.7.9 | identical | 0.30 | **35** | **56** |
| `hdf5-reader` | 0.11.0 | identical | 0.36 | 86 | 89 |
| `hdf5-rust` | 2.0.0 | identical | 0.85 | 201 | 232 |
| R `rhdf5` (reference) | 2.56.1 | — | 10 | 135 | 134 |

Large file: 98,941 IDs and 3,166,112 float64 values, all identical for every crate.

| Crate | Unique deps | Clean release build | rust-version | License | crates.io downloads (90 days) | Versions since |
|---|---:|---:|---|---|---:|---|
| `hdf5-pure` | 8 | 11.8 s | 1.89 | MIT OR Apache-2.0 | 140,522 | 59 since 2026-03 |
| `rust-hdf5` | 14 | 19.5 s | 1.89 | MIT | 10,856 | 50 since 2026-04 |
| `hdf5-reader` | 40 | 8.1 s | 1.81 | MIT OR Apache-2.0 | 44,322 | 20 since 2026-03 |
| `hdf5-rust` | 1 | 1.5 s | 1.99 | MIT OR Apache-2.0 | 807 | 9 since 2026-08 |

Every dependency tree is pure Rust: no `-sys`, `cc` or `cmake`; deflate comes from
`miniz_oxide` (and `zlib-rs` for `rust-hdf5`). `hdf5-reader` pulls in `ndarray` 0.17 (this
crate uses 0.16, so two ndarray versions would be compiled) and `rayon`/`lz4` by default.

API notes:
- `hdf5-pure`: `File::open`, `dataset(path)?.read_f64()`, `read_string()`; returns `Vec`s.
- `rust-hdf5`: `H5File::open`, `dataset(path)?.read_raw::<f64>()`, `read_strings()`.
- `hdf5-reader`: `Hdf5File::open`, `dataset(path)?.read_array::<f64>()` (an ndarray
  `ArrayD`), `read_strings()`.
- `hdf5-rust`: `read_f64` returns IEEE bit patterns (`Vec<u64>`); there is no string-dataset
  reader, so the IDs were decoded by hand from `read_raw` bytes. Reads the whole file into
  memory first.

## Recommendation

All four read kallisto's files exactly, so the choice is about cost and maintenance:

- **`hdf5-pure`**: few dependencies (8), the most used (140k downloads in 90 days), a
  direct `Vec<f64>`/`Vec<String>` API, and fast enough. The large file takes 77 ms, faster
  than R's 135 ms and negligible next to a DESeq2 run. **Recommended.**
- `rust-hdf5` is the fastest on the large file (35-56 ms, about 1.5-2x `hdf5-pure`), but
  the gain is about 40 ms per file (about 4 s for 100 samples). It has more dependencies (14)
  and the longest build (19.5 s), and is used about 13x less. A reasonable choice if
  thousands of files are read per run.
- `hdf5-reader` is fine but brings 40 dependencies, including a second ndarray.
- `hdf5-rust` has almost no dependencies, but it is the slowest, its API is low-level, and
  it is very new (807 downloads).

Both `hdf5-pure` (59 releases since 2026-03) and `rust-hdf5` (50 since 2026-04) release often
and are 0.x, so minor versions may break the API: pin the minor version (e.g.
`hdf5-pure = "~0.47"`) and re-run this benchmark when upgrading.

Caveat: these files are what kallisto writes (superblock 0, fixed-length strings,
deflate). Other HDF5 layouts (superblock 2/3, variable-length strings, other filters)
were not tested, and tximport only needs kallisto's.

## Reproduce

```bash
# the harness: rhdf5 reference + timing, then each crate (accuracy and timing)
CARGO_TARGET_DIR=/tmp/h5bench-target pixi run --manifest-path tool/rust/pixi.toml \
  cargo build --release --manifest-path validation/h5bench/Cargo.toml
REPEATS=20 bash validation/h5bench/run.sh /tmp/h5bench-target/release/h5bench /tmp/h5out \
  path/to/abundance.h5 ...
# build cost and native-dependency check (cargo from tool/rust)
pixi run --manifest-path tool/rust/pixi.toml bash validation/h5bench/build_cost.sh /tmp/h5cost
pixi run --manifest-path tool/rust/pixi.toml bash validation/h5bench/native_deps.sh /tmp/h5cost
```
kallisto: `tool/kallisto/` (0.52.0) and `tool/kallisto-0.46/` (0.46.2) pixi environments.
