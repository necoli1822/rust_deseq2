#!/bin/bash
# usage: run.sh <h5bench binary> <work dir> <abundance.h5>...
# For each file: rhdf5 reference + timing, then each crate (accuracy vs rhdf5, timing).
B=$1; W=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
RS="pixi run --manifest-path $HERE/../../tool/deseq2/pixi.toml Rscript"
[ -n "$H5BENCH_RSCRIPT" ] && RS=$H5BENCH_RSCRIPT
mkdir -p "$W"
REPEATS=${REPEATS:-20}
i=0
for f in "$@"; do
  i=$((i + 1))
  echo "== $f ($(du -h "$f" | cut -f1))"
  $RS "$HERE/r_dump.R" "$f" "$W/r_$i.tsv" "$REPEATS" 2>/dev/null
  for c in hdf5-reader hdf5-pure rust-hdf5 hdf5-rust; do
    line=$("$B" "$c" "$f" "$W/${c}_$i.tsv" "$REPEATS")
    case "$line" in
      *FAILED*) echo "$line" ;;
      *) echo "$line	$(python3 "$HERE/compare.py" "$W/r_$i.tsv" "$W/${c}_$i.tsv")" ;;
    esac
  done
done
