#!/bin/bash
# kallisto part of h5bench.
# usage: run.sh <h5bench> <h5bench built with --features par> <work dir> <abundance.h5>...
# For each file: rhdf5 reference + timing, then each reader (accuracy vs rhdf5, timing).
B=$1; BP=$2; W=$3; shift 3
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
  for c in hdf5-pure hdf5-pure-stream rust-hdf5 rust-hdf5+parallel; do
    case $c in
      rust-hdf5+parallel) line=$("$BP" kallisto rust-hdf5 "$f" "$W/${c}_$i.tsv" "$REPEATS") ;;
      *) line=$("$B" kallisto "$c" "$f" "$W/${c}_$i.tsv" "$REPEATS") ;;
    esac
    case "$line" in
      *FAILED*) echo "$line" ;;
      *) echo "$line	$(python3 "$HERE/compare.py" "$W/r_$i.tsv" "$W/${c}_$i.tsv")" ;;
    esac
  done
done
