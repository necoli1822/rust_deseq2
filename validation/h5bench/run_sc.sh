#!/bin/bash
# Single-cell part of h5bench.
# usage: run_sc.sh <h5bench> <h5bench built with --features par> <work dir> <file:layout>...
#   layout: 10x | h5ad.  Env: PYTHON (python with h5py, default: tool/h5py pixi env), REPEATS (3)
# For each file and mode (full, partial): h5py reference (checksums; time and peak RSS of the
# read alone), then
# hdf5-pure, hdf5-pure-stream, rust-hdf5, rust-hdf5+parallel (all threads, and 8 threads).
# Prints: file mode reader median_seconds max_rss_MB check
B=$1; BP=$2; W=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-"pixi run --manifest-path $HERE/../../tool/h5py/pixi.toml python"}
N=${REPEATS:-3}
mkdir -p "$W"

# run "<label>" <cmd...>: one run that checks the values against the spec, then N runs
# without the check (H5BENCH_NO_CHECK) for the median seconds and the max RSS of the read
measure() {
  local label=$1; shift
  local secs=() rss=0 line check
  if [ "$label" = h5py ]; then
    check=reference
  else
    line=$("$@")
    check=$(echo "$line" | awk -F'\t' '{print $NF}')
    case "$line" in *FAILED*) printf '%s\t%s\t-\t-\t%s\n' "$CUR" "$label" "$line"; return ;; esac
  fi
  for _ in $(seq "$N"); do
    line=$( { H5BENCH_NO_CHECK=1 /usr/bin/time -v "$@" 2> "$W/time.txt"; } )
    r=$(grep "Maximum resident set size" "$W/time.txt" | awk '{print $NF}')
    [ "$r" -gt "$rss" ] && rss=$r
    secs+=("$(echo "$line" | grep -o 'seconds=[0-9.]*' | cut -d= -f2)")
  done
  med=$(printf '%s\n' "${secs[@]}" | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  printf '%s\t%s\t%.3f\t%d\t%s\n' "$CUR" "$label" "$med" $((rss / 1024)) "$check"
}

for fl in "$@"; do
  f=${fl%:*}; layout=${fl##*:}; name=$(basename "$f")
  for mode in full partial; do
    spec=$W/$name.$mode.spec
    CUR="$name ($(du -h "$f" | cut -f1)) $mode"
    $PY "$HERE/ref_sc.py" "$f" "$layout" "$mode" "$spec" > /dev/null   # spec + warm the page cache
    measure h5py $PY "$HERE/ref_sc.py" "$f" "$layout" "$mode" -
    for r in hdf5-pure hdf5-pure-stream rust-hdf5; do
      measure "$r" "$B" sc "$r" "$f" "$spec" "$mode"
    done
    measure "rust-hdf5+parallel (all threads)" "$BP" sc rust-hdf5 "$f" "$spec" "$mode"
    RAYON_NUM_THREADS=8 measure "rust-hdf5+parallel (8 threads)" "$BP" sc rust-hdf5 "$f" "$spec" "$mode"
  done
done
