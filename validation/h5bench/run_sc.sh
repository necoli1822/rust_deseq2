#!/bin/bash
# Single-cell part of h5bench.
# usage: run_sc.sh <h5bench> <h5bench --features zlibrs> <h5bench --features par> <work dir> <file:layout>...
#   layout: 10x | h5ad.  Env: PYTHON (python with h5py, default: tool/h5py pixi env), REPEATS (3)
# For each file and mode (full, partial): h5py reference (checksums; time and peak RSS of the
# read alone), then every reader:
#   hdf5-pure (File::open / open_streaming / 8 application threads), with flate2's default
#   miniz_oxide backend and with its zlib-rs backend;
#   rust-hdf5 (single thread / 8 application threads / its `parallel` feature with 8 and all
#   threads).
# Prints: file mode reader median_seconds max_rss_MB check
B=$1; BZ=$2; BP=$3; W=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-"pixi run --manifest-path $HERE/../../tool/h5py/pixi.toml python"}
N=${REPEATS:-3}
export H5BENCH_THREADS=8
mkdir -p "$W"

# run "<label>" <cmd...>: one run that checks the values against the spec, then N runs
# without the check (H5BENCH_NO_CHECK) for the median seconds and the max RSS of the read
measure() {
  local label=$1; shift
  local secs=() rss=0 line check kb
  if [ "$label" = h5py ]; then
    check=reference
  else
    line=$("$@")
    check=$(echo "$line" | awk -F'\t' '{print $NF}')
    case "$line" in *FAILED*) printf '%s\t%s\t-\t-\t%s\n' "$CUR" "$label" "$line"; return ;; esac
  fi
  for _ in $(seq "$N"); do
    line=$( { H5BENCH_NO_CHECK=1 /usr/bin/time -v "$@" 2> "$W/time.txt"; } )
    kb=$(grep "Maximum resident set size" "$W/time.txt" | awk '{print $NF}')
    [ "$kb" -gt "$rss" ] && rss=$kb
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
    for r in hdf5-pure hdf5-pure-stream hdf5-pure-threads; do
      measure "$r (miniz_oxide)" "$B" sc "$r" "$f" "$spec" "$mode"
      measure "$r (zlib-rs)" "$BZ" sc "$r" "$f" "$spec" "$mode"
    done
    measure "rust-hdf5" "$B" sc rust-hdf5 "$f" "$spec" "$mode"
    measure "rust-hdf5-threads (8)" "$B" sc rust-hdf5-threads "$f" "$spec" "$mode"
    RAYON_NUM_THREADS=8 measure "rust-hdf5+parallel (8)" "$BP" sc rust-hdf5 "$f" "$spec" "$mode"
    measure "rust-hdf5+parallel (all)" "$BP" sc rust-hdf5 "$f" "$spec" "$mode"
  done
done
