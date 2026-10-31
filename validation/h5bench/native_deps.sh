#!/bin/bash
# List native-code (-sys, cc, cmake) and compression/ndarray dependencies of each crate alone.
# usage: native_deps.sh <work dir of build_cost.sh>
W=$1
for c in hdf5-reader hdf5-pure rust-hdf5 hdf5-rust; do
  t=$(cargo tree --manifest-path "$W/only-$c/Cargo.toml" --prefix none -e normal,build 2>/dev/null \
      | sed 's/ (\*)//' | sort -u | grep -E '(-sys |^cc |^cmake |^ndarray |^flate2 |^miniz_oxide |^zlib-rs |^libz)' | tr '\n' ';')
  echo "$c: ${t:-none}"
done
