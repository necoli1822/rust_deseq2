#!/bin/bash
# Compare rust_deseq2 against R DESeq2 on simulated scenarios.
#
# usage: validation/run.sh <Rscript command> <work dir> [tag]
#   e.g. validation/run.sh "pixi run --manifest-path tool/deseq2/pixi.toml Rscript" /tmp/deseq2-val
#
# Needs R with DESeq2, apeglm and ashr. Generates the data and R references once
# (skipped if <work dir>/data exists), builds the release binary, runs every
# scenario and prints max |R - Rust| per output column.
set -e
RSCRIPT=$1
WORK=$2
TAG=${3:-cur}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA="$WORK/data"

if [ ! -d "$DATA" ]; then
  mkdir -p "$DATA"
  for g in gen.R gen2.R gen3.R gen4.R gen5.R gen6.R; do
    echo "generating R references: $g"
    $RSCRIPT "$HERE/$g" "$DATA"
  done
fi

cargo build --release --manifest-path "$HERE/../Cargo.toml"
TARGET=${CARGO_TARGET_DIR:-$HERE/../target}
set +e
bash "$HERE/run_rust.sh" "$TARGET/release/rust_deseq2" "$DATA" "$TAG"
python3 "$HERE/compare.py" "$DATA" "$TAG"
