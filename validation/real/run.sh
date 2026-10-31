#!/bin/bash
# Compare rust_deseq2 against R DESeq2 on public bacterial RNA-seq data from GEO.
#
# usage: validation/real/run.sh <Rscript command> <work dir> [tag]
# Downloads the data and generates the R references once (skipped if present),
# builds the release binary and prints max |R - Rust| per scenario.
set -e
RSCRIPT=$1
WORK=$2
TAG=${3:-cur}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -f "$WORK/GSE151925_readcount.txt.gz" ]; then
  bash "$HERE/fetch.sh" "$WORK"
fi
if [ ! -d "$WORK/eco_lrt" ]; then
  $RSCRIPT "$HERE/realdata.R" "$WORK" "$WORK" "$HERE/.."
fi
cargo build --release --manifest-path "$HERE/../../Cargo.toml"
TARGET=${CARGO_TARGET_DIR:-$HERE/../../target}
set +e
bash "$HERE/real_run.sh" "$TARGET/release/rust_deseq2" "$WORK" "$TAG"
python3 "$HERE/../compare.py" "$WORK" "$TAG" | grep -v "^transform/"
python3 - "$WORK" "$TAG" <<'EOF'
import os, sys
D, T = sys.argv[1], sys.argv[2]
TD = os.path.join(D, "sal_transform")
for name in ["vst", "vstnsub", "rlog"]:
    r = [l.rstrip("\n").split("\t") for l in open(os.path.join(TD, f"r_T_{name}.tsv"))][1:]
    p = os.path.join(TD, f"rust_T_{name}_{T}.tsv")
    if not os.path.exists(p):
        print(f"sal_transform/{name}: no rust output")
        continue
    u = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1:] for l in list(open(p))[1:]}
    d = max(abs(float(a) - float(b)) for row in r for a, b in zip(row[1:], u[row[0]]))
    print(f"sal_transform/{name:8s} max|diff| = {d:.3e}")
EOF
