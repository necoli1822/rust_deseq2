#!/bin/bash
# Build cost of each crate alone: unique dependency count and a clean release build time.
# usage: build_cost.sh <work dir>   (cargo/rustc from PATH)
W=$1
mkdir -p "$W"
for spec in "hdf5-reader=0.11" "hdf5-pure=0.47" "rust-hdf5=0.7" "hdf5-rust=2"; do
  c=${spec%%=*}; v=${spec#*=}
  P=$W/only-$c
  mkdir -p "$P/src"
  printf '[package]\nname = "only"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n%s = "%s"\n' "$c" "$v" > "$P/Cargo.toml"
  printf 'fn main() {}\n' > "$P/src/main.rs"
  rm -rf "$P/target"
  deps=$(cargo tree --manifest-path "$P/Cargo.toml" --prefix none -e normal 2>/dev/null | sed 's/ (\*)//' | sort -u | grep -vc '^only ')
  s=$(date +%s.%N)
  cargo build --release --manifest-path "$P/Cargo.toml" -q 2>/dev/null
  t=$(echo "$(date +%s.%N) - $s" | bc)
  msrv=$(cargo info "$c@$v" 2>/dev/null | grep '^rust-version' | cut -d' ' -f2)
  lic=$(cargo info "$c@$v" 2>/dev/null | grep '^license' | cut -d' ' -f2-)
  echo "$c	deps=$deps	clean_build_s=$t	rust_version=$msrv	license=$lic"
done
