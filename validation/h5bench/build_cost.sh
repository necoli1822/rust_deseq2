#!/bin/bash
# Build cost of each crate alone: unique dependency count and a clean release build time.
# usage: build_cost.sh <work dir>   (cargo/rustc from PATH)
W=$1
mkdir -p "$W"
# crate=version[=feature]
for spec in "hdf5-pure=0.47" "rust-hdf5=0.7" "rust-hdf5=0.7=parallel"; do
  IFS='=' read -r c v feat <<< "$spec"
  label=$c${feat:++$feat}
  P=$W/only-$label
  mkdir -p "$P/src"
  if [ -n "$feat" ]; then dep="$c = { version = \"$v\", features = [\"$feat\"] }"; else dep="$c = \"$v\""; fi
  printf '[package]\nname = "only"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n%s\n' "$dep" > "$P/Cargo.toml"
  printf 'fn main() {}\n' > "$P/src/main.rs"
  rm -rf "$P/target"
  deps=$(cargo tree --manifest-path "$P/Cargo.toml" --prefix none -e normal 2>/dev/null | sed 's/ (\*)//' | sort -u | grep -vc '^only ')
  s=$(date +%s.%N)
  cargo build --release --manifest-path "$P/Cargo.toml" -q 2>/dev/null
  t=$(echo "$(date +%s.%N) - $s" | bc)
  msrv=$(cargo info "$c@$v" 2>/dev/null | grep '^rust-version' | cut -d' ' -f2)
  lic=$(cargo info "$c@$v" 2>/dev/null | grep '^license' | cut -d' ' -f2-)
  echo "$label	deps=$deps	clean_build_s=$t	rust_version=$msrv	license=$lic"
done
