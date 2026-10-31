#!/bin/bash
# Download the GEO supplementary count data used by realdata.R
# usage: fetch.sh <download dir>
set -eu
OUT=$1
mkdir -p "$OUT"
cd "$OUT"
base=https://ftp.ncbi.nlm.nih.gov/geo/series

# NCBI's FTP-over-HTTPS occasionally drops transfers (curl exit 18), so retry
dl() {
  local url=$1 dest=$2 i
  for i in 1 2 3 4 5; do
    if curl -sfL -o "$dest" "$url"; then return 0; fi
    echo "retrying $url ($i)" >&2
    sleep 3
  done
  echo "failed to download $url" >&2
  return 1
}

dl $base/GSE151nnn/GSE151925/suppl/GSE151925_readcount.txt.gz GSE151925_readcount.txt.gz
dl $base/GSE46nnn/GSE46391/suppl/GSE46391_HP_and_LP_Total_gene_reads.txt.gz GSE46391_HP_and_LP_Total_gene_reads.txt.gz
dl $base/GSE130nnn/GSE130777/suppl/GSE130777_mgrA_vs_LAC.csv.gz GSE130777_mgrA_vs_LAC.csv.gz
for g in GSE100nnn/GSE100097 GSE55nnn/GSE55197 GSE220nnn/GSE220559; do
  id=${g#*/}
  mkdir -p "$id"
  dl "$base/$g/suppl/${id}_RAW.tar" "$id/RAW.tar"
  tar xf "$id/RAW.tar" -C "$id"
done
echo "downloaded to $OUT"
