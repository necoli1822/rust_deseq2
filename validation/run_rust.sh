#!/bin/bash
# usage: run_rust.sh <binary> <data_dir> <tag>
B=$1; D=$2; T=$3
run() { local s=$1; shift; "$B" run -c "$D/$s/counts.csv" -m "$D/$s/metadata.csv" -o "$D/$s/rust_$T.tsv" "$@" > "$D/$s/rust_$T.log" 2>&1 || echo "$s: FAILED ($(tail -1 "$D/$s/rust_$T.log"))"; }
W="-d condition --numerator trt --denominator ctrl"
run basic_parametric $W
run basic_local $W --fit-type local
run basic_mean $W --fit-type mean
run use_t $W --use-t
run lfcthr $W --lfc-threshold 1
run shrink_normal $W --shrinkage --shrinkage-method normal
run shrink_apeglm $W --shrinkage --shrinkage-method apeglm
run shrink_ashr $W --shrinkage --shrinkage-method ashr
run three_CvsB -d condition --numerator C --denominator B
run lrt_three -d condition --test lrt --reduced "~1"
run batch $W --covariate batch
run shrink_apeglm_batch $W --covariate batch --shrinkage --shrinkage-method apeglm
run lrt_batch -d condition --covariate batch --test lrt --reduced "~batch"
run continuous $W --continuous age
run lrt_continuous -d condition --continuous age --test lrt --reduced "~age"
run outlier_replace $W
run outlier_replace_local $W --fit-type local
run outlier_filter $W
run poscounts $W --sf-type poscounts
run geno_KOvsWT -d genotype --numerator KO --denominator WT --reference genotype=WT
run small_2v2 $W
run outlier_batch16 $W --covariate batch
run unbal_2v5 $W
run small_var_2v2 $W
run small_var_2v3 $W
TD=$D/transform
"$B" normalize -c $TD/counts.csv -o $TD/rust_norm_$T.tsv >/dev/null 2>&1 || echo "normalize FAILED"
"$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition -o $TD/rust_vst_TRUE_$T.tsv --blind >/dev/null 2>&1 || echo "vst blind FAILED"
"$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition -o $TD/rust_vst_FALSE_$T.tsv >/dev/null 2>&1 || echo "vst FAILED"
"$B" rlog -c $TD/counts.csv -m $TD/metadata.csv -d condition -o $TD/rust_rlog_TRUE_$T.tsv --blind >/dev/null 2>&1 || echo "rlog blind FAILED"
"$B" rlog -c $TD/counts.csv -m $TD/metadata.csv -d condition -o $TD/rust_rlog_FALSE_$T.tsv >/dev/null 2>&1 || echo "rlog FAILED"
for ft in local mean; do
  "$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition --method $ft --blind -o $TD/rust_vst${ft}_TRUE_$T.tsv >/dev/null 2>&1 || echo "vst $ft blind FAILED"
  "$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition --method $ft -o $TD/rust_vst${ft}_FALSE_$T.tsv >/dev/null 2>&1 || echo "vst $ft FAILED"
done
