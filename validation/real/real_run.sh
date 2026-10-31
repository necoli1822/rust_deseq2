#!/bin/bash
# usage: real_run.sh <binary> <realval dir> <tag>
B=$1; D=$2; T=$3
run() { local s=$1; shift; "$B" run -c "$D/$s/counts.csv" -m "$D/$s/metadata.csv" -o "$D/$s/rust_$T.tsv" "$@" > "$D/$s/rust_$T.log" 2>&1 || echo "$s: FAILED ($(tail -1 "$D/$s/rust_$T.log"))"; }
ABA="-d condition --numerator abaI --denominator WT --reference condition=WT"
run aba_abaI_vs_WT $ABA
run aba_abaI_vs_abaM -d condition --numerator abaI --denominator abaM --reference condition=WT
run aba_lrt -d condition --test lrt --reduced "~1"
run aba_apeglm $ABA --shrinkage --shrinkage-method apeglm
run aba_ashr $ABA --shrinkage --shrinkage-method ashr
run aba_weights $ABA --weights "$D/aba_weights/weights.csv"
SAL="-d condition --numerator LP --denominator HP"
run sal_HP_vs_LP $SAL
run sal_local $SAL --fit-type local
run sal_iterate $SAL --sf-type iterate
SAU="-d condition --numerator mgrA --denominator WT --reference condition=WT"
run sau_mgrA_vs_WT $SAU
run sau_poscounts $SAU --sf-type poscounts
run sau_apeglm $SAU --shrinkage --shrinkage-method apeglm
run mtb_carbon_batch -d carbon --covariate phase --numerator lipid --denominator dextrose
run mtb_lrt_phase -d carbon --covariate phase --test lrt --reduced "~phase"
run pae_2v2 -d condition --numerator att_c --denominator anox
run pae_6v4 -d condition --numerator heat50 --denominator exp
run eco_CIP_vs_CAZ -d condition --numerator CIP --denominator CAZ
run eco_lrt -d condition --test lrt --reduced "~1"
TD=$D/sal_transform
"$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition --blind -o $TD/rust_T_vst_$T.tsv >/dev/null 2>&1 || echo "vst FAILED"
"$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition --blind --nsub 1000 -o $TD/rust_T_vstnsub_$T.tsv >/dev/null 2>&1 || echo "vst nsub FAILED"
"$B" rlog -c $TD/counts.csv -m $TD/metadata.csv -d condition --blind -o $TD/rust_T_rlog_$T.tsv >/dev/null 2>&1 || echo "rlog FAILED"
