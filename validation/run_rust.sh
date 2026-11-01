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
run iterate $W --sf-type iterate
run weights $W --weights $D/weights/weights.csv
run weights_use_t $W --use-t --weights $D/weights_use_t/weights.csv
run weights_batch $W --covariate batch --weights $D/weights_batch/weights.csv
HD=$D/htseq
"$B" run --htseq $HD/sample_table.csv --htseq-dir $HD/files $W --covariate batch -o $HD/rust_$T.tsv > $HD/rust_$T.log 2>&1 || echo "htseq: FAILED"
"$B" vst --htseq $HD/sample_table.csv --htseq-dir $HD/files -d condition --blind -o $HD/rust_vst_$T.tsv > /dev/null 2>&1 || echo "htseq vst: FAILED"
"$B" normalize --htseq $HD/sample_table.csv --htseq-dir $HD/files -o $HD/rust_norm_$T.tsv > /dev/null 2>&1 || echo "htseq normalize: FAILED"
SD=$D/tximport_salmon
SALMON="--salmon $SD/sample_table.csv --salmon-dir $SD/files --tx2gene $SD/tx2gene.csv"
if [ -d "$SD" ]; then
  "$B" run $SALMON $W -o $SD/rust_$T.tsv > $SD/rust_$T.log 2>&1 || echo "tximport_salmon: FAILED"
  "$B" vst $SALMON -d condition --blind -o $SD/rust_vst_$T.tsv > /dev/null 2>&1 || echo "salmon vst: FAILED"
  "$B" vst $SALMON -d condition --blind --method local -o $SD/rust_vstlocal_$T.tsv > /dev/null 2>&1 || echo "salmon vst local: FAILED"
  "$B" vst $SALMON -d condition --blind --nsub 500 -o $SD/rust_fastvst_$T.tsv > /dev/null 2>&1 || echo "salmon vst nsub: FAILED"
  "$B" rlog $SALMON -d condition --blind -o $SD/rust_rlog_$T.tsv > /dev/null 2>&1 || echo "salmon rlog: FAILED"
  "$B" normalize $SALMON -o $SD/rust_norm_$T.tsv > /dev/null 2>&1 || echo "salmon normalize: FAILED"
  run tximport_matrix $W --avg-tx-length $D/tximport_matrix/avg_tx_length.csv
  run tximport_poscounts $W --sf-type poscounts --avg-tx-length $D/tximport_poscounts/avg_tx_length.csv
  run tximport_lstpm $W --tximport
fi
ND=$D/normfactors
if [ -d "$ND" ]; then
  NF="--norm-factors $ND/nf.csv"
  run normfactors $W $NF
  run normfactors_lrt -d condition --test lrt --reduced "~1" $NF
  run normfactors_apeglm $W --shrinkage --shrinkage-method apeglm $NF
  run normfactors_normal $W --shrinkage --shrinkage-method normal $NF
  run normfactors_local $W --fit-type local $NF
  run normfactors_weights $W --weights $D/normfactors_weights/weights.csv $NF
  run normfactors_batch $W --covariate batch $NF
  NI="-c $ND/counts.csv -m $ND/metadata.csv -d condition $NF"
  "$B" vst $NI --blind -o $ND/rust_vst_$T.tsv > /dev/null 2>&1 || echo "normfactors vst: FAILED"
  "$B" vst $NI --blind --method local -o $ND/rust_vstlocal_$T.tsv > /dev/null 2>&1 || echo "normfactors vst local: FAILED"
  "$B" vst $NI --blind --nsub 500 -o $ND/rust_fastvst_$T.tsv > /dev/null 2>&1 || echo "normfactors vst nsub: FAILED"
  "$B" rlog $NI --blind -o $ND/rust_rlog_$T.tsv > /dev/null 2>&1 || echo "normfactors rlog: FAILED"
  "$B" normalize -c $ND/counts.csv $NF -o $ND/rust_norm_$T.tsv > /dev/null 2>&1 || echo "normfactors normalize: FAILED"
fi
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
for ft in parametric local mean; do
  "$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition --method $ft --nsub 1000 --blind -o $TD/rust_fastvst${ft}_TRUE_$T.tsv >/dev/null 2>&1 || echo "fast vst $ft blind FAILED"
  "$B" vst -c $TD/counts.csv -m $TD/metadata.csv -d condition --method $ft --nsub 1000 -o $TD/rust_fastvst${ft}_FALSE_$T.tsv >/dev/null 2>&1 || echo "fast vst $ft FAILED"
done

# gen9.R: --quant from other quantifiers
quant() { local s=$1; shift; local QD=$D/$s; [ -d "$QD" ] || return; "$B" run --quant $QD/sample_table.csv --quant-dir $QD/files $W -o $QD/rust_$T.tsv "$@" > $QD/rust_$T.log 2>&1 || echo "$s: FAILED ($(tail -1 $QD/rust_$T.log))"; }
quant quant_kallisto --quant-type kallisto --tx2gene $D/quant_kallisto/tx2gene.csv
quant quant_rsem --quant-type rsem --tx2gene $D/quant_rsem/tx2gene.csv
quant quant_rsem_genes --quant-type rsem
quant quant_stringtie --quant-type stringtie --read-length 100 --tx2gene $D/quant_stringtie/tx2gene.csv
quant quant_piscem --quant-type piscem --tx2gene $D/quant_piscem/tx2gene.csv
quant quant_oarfish --quant-type oarfish --tx2gene $D/quant_oarfish/tx2gene.csv
# gen10.R: kallisto abundance.h5 (HDF5)
quant quant_kallisto_h5 --quant-type kallisto --tx2gene $D/quant_kallisto_h5/tx2gene.csv
# gen11.R: contrast choices (design ~ batch + age + condition)
C="-d condition --covariate batch --continuous age"
run name_age $C --name age
run name_batch $C --name batch_b2_vs_b1
run cvar_batch $C --contrast-var batch --numerator b1 --denominator b2
run numeric_CvsB $C --contrast-vector 0,0,0,-1,1
run list_CvsB $C --contrast-list "condition_C_vs_A;condition_B_vs_A"
run list_avg $C --contrast-list "condition_B_vs_A,condition_C_vs_A" --list-values 0.5,-0.5
run default $C
run numeric_useT $C --use-t --contrast-vector 0,0,0,-1,1
run name_lfcthr $C --name batch_b2_vs_b1 --lfc-threshold 0.5
run lrt_list $C --test lrt --reduced "~batch + age" --contrast-list "condition_C_vs_A;condition_B_vs_A"
run shrink_normal_batch $C --name batch_b2_vs_b1 --shrinkage --shrinkage-method normal
run shrink_apeglm_age $C --name age --shrinkage --shrinkage-method apeglm
# gen12.R: design formulas (--formula)
if [ -d "$D/inter_default" ]; then
  F="--formula ~batch+condition+batch:condition"
  run inter_default $F
  run inter_name $F --name batchb2.conditionB
  run inter_cvar $F --contrast-var condition --numerator C --denominator B
  run inter_lrt $F --test lrt --reduced "~ batch + condition"
  run inter_apeglm $F --name batchb2.conditionB --shrinkage --shrinkage-method apeglm
  ID=$D/inter_default
  "$B" vst -c $ID/counts.csv -m $ID/metadata.csv $F -o $ID/rust_vst_$T.tsv > /dev/null 2>&1 || echo "inter vst: FAILED"
  "$B" rlog -c $ID/counts.csv -m $ID/metadata.csv $F -o $ID/rust_rlog_$T.tsv > /dev/null 2>&1 || echo "inter rlog: FAILED"
  run nested_name --formula "~ condition + condition:batch" --name conditionB.batchb2
  G="--formula ~genotype*treatment --reference genotype=WT"
  run gt_default $G
  run gt_list $G --contrast-list "treatment_trt_vs_ctrl,genotypeKO.treatmenttrt"
  run gt_cvar_genotype $G --contrast-var genotype --numerator KO --denominator WT
  run gt_lrt $G --test lrt --reduced "~ genotype + treatment"
  X="--formula ~x+condition+x:condition --continuous x"
  run fxc_name $X --name x.conditionB
  run fxc_slope $X --name x
  run fxc_list $X --contrast-list "x,x.conditionB"
  run noint_default --formula "~ 0 + condition"
  run noint_BvsA --formula "~ 0 + condition" --contrast-var condition --numerator B --denominator A
  run noint_name --formula "~ 0 + condition" --name conditionB
  N="--formula ~0+batch+condition"
  run noint2_default $N
  run noint2_CvsB $N --contrast-var condition --numerator C --denominator B
  run noint2_AvsC $N --contrast-var condition --numerator A --denominator C
  run noint2_batch $N --contrast-var batch --numerator b2 --denominator b1
  run threeway_default --formula "~ batch * condition * genotype"
  run threeway_lrt --formula "~ batch * condition * genotype" --test lrt --reduced "~ (batch + condition + genotype)^2"
fi
# gen13.R: functions in design formulas (factor, relevel, ordered, log, scale, I, poly, ns, bs, offset)
if [ -d "$D/fdose_default" ]; then
  run fdose_default --formula "~ batch + factor(dose)" --continuous dose
  run fdose_name --formula "~ batch + factor(dose)" --continuous dose --name factor.dose.2
  run relevel_default --formula '~ batch + relevel(cond, ref = "B")'
  run relevel_fdose_lrt --formula '~ batch + relevel(factor(dose), "2")' --continuous dose --test lrt --reduced "~ batch"
  run logconc_name --formula "~ cond + log(conc)" --continuous conc --name log.conc.
  run scaleage_name --formula "~ cond + scale(age)" --continuous age --name scale.age.
  run isq_name --formula "~ cond + x + I(x^2)" --continuous x --name I.x.2.
  P="--formula ~cond+poly(x,2) --continuous x"
  run poly_name $P --name poly.x..2.2
  run shr_apeglm_poly $P --name poly.x..2.1 --shrinkage --shrinkage-method apeglm
  run poly_lrt $P --test lrt --reduced "~ cond"
  run polyraw_lrt --formula "~ cond + poly(x, 2, raw = TRUE)" --continuous x --test lrt --reduced "~ cond"
  S="--formula ~cond+ns(x,df=3) --continuous x"
  run ns_name $S --name ns.x..df...3.2
  ND=$D/ns_name
  "$B" vst -c $ND/counts.csv -m $ND/metadata.csv $S -o $ND/rust_vst_$T.tsv > /dev/null 2>&1 || echo "ns vst: FAILED"
  run ns_lrt $S --test lrt --reduced "~ cond"
  run bs_name --formula "~ cond + bs(x, df = 4)" --continuous x --name bs.x..df...4.3
  run bs_lrt --formula "~ cond + bs(x, df = 4)" --continuous x --test lrt --reduced "~ cond"
  run condpoly_name --formula "~ cond * poly(x, 2)" --continuous x --name condB.poly.x..2.1
  run condpoly_lrt --formula "~ cond * poly(x, 2)" --continuous x --test lrt --reduced "~ cond + poly(x, 2)"
  run condns_name --formula "~ cond + cond:ns(x, df = 2)" --continuous x --name condC.ns.x..df...2.2
  run ordered_default --formula "~ batch + ordered(cond)"
  run ordered_L --formula "~ batch + ordered(cond)" --name ordered.cond..L
  run rankdef_lrt $P --test lrt --reduced "~ x + poly(x, 2)"
  run rankdef0_lrt --formula "~ 0 + cond + poly(x, 3)" --continuous x --test lrt --reduced "~ 0 + cond + x + scale(x)"
  run offset_default --formula "~ cond + offset(conc)" --continuous conc
  run mknames_cvar --formula "~ batch + geno" --contrast-var geno --numerator KO-2 --denominator "wild type"
  run mknames_default --formula "~ batch + geno"
  run logical_name --formula "~ flag + cond" --name flagTRUE
  run logical_default --formula "~ flag + cond"
  run numeric_only --formula "~ x + scale(age)" --continuous x --continuous age
  run noint_fdose --formula "~ 0 + factor(dose)" --continuous dose
  run shr_normal_logconc --formula "~ log(conc) + cond" --continuous conc --name cond_C_vs_A --shrinkage --shrinkage-method normal
fi
