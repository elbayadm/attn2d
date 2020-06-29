#!/bin/bash

FAIR=$HOME/work/fairseq-main
DATA=$HOME/data/iwslt14_deen_bpe10k_binaries

src=de
trg=en
SAVE=$1
tol=$2
bsz=1
CKP="_best"
SPLIT=test
BEAM=1
lenpen=1
shft=1
kmin=-5
kmax=5


cd $FAIR

CUDA=$(gpu_getIDs.sh)
CUDA=${CUDA// /,}


output=$SAVE/${SPLIT}${CKP}_bsz${bsz}_oracle_penal${tol}_k${kmin}_${kmax}.res_bpe_largestK_newmask
rm $output
echo $HOST > $output

CUDA_VISIBLE_DEVICES=$CUDA python generate.py $DATA --gen-subset $SPLIT \
    --path $SAVE/checkpoint$CKP.pt -s $src -t $trg \
     --task dynamic_transformer_simultaneous_translation --policy oracle --path-oracle-tol $tol --oracle-kmin $kmin --oracle-kmax $kmax\
     --model-overrides "{'max_source_positions': 1024, 'max_target_positions': 1024, 'arch': 'waitk_transformer'}" \
     --left-pad-source False --lenpen ${lenpen}  \
     --max-source-positions 1024 --max-target-positions 1024 \
     --user-dir $FAIR/examples/simultaneous --remove-bpe \
     --max-sentences ${bsz} --beam $BEAM  2>&1 | tee -a $output



