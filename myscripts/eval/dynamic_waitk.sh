#!/bin/bash

FAIR=$HOME/work/fairseq-main
DATA=$HOME/data/iwslt14_deen_bpe10k_binaries

src=de
trg=en
SAVE=$1
bsz=1
CKP="_best"
SPLIT=test
BEAM=1
lenpen=1
th=$2
shft=1


cd $FAIR

CUDA=$(gpu_getIDs.sh)
CUDA=${CUDA// /,}


output=$SAVE/${SPLIT}${CKP}_bsz${bsz}_s${shft}_th${th}.res_bpe
rm $output
echo $HOST > $output

CUDA_VISIBLE_DEVICES=$CUDA python generate.py $DATA --gen-subset $SPLIT \
    --path $SAVE/checkpoint$CKP.pt -s $src -t $trg \
     --task dynamic_transformer_simultaneous_translation --policy dynamic \
     --model-overrides "{'max_source_positions': 1024, 'max_target_positions': 1024}" \
     --left-pad-source False --lenpen ${lenpen} --write-threshold ${th} --shift ${shft} \
     --max-source-positions 1024 --max-target-positions 1024 \
     --user-dir $FAIR/examples/simultaneous --remove-bpe \
     --max-sentences ${bsz} --beam $BEAM  2>&1 | tee -a $output



