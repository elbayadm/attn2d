#!/bin/bash
FAIR=/home/melbayad/work/fairseq-main
DATA=$HOME/data/iwslt15_envi_binaries

cd $FAIR
maxlen=50
tok=500
freq=8
   
CUDA=$(gpu_getIDs.sh)
CUDA=${CUDA// /,}


crit=simultrans_dynamic_loss
arch=branch_dynamic_simultaneous_transformer_small
kmin=-5
kmax=10
closs=1
wloss=1
clayers=2
glayers=2
special=" --copy-embeddings --copy-network --restrain-writer "
#--control-remove-writer-dropout
#--restrain-writer
#--observe-encoder

EXP="envi_restrain_branch_dynamic_wl${wloss}_cl${closs}_k${kmin}_${kmax}_clyr${clayers}_glyr${glayers}_copyemb+net"
SAVE=/home/melbayad/work/checkpoints/$EXP
LOG=/home/melbayad/work/tensorboard_logs/$EXP

mkdir -p $SAVE  $LOG
echo Saving to $SAVE

#lrs= "--lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr '1e-07' --lr 0.001 --min-lr 1e-9 "
lrs="--lr-scheduler fixed --lr 0.0005 --min-lr 1e-9 "

CUDA_VISIBLE_DEVICES=$CUDA python train.py $DATA --distributed-port 28275 --distributed-world-size 1 --user-dir $FAIR/examples/simultaneous \
    -a ${arch} --optimizer adam -s en -t vi $special \
    --task dynamic_transformer_simultaneous_translation --control-kmax ${kmax} --control-kmin ${kmin} \
    --save-dir $SAVE --tensorboard-logdir $LOG --seed 1  --no-epoch-checkpoints --no-progress-bar --log-interval 1 --save-interval-updates 100 \
    --max-source-positions $maxlen --max-target-positions $maxlen --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --max-tokens $tok --update-freq $freq --max-update 50000  $lrs \
    --criterion $crit --label-smoothing 0.1 --left-pad-source False --control-scale ${closs} --write-scale $wloss  \
    --share-decoder-input-output-embed  --control-gate-dropout 0.2 --control-num-encoder-layers ${clayers} --control-num-decoder-layers ${clayers} --control-gate-layers $glayers \
    --pretrained /home/melbayad/work/checkpoints/transformer_sample_waitk_iwslt_envi/iwslt_envi_sampleK/seed1/checkpoint_best.pt



