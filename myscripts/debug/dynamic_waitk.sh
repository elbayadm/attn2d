#!/bin/bash
FAIR=/home/melbayad/work/fairseq-main
penal=0
kmin=-5
kmax=10
closs=1
wloss=1
EXP="dynamic_waitk_w${wloss}_c${closs}_penal${penal}_k${kmin}_k${kmax}_2506"
SAVE=/home/melbayad/work/checkpoints/$EXP
LOG=/home/melbayad/work/tensorboard_logs/$EXP
DATA=$HOME/data/iwslt14_deen_bpe10k_binaries

mkdir -p $SAVE  $LOG

echo Saving to $SAVE

cd $FAIR
maxlen=30
tok=1000
freq=8
   
CUDA=$(gpu_getIDs.sh)
CUDA=${CUDA// /,}


crit=simultrans_dynamic_loss
arch=dynamic_simultaneous_transformer_small



#lrs= "--lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr '1e-07' --lr 0.001 --min-lr 1e-9 "
lrs="--lr-scheduler fixed --lr 0.0005 --min-lr 1e-9 "
echo Using GPU $CUDA
CUDA_VISIBLE_DEVICES=$CUDA python train.py $DATA --distributed-port 28275 --distributed-world-size 1 --user-dir $FAIR/examples/simultaneous \
    -a ${arch} --optimizer adam -s de -t en \
    --task dynamic_transformer_simultaneous_translation --control-remove-writer-dropout --control-kmax ${kmax} --control-kmin ${kmin} \
    --save-dir $SAVE --tensorboard-logdir $LOG --seed 1  --no-epoch-checkpoints --no-progress-bar --log-interval 1 --save-interval-updates 100 \
    --max-source-positions $maxlen --max-target-positions $maxlen --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --max-tokens $tok --update-freq $freq --max-update 5000 \
    $lrs \
    --criterion $crit --label-smoothing 0.1 --left-pad-source False --control-scale ${closs} --write-scale ${wloss} \
    --share-decoder-input-output-embed  --control-gate-dropout 0.2 --control-oracle-penalty $penal \
    --pretrained /scratch/zeus/melbayad/work/checkpoints/transformer_waitk_iwslt14_deen_embed256_sample_waitk/tied_sweepv2_waitk1to300_dp0.3_wd0.0001_lbs0.1_lr0.0005_w4000_tok4000_freq2_g1/seed1/checkpoint_best.pt

#--copy-embeddings --copy-network 

