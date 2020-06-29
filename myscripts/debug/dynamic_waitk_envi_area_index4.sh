#!/bin/bash
FAIR=/home/melbayad/work/fairseq-main
penal=0
kmin=-5
kmax=10
closs=1
wloss=1
layers=2
EXP="id4_dynamic_envi_gelu_layers${layers}_detach_encdec_w${wloss}_c${closs}_penal${penal}_k${kmin}_k${kmax}_2506"
special=" --control-gate-layers ${layers} --control-detach --observe-encoder --control-nonlin gelu --observation-index 4"
#--control-remove-writer-dropout 
SAVE=/home/melbayad/work/checkpoints/$EXP
LOG=/home/melbayad/work/tensorboard_logs/$EXP
DATA=$HOME/data/iwslt15_envi_binaries
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
    -a ${arch} --optimizer adam -s en -t vi \
    --task dynamic_transformer_simultaneous_translation --control-kmax ${kmax} --control-kmin ${kmin} \
    --save-dir $SAVE --tensorboard-logdir $LOG --seed 1  --no-epoch-checkpoints --no-progress-bar --log-interval 1 --save-interval-updates 100 \
    --max-source-positions $maxlen --max-target-positions $maxlen --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --max-tokens $tok --update-freq $freq --max-update 5000 \
    $lrs  $special \
    --criterion $crit --label-smoothing 0.1 --left-pad-source False --control-scale ${closs} --write-scale ${wloss} \
    --share-decoder-input-output-embed  --control-gate-dropout 0.2 --control-oracle-penalty $penal \
    --pretrained /home/melbayad/work/checkpoints/transformer_sample_waitk_iwslt_envi/iwslt_envi_sampleK/seed1/checkpoint_best.pt

#--copy-embeddings --copy-network 

