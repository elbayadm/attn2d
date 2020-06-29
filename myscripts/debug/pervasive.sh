#!/bin/bash
FAIR=/home/melbayad/work/fairseq-main
EXP="new_code_grid"
SAVE=/home/melbayad/work/checkpoints/$EXP
LOG=/home/melbayad/work/tensorboard_logs/$EXP
DATA=$HOME/data/iwslt14_deen_bpe10k_binaries

mkdir -p $SAVE  $LOG
cd $FAIR
maxlen=50
tok=300
freq=20
   
CUDA=$(gpu_getIDs.sh)
CUDA=${CUDA// /,}


crit=label_smoothed_cross_entropy
arch=pervasive
#agg=grid-max

#arch=waitk-pervasive
agg=path-max

#arch=grid-pervasive
#agg=grid-max
#crit=grid_cross_entropy


CUDA_VISIBLE_DEVICES=$CUDA python train.py $DATA --distributed-port 28275 --distributed-world-size 1 --user-dir $FAIR/examples/pervasive \
    -a ${arch} --optimizer adam --lr 0.0005 -s de -t en \
    --save-dir $SAVE --tensorboard-logdir $LOG --seed 1 --memory-efficient --no-epoch-checkpoints --no-progress-bar --log-interval 10 \
    --max-source-positions $maxlen --max-target-positions $maxlen --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --max-tokens $tok --update-freq $freq --max-update 50000 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --lr 0.002 --min-lr '1e-9' \
    --prediction-dropout 0.2  --embeddings-dropout 0.2 --convolution-dropout 0.2 \
    --criterion $crit --label-smoothing 0.1 --left-pad-source False \
    --num-layers 14 --kernel-size 11 --convnet resnet --aggregator $agg \
    --decoder-embed-dim 256 --encoder-embed-dim 256 --ffn-dim 1024 --bottleneck 256 --output-dim 256 --conv-groups 256 --conv-bias \
    --skip-output-mapping --add-positional-embeddings --share-decoder-input-output-embed 


