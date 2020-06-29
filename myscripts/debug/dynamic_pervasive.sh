#!/bin/bash
FAIR=/home/melbayad/work/fairseq-main
EXP="sanity_dynamic_pa_wmt"
SAVE=/home/melbayad/work/checkpoints/$EXP
LOG=/home/melbayad/work/tensorboard_logs/$EXP
#DATA=$HOME/data/iwslt14_deen_bpe10k_binaries
DATA=$HOME/data/wmt15_ende_binaries

mkdir -p $SAVE  $LOG
cd $FAIR

# Optim:
maxlen=40
tok=300
freq=1
# IWSLT
eembed=256
ffn=1024
bottleneck=128
ker=11
nl=14
extra=" --share-decoder-input-output-embed --conv-bias "

# WMT:
embed=512
ffn=1024
bottleneck=128
nl=16
k=11
extra=" --add-conv-relu --share-all-embeddings"

CUDA=$(gpu_getIDs.sh)
CUDA=${CUDA// /,}


arch=simultrans_pervasive_oracle
agg=grid-max
crit=simultrans_dynamic_loss


CUDA_VISIBLE_DEVICES=$CUDA python train.py $DATA --distributed-port 28275 --distributed-world-size 1 --user-dir $FAIR/examples/pervasive --ddp-backend=no_c10d \
    -a ${arch} --optimizer adam --lr 0.0005 -s de -t en --task dynamic_pervasive_simultaneous_translation \
    --save-dir $SAVE --tensorboard-logdir $LOG --seed 1 --memory-efficient --no-epoch-checkpoints --no-progress-bar --log-interval 10 \
    --max-source-positions $maxlen --max-target-positions $maxlen --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --max-tokens $tok --update-freq $freq --max-update 50000 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --lr 0.002 --min-lr '1e-9' \
    --criterion $crit --label-smoothing 0.1 --left-pad-source False  \
    --skip-output-mapping --add-positional-embeddings  \
    --prediction-dropout 0.2 --embeddings-dropout 0.2 --convolution-dropout 0.2 \
    --num-layers ${nl} --kernel-size ${ker} --convnet resnet --aggregator $agg --unidirectional \
    --decoder-embed-dim ${embed} --encoder-embed-dim ${embed} --ffn-dim ${ffn} --bottleneck  ${bottleneck} --conv-groups $bottleneck \
    --control-oracle-penalty 1 --control-num-layers 6 --control-kernel-size $ker   \
    --control-write-right --control-gate-dropout 0.1  --control-embeddings-dropout 0.1 --control-convolution-dropout 0.1 \
    --control-add-positional-embeddings --control-embed-dim $embed \
    --copy-embeddings --copy-network  --control-remove-writer-dropout $extra \
    --pretrained /scratch/zeus/melbayad/work/checkpoints/attn2d/wmt15_deen/baselines/unidir/seed1/checkpoint_best.pt


