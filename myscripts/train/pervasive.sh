#! /usr/bin/env bash
freq=12
tok=354
maxup=50000

g=2
N=14
seed=1
emb=256
reduce=256

k=11
dp=0.2
lro=0.002
warm=4000
wd=0.0001
maxlen=100
lower=0

DATA=$HOME/data/iwslt14_deen_bpe10k_binaries
#DATA=$HOME/data/iwslt14_deen_bpe14k_binaries
#DATA=$HOME/data/iwslt15_envi_binaries

src=de
trg=en
src=en
trg=de
#src=en
#trg=vi
#src=vi
#trg=en

GROUP=attn2d/iwslt_${src}${trg}/grid_ablation_feb27_10k

LOGS=$HOME/work/tensorboard_logs/$GROUP

network="resnet_addup_nonorm2"
agg=max
#agg=gated_max
#agg=attn

SHORTNAME=$src${trg}_pa_gridv4_agg${agg}_l${lower}
JOBNAME=gridv4_agg${agg}_${network}_maxlen${maxlen}_n${N}_ker${k}_dr${dp}_l${lower}

lrsettings=" --lr-scheduler inverse_sqrt --warmup-updates ${warm} --warmup-init-lr '1e-07' --lr ${lro} "
JOBNAME=${JOBNAME}_rsqrt_lr${lro}_warm${warm}
JOBNAME=${JOBNAME}_tok${tok}_freq${freq}_g${g}

# Prepare the experiment directories
MODELDIR=$HOME/work/checkpoints/$GROUP/$JOBNAME/seed${seed}
echo MODELDIR: $MODELDIR
mkdir -p $MODELDIR 

echo LOGS $LOGS
mkdir -p $LOGS

FAIR=$HOME/work/fairseq_source/fairseq-py-attn2d-oar


SCRIPT=$MODELDIR/script


echo "#!/usr/bin/env zsh " > $SCRIPT
echo "tok=$tok" >> $SCRIPT
echo "freq=$freq" >> $SCRIPT
echo "g=$g" >> $SCRIPT
echo "cd $FAIR" >> $SCRIPT
#echo "conda activate pytorch101_cuda92_cudnn76_nccl135" >> $SCRIPT
echo "source activate pytorch120" >> $SCRIPT

PORT=$(shuf -i 20000-30000 -n 1)
train_cmd="CUDA_VISIBLE_DEVICES=\$(gpu_getIDs.sh) python train.py $DATA --distributed-port ${PORT} --distributed-world-size \$g 
    -s $src -t $trg --ddp-backend=no_c10d  --left-pad-source False
    --max-source-positions ${maxlen} --max-target-positions ${maxlen} --skip-invalid-size-inputs-valid-test 
    --save-dir $MODELDIR 
    --tensorboard-log-dir $LOGS/s${seed}.$JOBNAME --seed ${seed}
   --no-epoch-checkpoints --no-progress-bar 
    -a attn2d_waitk_v2 
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay ${wd}
    --max-tokens \$tok --update-freq \$freq --max-update $maxup
    ${lrsettings} --min-lr '1e-9' --maintain-resolution 1 
    --prediction-dropout ${dp} --input-dropout 0 --embeddings-dropout ${dp}  --convolution-dropout ${dp}
    --criterion pa_grid_cross_entropy_v2 --label-smoothing 0.1 
    --decoder-embed-dim 256 --encoder-embed-dim 256 --ffn-dim 1024  
    --num-layers $N --kernel-size $k --network ${network} --aggregation ${agg} --lower-diag ${lower} 
    --add-positional-embeddings  --skip-output-mapping --share-decoder-input-output-embed --conv-bias
    "

echo $train_cmd >> $SCRIPT
# OAR 
EXCLUDE=$(cat ~/exclude.gpus)

#append="-t besteffort -t idempotent -p \"gpumem>24000\""
append=" -t besteffort -t idempotent -p \"gpumem>11200  \""
cmd="oarsub -l \"host=1/gpuid=$g,walltime=100:0:0\" -n $SHORTNAME $append -O  $MODELDIR/stdout -E $MODELDIR/stderr    'bash $SCRIPT'" 
echo Running $cmd
eval $cmd



