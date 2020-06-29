#!/bin/bash
FAIR=/home/melbayad/work/fairseq-main
EXP="branch_dynamic_waitk_w1_c1_penal0_wodp_kmin-1_kmax10_nocopy_n2n2_2306"
SAVE=/home/melbayad/work/checkpoints/$EXP
LOG=/home/melbayad/work/tensorboard_logs/$EXP
DATA=$HOME/data/iwslt14_deen_bpe10k_binaries

mkdir -p $SAVE  $LOG
cd $FAIR
# Optim:
g=2
tok=500
freq=32
maxlen=50
kmin=-1
kmax=10
cl=1
wl=1

lrs="--max-update 50000 --lr-scheduler inverse_sqrt 
--warmup-updates 4000 --warmup-init-lr '1e-07' 
--lr 0.0005 --min-lr '1e-9'"
extra=""
#extra=" --copy-embeddings --copy-network "

GROUP=dynamic_tf/iwslt_deen
LOGS=$HOME/work/tensorboard_logs/$GROUP
JOBNAME=pretrain_multi_k${kmin}_${kmax}_lrs1_p${penal}_w${wl}_c${cl}_tok${tok}_f${freq}_g${g}_nocopy

# Prepare the experiment directories
MODELDIR=$HOME/work/checkpoints/$GROUP/$JOBNAME/

echo MODELDIR: $MODELDIR
mkdir -p $MODELDIR $LOGS

FAIR=$HOME/work/fairseq-main
SCRIPT=$MODELDIR/script

echo "#!/usr/bin/env zsh " > $SCRIPT
echo "tok=$tok" >> $SCRIPT
echo "freq=$freq" >> $SCRIPT
echo "g=$g" >> $SCRIPT
echo "CUDA=\$(gpu_getIDs.sh)" >> $SCRIPT  
echo "echo Parsed Cuda devices \$CUDA" >> $SCRIPT
echo "CUDA=\${CUDA// /,}" >> $SCRIPT
echo "echo Cuda devices \$CUDA" >> $SCRIPT

echo "cd $FAIR" >> $SCRIPT
echo "source activate pytorch120-simul" >> $SCRIPT

PORT=$(shuf -i 20000-30000 -n 1)
train_cmd="CUDA_VISIBLE_DEVICES=\$CUDA python train.py $DATA 
    --distributed-port ${PORT} --distributed-no-spawn 
    --ddp-backend=no_c10d  --distributed-world-size \$g 
    --user-dir $FAIR/examples/simultaneous 
    --max-tokens \$tok --update-freq \$freq 
    -a branch_dynamic_simultaneous_transformer_small --optimizer adam -s de -t en 
    --task dynamic_transformer_simultaneous_translation --control-remove-writer-dropout 
    --control-kmax $kmax --control-kmin $kmin 
    --save-dir $MODELDIR --tensorboard-logdir $LOGS/$JOBNAME
    --seed 1  --no-epoch-checkpoints --no-progress-bar --log-interval 10
    --save-interval-updates 100 --max-source-positions $maxlen 
    --max-target-positions $maxlen --skip-invalid-size-inputs-valid-test 
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 
    $lrs
    --criterion simultrans_dynamic_loss --label-smoothing 0.1 
    --left-pad-source False 
    --control-scale $cl --write-scale $wl --share-decoder-input-output-embed  
    --control-gate-dropout 0.2 --control-num-encoder-layers 2 --control-num-decoder-layers 2 $extra
    --pretrained /scratch/zeus/melbayad/work/checkpoints/transformer_waitk_iwslt14_deen_embed256_sample_waitk/tied_sweepv2_waitk5to300_dp0.3_wd0.0001_lbs0.1_lr0.0005_w4000_tok4000_freq2_g1/seed1/checkpoint_best.pt"

echo $train_cmd >> $SCRIPT
# OAR 
EXCLUDE=$(cat ~/exclude.gpus)
#append="-p \"gpumem>11500\""
append=" -t besteffort -t idempotent -p \"gpumem>12000  \""
cmd="oarsub -l \"host=1/gpuid=$g,walltime=100:0:0\" -n $JOBNAME $append -O  $MODELDIR/stdout -E $MODELDIR/stderr    'bash $SCRIPT'" 
echo Running $cmd
eval $cmd



