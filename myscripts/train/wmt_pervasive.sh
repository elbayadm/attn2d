#! /usr/bin/env bash
g=2
tok=300
freq=32

nl=14
emb=512
bottleneck=256
ffn=2048
maxlen=100
ker=11
div=4
lr=0.001
JOBNAME=n${nl}_k${ker}_emb${emb}_div${div}_btn${bottleneck}_ff${ffn}_gated_lr${lr}_bsz19K

GROUP=fairseq_pervasive/wmt15_deen/bidir
LOGS=$HOME/work/tensorboard_logs/$GROUP

# Prepare the experiment directories
MODELDIR=$HOME/work/checkpoints/$GROUP/$JOBNAME
echo MODELDIR: $MODELDIR
mkdir -p $MODELDIR  $LOGS

DATA=$HOME/data/wmt15_ende_binaries
src=de
trg=en

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
    --distributed-port ${PORT} --distributed-no-spawn --ddp-backend=no_c10d  
    --distributed-world-size \$g --user-dir $FAIR/examples/pervasive 
    --arch pervasive --optimizer adam -s de -t en 
    --save-dir $MODELDIR --tensorboard-logdir $LOGS/$JOBNAME --seed 1 
    --no-epoch-checkpoints --no-progress-bar --log-interval 10  --save-interval-updates 500
    --max-source-positions $maxlen --max-target-positions $maxlen 
    --skip-invalid-size-inputs-valid-test 
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 
    --max-tokens \$tok --update-freq \$freq --max-update 50000 
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' 
    --lr ${lr} --min-lr '1e-9' 
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --left-pad-source False  
    --skip-output-mapping --add-positional-embeddings  
    --prediction-dropout 0.2 --embeddings-dropout 0.2 --convolution-dropout 0.2 
    --num-layers ${nl} --kernel-size ${ker} --convnet resnet --aggregator gated-max --divide-channels ${div}
    --decoder-embed-dim ${emb} --encoder-embed-dim ${emb} --ffn-dim ${ffn} --bottleneck  ${bottleneck} --conv-groups $bottleneck 
    --share-all-embeddings --conv-bias "

echo $train_cmd >> $SCRIPT
# OAR 
EXCLUDE=$(cat ~/exclude.gpus)
append=" -t besteffort -t idempotent -p \"gpumem>12000 $EXCLUDE \""
cmd="oarsub -l \"host=1/gpuid=$g,walltime=100:0:0\" -n $JOBNAME $append -O  $MODELDIR/stdout -E $MODELDIR/stderr    'bash $SCRIPT'" 
echo Running $cmd
eval $cmd



