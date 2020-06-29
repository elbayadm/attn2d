#! /usr/bin/env bash
DATA=$HOME/data/wmt15_ende_binaries

# Optim:
g=2
maxlen=40

#tok=600
#freq=43

tok=310
freq=83

# WMT:
embed=512
ffn=1024
bottleneck=128
nl=16
cnl=8
ker=11
penal=0
cs=0.5
ws=1
extra=" --add-conv-relu --share-all-embeddings"

GROUP=fairseq_pervasive/wmt15_deen/dynamic_unidir
LOGS=$HOME/work/tensorboard_logs/$GROUP

JOBNAME=dynamic_wl${ws}_cl${cs}_penal${penal}_eval_mode_copy_n${cnl}


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
    --distributed-port ${PORT} --distributed-no-spawn --ddp-backend=no_c10d  
    --distributed-world-size \$g --user-dir $FAIR/examples/pervasive 
    --arch simultrans_pervasive_oracle --optimizer adam --lr 0.0005 -s de -t en 
    --task dynamic_pervasive_simultaneous_translation 
    --save-dir $MODELDIR --tensorboard-logdir $LOGS/$JOBNAME --seed 1 
    --memory-efficient --no-epoch-checkpoints --no-progress-bar --log-interval 10  --save-interval-updates 300
    --max-source-positions $maxlen --max-target-positions $maxlen 
    --skip-invalid-size-inputs-valid-test 
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 
    --max-tokens \$tok --update-freq \$freq --max-update 50000 
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' 
    --lr 0.002 --min-lr '1e-9' 
    --criterion simultrans_dynamic_loss --label-smoothing 0.1 --left-pad-source False  --control-scale ${cs} --write-scale ${ws}
    --skip-output-mapping --add-positional-embeddings  --unidirectional
    --prediction-dropout 0.2 --embeddings-dropout 0.2 --convolution-dropout 0.2 
    --num-layers ${nl} --kernel-size ${ker} --convnet resnet --aggregator grid-max
    --decoder-embed-dim ${embed} --encoder-embed-dim ${embed} --ffn-dim ${ffn} --bottleneck  ${bottleneck} --conv-groups $bottleneck 
    --control-oracle-penalty ${penal} --control-num-layers $cnl --control-kernel-size $ker  
    --control-write-right --control-gate-dropout 0.1  --control-embeddings-dropout 0.1 
    --control-convolution-dropout 0.1 --control-add-positional-embeddings 
    --control-embed-dim $embed 
    --copy-embeddings --copy-network  --control-remove-writer-dropout $extra 
    --pretrained /scratch/zeus/melbayad/work/checkpoints/attn2d/wmt15_deen/baselines/unidir/seed1/checkpoint_best.pt
        "

echo $train_cmd >> $SCRIPT
# OAR 
EXCLUDE=$(cat ~/exclude.gpus)
#append="-p \"gpumem>20000\""
#append=" -t besteffort -t idempotent -p \"gpumem>12000  \""
cmd="oarsub -l \"host=1/gpuid=$g,walltime=100:0:0\" -n $JOBNAME $append -O  $MODELDIR/stdout -E $MODELDIR/stderr    'bash $SCRIPT'" 
echo Running $cmd
eval $cmd



