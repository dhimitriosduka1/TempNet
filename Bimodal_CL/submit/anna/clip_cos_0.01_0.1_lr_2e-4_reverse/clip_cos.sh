#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=23:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

set -ex

#mkdir /BS/kukleva3/work/logs/diff_mot
export MAMBA_EXE='/BS/kukleva3/work/bin/micromamba';
export MAMBA_ROOT_PREFIX='/BS/kukleva3/work/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup

micromamba activate bimodal_cl

DATA_PATH=.
DATA=cc3m
LR=2e-4
TAU_MIN=0.01
TAU_MAX=0.1

DESC=BASELINE_CLIP_COS_${TAU_MIN}_${TAU_MAX}_LR_${LR}_REV

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env /BS/dduka/work/projects/TempNet/Bimodal_CL/clip.py \
    --run_name $DESC \
    --data_path $DATA_PATH \
    --data $DATA \
    --output_dir /BS/dduka/work/projects/TempNet/Bimodal_CL/submit/anna/clip_cos_0.01_0.1_lr_2e-4_reverse \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type clip \
    --temperature_scheduler cos \
    --tau_min $TAU_MIN \
    --tau_max $TAU_MAX \
    --offset 3.14159265358979323846
