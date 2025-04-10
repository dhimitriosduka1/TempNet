#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=11:59:00
#SBATCH -a 1-4%1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

# ============================ BEGIN OF HEADER ============================

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

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

OUTPUT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL/submit/anna/clip_tau_0.01_lr_8e-4_extended/"

MASTER_PORT=$((12000 + $RANDOM % 20000))
# ============================ END OF HEADER =============================

DATA=cc3m
LR=8e-4
ITA_TYPE=clip

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT \
    --use_env clip.py \
    --run_name BASELINE_CLIP_EXTENDED \
    --data $DATA \
    --output_dir $OUTPUT_DIR \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --cc3m_extended_captions_path /BS/dduka/work/databases/cc3m/train/captions_extended_llm.json \