#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=05:59:00
#SBATCH -a 1-4%1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4
TAU_MIN=0.01
TAU_MAX=0.05

DESC=BASELINE_CLIP_COS_${TAU_MIN}_${TAU_MAX}_VITB16

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/$DESC/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type clip \
    --temperature_scheduler cos \
    --tau_min $TAU_MIN \
    --tau_max $TAU_MAX \
    --image_encoder vit_base_patch16_224 \
    --image_res 224 \