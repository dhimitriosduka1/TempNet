#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=23:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA_PATH=.
DATA=cc3m
LR=8e-4
ITA_TYPE=clip

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name BASELINE_CLIP_RESUMED \
    --data_path $DATA_PATH \
    --data $DATA \
    --output_dir /BS/dduka/work/data/bimodal_cl/dhimitrios/clip_tau_0.01_lr_8e-4 \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --checkpoint /BS/dduka/work/projects/TempNet/Bimodal_CL/output/baseline_clip/2025-03-10_23-46/checkpoint_best.pth \