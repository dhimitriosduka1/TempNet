#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu24

#SBATCH --time=01:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4
ITA_TYPE=clip

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4820 \
    --use_env clip.py \
    --run_name BASELINE_CLIP_RESUMED \
    --data $DATA \
    --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/clip_tau_0.01_lr_8e-4_Dasdasdsdsadas \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --checkpoint /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_2e-4_quad_crossmodal_and_unimodal_augmented/checkpoint_best.pth \
    --sim_threshold 0.1729 \
    --compute_temperature_assignments