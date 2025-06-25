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
ITA_TYPE=scheduled_crossmodal_clip_loss

BASE_TAU=0.01
ALPHA=0.02

DESC=SCHEDULED_CLIP_${BASE_TAU}_${ALPHA}_${LR}_FIXED_0.5_0.5_CROSSMODAL_VITB16

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/$DESC/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --sim_based_loss_alpha $ALPHA \
    --temp $BASE_TAU \
    --clip_scheduled_loss_type fixed \
    --per_sample_temp_mapping adaptive_with_base \
    --clip_loss_weight 0.5 \
    --sim_loss_weight 0.5 \
    --image_encoder vit_base_patch16_224 \
    --image_res 224