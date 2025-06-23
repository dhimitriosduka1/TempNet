#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=05:59:00
#SBATCH -a 1-8%1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=2e-4
ITA_TYPE=scheduled_crossmodal_with_augmentations_and_unimodal_clip_loss

BASE_TAU=0.01
ALPHA=0.04

DESC=SCHEDULED_CLIP_${BASE_TAU}_${ALPHA}_${LR}_QUAD_CROSSMODAL_AND_UNIMODAL_AUGMENTED_METADATA

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
    --clip_scheduled_loss_type quadratic \
    --enable_i2i_loss \
    --enable_t2t_loss \
    --cc3m_extended_captions_path /BS/dduka/work/databases/cc3m/train/captions_extended_llm.json \