#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --mem=480000

#SBATCH --time=11:59:59
#SBATCH --array=1-6%1

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4

BASE_TAU=0.01
ALPHA=0.02

GAMMA=0.8

ITA_TYPE=scheduled_sogclr_crossmodal

DESC=R_SCHEDULED_SogCLR_GAMMA_${GAMMA}_${BASE_TAU}_${ALPHA}_${LR}_QUAD_CROSSMODAL

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4821 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_sogclr_0.01_0.02_lr_8e-4_quad_crossmodal/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --sogclr_gamma $GAMMA \
    --clip_scheduled_loss_type quadratic \
    --per_sample_temp_mapping adaptive_with_base \