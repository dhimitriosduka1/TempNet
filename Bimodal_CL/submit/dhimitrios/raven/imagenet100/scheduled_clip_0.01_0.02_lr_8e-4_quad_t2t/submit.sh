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

DATA=imagenet100
LR=8e-4
ITA_TYPE=scheduled_clip_loss

BASE_TAU=0.01
ALPHA=0.02

DESC=R_SCHEDULED_CLIP_${BASE_TAU}_${ALPHA}_${LR}_QUAD_T2T_IMAGENET100

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/$DESC/ \
    --init_model \
    --use_amp \
    --epochs 200 --lr $LR \
    --ita_type $ITA_TYPE \
    --sim_based_loss_alpha $ALPHA \
    --temp $BASE_TAU \
    --clip_scheduled_loss_type quadratic \
    --per_sample_temp_similarity t2t \
    --per_sample_temp_mapping adaptive_with_base \
    --train_image_root /ptmp/dduka/work/data/imagenet100/ \