#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err
#SBATCH --job-name bcl
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --time=01:59:59

module purge
module load anaconda/3/2023.03
conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

# CLIP Baseline
# Cos Baseline
# M-I2I
# M-T2T
# M-I2T
# M-T2I
# Text Expert
# Vision Expert
# Vision and Text Expert
# CLIP Baseline with I2I
# CLIP Baseline with T2T 
# CLIP Baseline with I2I and T2T
# TeMo Baseline

# /BS/dduka/work/projects/TempNet/Bimodal_CL/submit/dhimitrios/clip_cos_0.01_0.05_lr_8e-4/checkpoint_best.pth # Cos Baseline Not Yet Submitted

DATASETS=(cifar10 cifar100)
MODEL_PATHS=(
    /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_moe_tau_0.01_0.04_lr_8e-4_txt_weight_0.4/checkpoint_best.pth
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $(dirname "$MODEL_PATH"))
    for DATASET in "${DATASETS[@]}"; do
        DESC=ZS_${DATASET}_${MODEL_NAME}

        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7800 \
            --use_env clip.py \
            --run_name "$DESC" \
            --data cc3m \
            --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/$DESC/ \
            --init_model \
            --use_amp \
            --zs_dataset $DATASET \
            --ita_type clip \
            --checkpoint $MODEL_PATH \
            --zsh_eval
    done
done