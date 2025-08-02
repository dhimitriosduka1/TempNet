#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu20

#SBATCH --time=01:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATASETS=(cifar10 cifar100 imagenet)
MODEL_PATHS=(
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/clip_vit_b16_tau_0.01_lr_8e-4/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/BASELINE_CLIP_COS_0.01_0.05_VITB16/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/SCHEDULED_CLIP_0.01_0.04_2e-4_QUAD_CROSSMODAL_AND_UNIMODAL_AUGMENTED_VITB16/checkpoint_best.pth
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $(dirname "$MODEL_PATH"))
    for DATASET in "${DATASETS[@]}"; do
        DESC=ZS_${DATASET}_${MODEL_NAME}

        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7800 \
            --use_env clip.py \
            --run_name "$DESC" \
            --data cc3m \
            --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/${DESC}/ \
            --init_model \
            --use_amp \
            --zs_dataset $DATASET \
            --ita_type clip \
            --checkpoint $MODEL_PATH \
            --zsh_eval \
            --zs_datafolder /BS/databases23/imagenet/original/ \
            --image_encoder vit_base_patch16_224 \
            --image_res 224
    done
done