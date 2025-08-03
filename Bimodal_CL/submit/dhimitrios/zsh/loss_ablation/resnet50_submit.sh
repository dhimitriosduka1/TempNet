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
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_tau_0.01_lr_8e-4/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_sim_clip_0.01_0.05_lr_8e-4_for_the_paper/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_8e-4_fixed_0.5_0.5_crossmodal/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_8e-4_quad_crossmodal/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_2e-4_fixed_0.5_0.5_crossmodal_and_unimodal_augmented/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_2e-4_quad_crossmodal_and_unimodal_augmented/checkpoint_best.pth
)

# First check if all paths exist
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model path $MODEL_PATH does not exist"
        exit 1
    fi
done

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $(dirname "$MODEL_PATH"))
    for DATASET in "${DATASETS[@]}"; do
        DESC=ZS_${DATASET}_${MODEL_NAME}_RESNET50_FOR_THE_PAPER

        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7800 \
            --use_env clip.py \
            --run_name "$DESC" \
            --data cc3m \
            --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/${DESC}_RESNET50_FOR_THE_PAPER/ \
            --init_model \
            --use_amp \
            --zs_dataset $DATASET \
            --ita_type clip \
            --checkpoint $MODEL_PATH \
            --zsh_eval
    done
done