#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --mem=480000

#SBATCH --time=01:59:59

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"


MODEL_PATHS=(
    /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_tau_0.01_lr_8e-4/checkpoint_best.pth
    /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_8e-4_quad_crossmodal/checkpoint_best.pth
    /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_2e-4_quad_crossmodal_and_unimodal_augmented/checkpoint_best.pth
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $(dirname "$MODEL_PATH"))
    DESC=KNN_EVAL_${MODEL_NAME}

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=7800 \
        --use_env clip.py \
        --run_name "$DESC" \
        --data cc3m \
        --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/${DESC}/ \
        --init_model \
        --use_amp \
        --ita_type clip \
        --checkpoint $MODEL_PATH \
        --knn_eval
done