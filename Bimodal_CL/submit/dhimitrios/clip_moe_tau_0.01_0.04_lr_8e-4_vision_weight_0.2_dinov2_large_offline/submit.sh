#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=11:59:00
#SBATCH -a 1-5%1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4
ITA_TYPE=clip_moe_vision

SIM_BASED_LOSS_ALPHA=0.04
I2I_LOSS_WEIGHT=0.2

DESC=CLIP_MoE_VISION_${SIM_BASED_LOSS_ALPHA}_LR_${LR}_LOSS_WEIGHT_${I2I_LOSS_WEIGHT}_DINOV2_LARGE_OFFLINE

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/clip_moe_tau_0.01_0.04_lr_8e-4_vision_weight_0.2_dinov2_large_offline \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --enable_vision_expert \
    --i2i_loss_weight $I2I_LOSS_WEIGHT \
    --sim_based_loss_alpha $SIM_BASED_LOSS_ALPHA \