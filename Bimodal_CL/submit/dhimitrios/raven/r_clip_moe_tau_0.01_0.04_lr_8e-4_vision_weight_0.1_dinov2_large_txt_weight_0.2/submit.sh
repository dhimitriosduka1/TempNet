#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --mem=480000

#SBATCH --time=11:59:59
#SBATCH --array=1-2%1

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4
ITA_TYPE=clip_moe_vision_and_text

SIM_BASED_LOSS_ALPHA=0.04
I2I_LOSS_WEIGHT=0.1
T2T_LOSS_WEIGHT=0.2

DESC=R_CLIP_MoE_VISION_${SIM_BASED_LOSS_ALPHA}_LR_${LR}_VISION_LOSS_WEIGHT_${I2I_LOSS_WEIGHT}_DINOV2_LARGE_OFFLINE_TXT_WEIGHT_${T2T_LOSS_WEIGHT}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_moe_tau_0.01_0.04_lr_8e-4_vision_weight_0.1_dinov2_large_txt_weight_0.2 \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --sim_based_loss_alpha $SIM_BASED_LOSS_ALPHA \
    --enable_txt_expert \
    --enable_vision_expert \
    --i2i_loss_weight $I2I_LOSS_WEIGHT \
    --t2t_loss_weight $T2T_LOSS_WEIGHT \