#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --mem=480000

#SBATCH --time=11:59:59
#SBATCH --array=1-3%1

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=2e-4

TAU_MIN=0.01
TAU_MAX=0.05

I2I_LOSS_WEIGHT=0.5
T2T_LOSS_WEIGHT=0.2

ITA_TYPE=sogclr_with_cosine_and_unimodal_loss

DESC=R_SOGCLR_COS_${TAU_MIN}_${TAU_MAX}_${LR}_I2I_${I2I_LOSS_WEIGHT}_T2T_${T2T_LOSS_WEIGHT}_UNIMODAL_AUGMENTED

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_sogclr_cos_0.01_0.05_lr_2e-4_i2i_0.5_t2t_0.2_unimodal_augmented/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --temperature_scheduler cos \
    --temp $TAU_MIN \
    --tau_min $TAU_MIN \
    --tau_max $TAU_MAX \
    --enable_i2i_loss \
    --enable_t2t_loss \
    --i2i_loss_weight $I2I_LOSS_WEIGHT \
    --t2t_loss_weight $T2T_LOSS_WEIGHT \
    --cc3m_extended_captions_path /ptmp/dduka/work/data/cc3m/training/captions_extended_llm.json \