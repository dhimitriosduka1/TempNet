#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu22

#SBATCH --time=11:59:00
#SBATCH -a 1-4%1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=2e-4
TAU_MIN=0.01
TAU_MAX=0.1

LOSS_WEIGHT=0.2

DESC=CLIP_COS_${TAU_MIN}_${TAU_MAX}_LR_${LR}_T2T_${LOSS_WEIGHT}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $DESC \
    --data $DATA \
    --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/clip_cos_0.01_0.1_lr_2e-4_t2t_0.2/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type clip \
    --temperature_scheduler cos \
    --tau_min $TAU_MIN \
    --tau_max $TAU_MAX \
    --enable_t2t_loss \
    --t2t_loss_weight $LOSS_WEIGHT \
    --cc3m_extended_captions_path /BS/dduka/work/databases/cc3m/train/captions_extended_llm.json \