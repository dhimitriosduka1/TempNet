#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu20

#SBATCH --time=11:59:00
#SBATCH -a 1-8%1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA_PATH=.
DATA=cc3m
LR=8e-4

GAMMA=0.8

DESC=SOGCLR_BASELINE

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4821 \
    --use_env clip.py \
    --run_name $DESC \
    --data_path $DATA_PATH \
    --data $DATA \
    --output_dir /BS/dduka/work/projects/TempNet/Bimodal_CL/submit/dhimitrios/sogclr_baseline_copy/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type sogclr \
    --sogclr_gamma $GAMMA \