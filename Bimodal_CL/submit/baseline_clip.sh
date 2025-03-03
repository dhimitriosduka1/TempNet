#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu24

#SBATCH --time=00:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

data_path=.
data=cc3m
lr=8e-4
frac=1.0
desc=clip

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name CLIP_BASELINE \
    --data_path . \
    --data cc3m \
    --output_dir output/clip_cc3m_clip \
    --init_model \
    --use_amp \
    --epochs 30 --lr 8e-4 \
    --lr_temp_net 3e-5 \
    --train_frac 1.0 \
    --zs_dataset cifar10 \
    --ita_type clip