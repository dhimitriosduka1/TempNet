#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition gpu20

#SBATCH --time=35:59:00

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

tau_min=0.01
tau_max=0.05

pct_tau_min=0.01
pct_tau_max=0.05

run_name=CLIP_COS_${tau_min}_${tau_max}_${pct_tau_min}_${pct_tau_max}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name $run_name \
    --data_path $data_path \
    --data $data \
    --output_dir output/${run_name} \
    --init_model \
    --use_amp \
    --epochs 30 --lr $lr \
    --lr_temp_net 3e-5 \
    --train_frac $frac \
    --zs_dataset cifar10 \
    --ita_type clipPCT \
    --temperature_scheduler cosPCT \
    --tau_min 0.01 \
    --tau_max 0.05 \
    --pct_tau_min 0.01 \
    --pct_tau_max 0.05 \