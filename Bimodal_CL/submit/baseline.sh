#!/bin/bash

#SBATCH --job-name bimodal_cl
#SBATCH --partition gpu20

#SBATCH --time=09:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

TRANSFORMERS_OFFLINE=1
data_path=/data1/VLP
imagenet_val_path=/data1/imagenet/val
train_image_root=cc3m
data=cc3m
train_file=clip_train/${data}_train_new.json
lr=8e-4
frac=1.0
desc=isogclr_tempnet
gamma=0.8
rho=7.0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4820 \
    --use_env clip.py \
    --data_path ${data_path} \
    --data ${data} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/isogclr_tempnet_${data}_gamma${gamma}_rho${rho}_${desc} \
    --init_model \
    --use_amp \
    --epochs 30 --lr ${lr} \
    --lr_temp_net 3e-5 \
    --rho ${rho} \
    --train_frac ${frac} \
    --zs_dataset imagenet \
    --zs_datafolder ${imagenet_val_path} \
    --ita_type isogclr_tempnet \
    --sogclr_gamma ${gamma} > isogclr_tempnet_${data}_gamma${gamma}_rho${rho}_${desc}.log &