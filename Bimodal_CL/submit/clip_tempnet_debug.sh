#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --ntasks=1

#SBATCH --gres=gpu:4
#SBATCH --mem=480000

#SBATCH --time=00:59:59

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet_/Bimodal_CL"
cd "${PROJECT_DIR}"

data=cc3m

lr=8e-4
desc=clip_tempnet_lr8e-4_M256_pt_dataloader_debug_3
rho=7.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --data ${data} \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/$desc \
    --init_model \
    --use_amp \
    --epochs 30 --lr ${lr} \
    --lr_temp_net 3e-5 \
    --rho ${rho} \
    --ita_type clip_tempnet \
    --batch_size_train 256 \
    --run_name $desc