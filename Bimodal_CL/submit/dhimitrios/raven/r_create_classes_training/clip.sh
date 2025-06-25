#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --nodes=2              
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4           
#SBATCH --mem=240000          

#SBATCH --time=03:59:59

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1

PROJECT_DIR="/u/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4
ITA_TYPE=clip

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name R_CREATE_CLASSES_TRAINING \
    --data $DATA \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_create_classes_training/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr $LR \
    --ita_type $ITA_TYPE \
    --find_clusters \
    --num_clusters 18 \