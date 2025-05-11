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

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4820 \
    --use_env clip.py \
    --run_name iSogCLR_TempNet_Baseline \
    --data_path /ptmp/dduka/work/data/ \
    --data cc3m \
    --output_dir /ptmp/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_tempnet_baseline_rerun/ \
    --init_model \
    --use_amp \
    --epochs 30 --lr 8e-4 \
    --lr_temp_net 3e-5 \
    --rho 7.0 \
    --ita_type isogclr_tempnet \
    --sogclr_gamma 0.8