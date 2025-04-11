#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name environment

#SBATCH --ntasks=1
#SBATCH --constraint="apu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=01:59:59

module purge
module load anaconda/3/2023.03

export RUSTFLAGS="-A invalid_reference_casting"

# Load env
conda init bash
source ~/.bashrc

conda env create --force -f /u/dduka/work/projects/TempNet/Bimodal_CL/environment.yaml
conda activate bimodal_cl

echo "Environment created"