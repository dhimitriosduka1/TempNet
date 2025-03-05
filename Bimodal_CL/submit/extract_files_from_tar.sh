#!/bin/bash

#SBATCH --job-name cc3m
#SBATCH --partition cpu20

#SBATCH --time=01:30:00
#SBATCH --array=0-110%20

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

# Use the SLURM_ARRAY_TASK_ID as the input for the Python script
python3 ./utils/extract_tar_data.py --i ${SLURM_ARRAY_TASK_ID}