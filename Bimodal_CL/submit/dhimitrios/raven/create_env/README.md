To create the env I had to:
1. Install [rustc](https://rustup.rs/)
2. export RUSTFLAGS="-A invalid_reference_casting"

Script header for Reaven:
```bash
#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name environment

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --mem=480000

#SBATCH --time=01:59:59

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

export mpcdf=1
```