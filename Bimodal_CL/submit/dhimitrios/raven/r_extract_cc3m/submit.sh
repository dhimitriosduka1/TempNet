#!/bin/bash -l

#SBATCH --job-name=cc3m_extract
#SBATCH --array=1-100%20

#SBATCH --output=tar_array_%A_%a.out
#SBATCH --error=tar_array_%A_%a.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4
#SBATCH --mem=16000

# Load required modules for Raven
module purge
module load gcc/13

# Configuration - modify these paths as needed
TAR_SOURCE_DIR="/ptmp/dduka/work/data/cc3m/training"
EXTRACT_DEST_DIR="/ptmp/dduka/work/data/cc3m/training/extracted"

# Create destination directory if it doesn't exist
mkdir -p "$EXTRACT_DEST_DIR"

# Create array of tar files
mapfile -t TAR_FILES < <(find "$TAR_SOURCE_DIR" -name "*.tar" -type f | sort)

# Calculate which file this job should process
TOTAL_FILES=${#TAR_FILES[@]}
FILES_PER_JOB=$(( (TOTAL_FILES + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * FILES_PER_JOB ))
END_INDEX=$(( START_INDEX + FILES_PER_JOB - 1 ))

# Ensure we don't go beyond the array bounds
if [ $END_INDEX -ge $TOTAL_FILES ]; then
    END_INDEX=$(( TOTAL_FILES - 1 ))
fi

echo "Job array task $SLURM_ARRAY_TASK_ID processing files $START_INDEX to $END_INDEX"
echo "Total files: $TOTAL_FILES"

# Process assigned files
for (( i=$START_INDEX; i<=$END_INDEX; i++ )); do
    if [ $i -lt $TOTAL_FILES ]; then
        tar_file="${TAR_FILES[$i]}"
        basename_file=$(basename "$tar_file")
        
        echo "Extracting: $basename_file"
        
        if tar -xf "$tar_file" -C "$EXTRACT_DEST_DIR"; then
            echo "SUCCESS: $basename_file"
        else
            echo "ERROR: Failed to extract $basename_file" >&2
        fi
    fi
done

echo "Job array task $SLURM_ARRAY_TASK_ID completed at $(date)"