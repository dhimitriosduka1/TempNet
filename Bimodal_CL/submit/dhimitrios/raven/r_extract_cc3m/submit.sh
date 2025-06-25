#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

#SBATCH --job-name bcl

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=23:59:59

module purge
module load anaconda/3/2023.03

conda activate bimodal_cl

TAR_SOURCE_DIR="/ptmp/dduka/work/data/cc3m/training"
EXTRACT_DEST_DIR="/ptmp/dduka/work/data/cc3m/training/extracted"

# Create destination directory if it doesn't exist
mkdir -p "$EXTRACT_DEST_DIR"

# Change to the destination directory
cd "$EXTRACT_DEST_DIR"

echo "Starting extraction at $(date)"
echo "Source directory: $TAR_SOURCE_DIR"
echo "Destination directory: $EXTRACT_DEST_DIR"

# Initialize counters
total_files=0
extracted_files=0
failed_files=0

# Count total tar files
total_files=$(find "$TAR_SOURCE_DIR" -name "*.tar" -type f | wc -l)
echo "Found $total_files tar files to extract"

# Extract each tar file
for tar_file in "$TAR_SOURCE_DIR"/*.tar; do
    if [ -f "$tar_file" ]; then
        echo "Extracting: $(basename "$tar_file")"
        
        # Extract with verbose output and preserve permissions
        if tar -xvf "$tar_file"; then
            extracted_files=$((extracted_files + 1))
            echo "Successfully extracted: $(basename "$tar_file")"
        else
            failed_files=$((failed_files + 1))
            echo "ERROR: Failed to extract $(basename "$tar_file")" >&2
        fi
        
        # Show progress
        echo "Progress: $extracted_files/$total_files extracted, $failed_files failed"
        echo "---"
    fi
done

echo "Extraction completed at $(date)"
echo "Summary:"
echo "  Total files found: $total_files"
echo "  Successfully extracted: $extracted_files"
echo "  Failed extractions: $failed_files"

# Show final disk usage
echo "Final disk usage in destination:"
du -sh "$EXTRACT_DEST_DIR"