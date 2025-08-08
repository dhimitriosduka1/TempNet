#!/bin/bash

#SBATCH --job-name cc3m_multi
#SBATCH --partition gpu22

#SBATCH --time=03:59:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH -o /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/bimodal_cl/%A_%a_%x_%j_%N.err

export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

DATA=cc3m
LR=8e-4
ITA_TYPE=clip

# Base checkpoint directory
BASE_CHECKPOINT_DIR="/BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_tau_0.01_lr_8e-4_multiple_checkpoints"

# WandB group name for organizing runs
WANDB_GROUP="CLIP_CHECKPOINT_EVALUATION_$(date +%Y%m%d_%H%M)"

# Function to check if a string is a number
is_number() {
    re='^[0-9]+$'
    if [[ $1 =~ $re ]] ; then
        return 0
    else
        return 1
    fi
}

# Find all checkpoint files and extract numbers
echo "Searching for checkpoints in: $BASE_CHECKPOINT_DIR"
checkpoint_numbers=()

for checkpoint_file in "$BASE_CHECKPOINT_DIR"/checkpoint_*.pth; do
    if [[ -f "$checkpoint_file" ]]; then
        # Extract the filename without path
        filename=$(basename "$checkpoint_file")
        # Extract the part between "checkpoint_" and ".pth"
        number_part=$(echo "$filename" | sed -n 's/checkpoint_\(.*\)\.pth/\1/p')
        
        # Check if it's a number (not "best" or other strings)
        if is_number "$number_part"; then
            checkpoint_numbers+=("$number_part")
            echo "Found numbered checkpoint: $number_part"
        else
            echo "Skipping non-numbered checkpoint: $filename"
        fi
    fi
done

# Sort checkpoint numbers numerically
IFS=$'\n' sorted_numbers=($(sort -n <<<"${checkpoint_numbers[*]}"))
unset IFS

echo "Found ${#sorted_numbers[@]} numbered checkpoints to process"

# Process each checkpoint
for checkpoint_num in "${sorted_numbers[@]}"; do
    checkpoint_path="$BASE_CHECKPOINT_DIR/checkpoint_${checkpoint_num}.pth"
    
    if [[ -f "$checkpoint_path" ]]; then
        echo "Processing checkpoint: $checkpoint_path"
        
        # Create unique run name
        RUN_NAME="CLIP_EVAL_CHECKPOINT_${checkpoint_num}"
        
        # Create unique output directory
        OUTPUT_DIR="/BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/clip_eval_checkpoint_${checkpoint_num}_$(date +%Y%m%d_%H%M%S)"
        
        echo "Starting run: $RUN_NAME"
        echo "Output directory: $OUTPUT_DIR"
        
        # Run the training/evaluation with current checkpoint
        CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4820 \
            --use_env clip.py \
            --run_name "$RUN_NAME" \
            --wandb_group "$WANDB_GROUP" \
            --data "$DATA" \
            --output_dir "$OUTPUT_DIR" \
            --init_model \
            --use_amp \
            --epochs 30 --lr "$LR" \
            --ita_type "$ITA_TYPE" \
            --checkpoint "$checkpoint_path" \
            --compute_temperature_assignments \
        
        # Check if the run completed successfully
        if [[ $? -eq 0 ]]; then
            echo "Successfully completed run for checkpoint $checkpoint_num"
        else
            echo "Error: Run failed for checkpoint $checkpoint_num"
            # Optionally continue with other checkpoints or exit
            # exit 1
        fi
        
        echo "----------------------------------------"
        
    else
        echo "Warning: Checkpoint file not found: $checkpoint_path"
    fi
done

echo "All checkpoint evaluations completed!"
echo "WandB group: $WANDB_GROUP"