export mpi=1
PROJECT_DIR="/BS/dduka/work/projects/old_projects/TempNet/Bimodal_CL"
cd "${PROJECT_DIR}"

MODEL_PATHS=(
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_tau_0.01_lr_8e-4/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_clip_cos_0.01_0.05_lr_8e-4/checkpoint_best.pth
    /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/r_scheduled_clip_0.01_0.04_lr_2e-4_quad_crossmodal_and_unimodal_augmented/checkpoint_best.pth
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $(dirname "$MODEL_PATH"))
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=7800 \
            --use_env clip.py \
            --run_name ${MODEL_NAME}_EEEdasdssads \
            --data cc3m \
            --output_dir /BS/dduka/work/training_metadata/bimodal_cl/dhimitrios/${MODEL_NAME}_ZSH_EVAL_REMOVEdasdsda_ \
            --init_model \
            --use_amp \
            --zs_dataset fgvc-aircraft oxford-pets eurosat country211 dtd sun397 \
            --ita_type clip \
            --checkpoint $MODEL_PATH \
            --zsh_eval \
            --zs_datafolder /BS/databases23/imagenet/original/
done