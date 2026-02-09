#!/bin/bash
# run_training.sh - Scripts to run contrastive pretraining experiments

# Set project directory to this script's location
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Function to run training
run_experiment() {
    CONFIG=$1
    GPU=$2
    echo "Starting experiment with config: $CONFIG"
    CUDA_VISIBLE_DEVICES=$GPU python3 pretrain.py \
        --config $CONFIG \
        --gpus 1 \
        --mixed-precision
}

# DeepConvLSTM (lower body) + X3D-XS
run_experiment "configs/pretrain_deepconv_lower_x3d_revalexo.yaml" 0

# DeepConvLSTM (lower body) + MViT-B (uncomment to run)
# run_experiment "configs/pretrain_deepconv_lower_mvit_revalexo.yaml" 0

echo "Training completed!"
