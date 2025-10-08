#!/bin/bash

# Fine-tune VGGT Camera Head on nuScenes
# This script freezes all VGGT parameters except the last 2 alternating attention blocks
# and trains only on camera pose estimation loss

set -e

echo "=========================================="
echo "VGGT Camera Head Fine-tuning on nuScenes"
echo "=========================================="

# Check if nuScenes path is set
if [ -z "$NUSCENES_DIR" ]; then
    echo "Warning: NUSCENES_DIR environment variable not set."
    echo "Please set it or update the config file with the correct path."
    echo "Example: export NUSCENES_DIR=/path/to/nuscenes"
fi

# Run training
echo "Starting training..."
cd /users/PAS2099/jasonzhangggg/vggt/training

python finetune_camera_nuscenes.py \
    data.nuscenes_dir="${NUSCENES_DIR:-/path/to/nuscenes}" \
    training.batch_size=2 \
    training.num_epochs=20 \
    model.n_blocks_unfreeze=2 \
    "$@"

echo "Training completed!"
