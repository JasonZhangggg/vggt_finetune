#!/bin/bash
#SBATCH --job-name=vggt_finetune_11_5
#SBATCH --account=PAS2099
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2099/jasonzhangggg/vggt/training/logs/vggt_finetune_11_5.out
#SBATCH --error=/users/PAS2099/jasonzhangggg/vggt/training/logs/vggt_finetune_11_5.err

# Load your environment (adjust if using conda or modules)
module load miniconda3/24.1.2-py310
conda activate vggt
module load cuda/11.8.0

torchrun --nproc_per_node=4 /users/PAS2099/jasonzhangggg/vggt/training/launch.py --config finetune_nuscenes_sparse