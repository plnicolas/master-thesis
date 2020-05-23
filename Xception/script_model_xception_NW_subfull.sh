#!/bin/bash
#
#SBATCH --job-name=TRAIN-XCEPTION-NW-SUBFULL
#SBATCH --output=/home/plnicolas/outputs/modelxception_NW_subfull.txt
#
#SBATCH --ntasks=1
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Xception..."
srun python model_xception_NW_subfull.py