#!/bin/bash
#
#SBATCH --job-name=TRAIN-XCEPTION-CL
#SBATCH --output=/home/plnicolas/outputs/modelxceptionCL.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Xception..."
srun python model_xception_contrastive_loss.py