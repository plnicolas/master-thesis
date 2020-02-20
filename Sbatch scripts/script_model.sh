#!/bin/bash
#
#SBATCH --job-name=papy
#SBATCH --output=/home/plnicolas/outputs/modelok.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running ResNet50..."
srun python model_resnet50.py

echo "Running Xception..."
srun python model_xception.py

echo "Running Papy-S-Net..."
srun python model_papysnet.py