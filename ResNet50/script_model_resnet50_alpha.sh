#!/bin/bash
#
#SBATCH --job-name=TRAIN-RESNET-ALPHA
#SBATCH --output=/home/plnicolas/outputs/modelresnetalpha.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running ResNet50 on alpha..."
srun python model_resnet50_alpha.py