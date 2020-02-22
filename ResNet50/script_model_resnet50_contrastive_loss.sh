#!/bin/bash
#
#SBATCH --job-name=TRAIN-RESNET-CL
#SBATCH --output=/home/plnicolas/outputs/modelresnetCL.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running ResNet50 with contrastive loss..."
srun python model_resnet50_contrastive_loss.py