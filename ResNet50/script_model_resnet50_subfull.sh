#!/bin/bash
#
#SBATCH --job-name=TRAIN-RESNET-SUBFULL
#SBATCH --output=/home/plnicolas/outputs/modelresnet_subfull.txt
#
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running ResNet50..."
srun python model_resnet50_subfull.py