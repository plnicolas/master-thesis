#!/bin/bash
#
#SBATCH --job-name=TRAIN-RESNET-IN
#SBATCH --output=/home/plnicolas/outputs/modelresnet_IN.txt
#
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1

echo "Running ResNet50..."
srun python model_resnet50_IN.py