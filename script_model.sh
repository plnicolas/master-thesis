#!/bin/bash
#
#SBATCH --job-name=papy
#SBATCH --output=/home/plnicolas/outputs/model.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

srun conda activate deeplearning
srun python model_resnet50.py
srun python model_xception.py
srun python model_papysnet.py