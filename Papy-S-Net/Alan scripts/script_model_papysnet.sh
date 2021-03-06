#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY
#SBATCH --output=/home/plnicolas/outputs/modelpapy.txt
#
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net..."
srun python model_papysnet.py