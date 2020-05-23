#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-ALPHA
#SBATCH --output=/home/plnicolas/outputs/modelpapyalpha.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net on alpha..."
srun python model_papysnet_alpha.py