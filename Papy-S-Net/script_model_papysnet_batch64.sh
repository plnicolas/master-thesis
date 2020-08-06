#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-BATCH64
#SBATCH --output=/home/plnicolas/outputs/modelpapybatch64.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net with batch size 64..."
srun python model_papysnet_batch64.py