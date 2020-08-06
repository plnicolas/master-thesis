#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-BRIGHT
#SBATCH --output=/home/plnicolas/outputs/modelpapybright.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net with brighness DA..."
srun python model_papysnet_bright.py