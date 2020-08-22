#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-BIGGER
#SBATCH --output=/home/plnicolas/outputs/modelpapybigger.txt
#
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net bigger on crop..."
srun python model_papysnet.py --size 224