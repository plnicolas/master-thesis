#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-BRIGHT-BIGGER
#SBATCH --output=/home/plnicolas/outputs/modelpapybrightbigger.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net with brighness DA on bigger images..."
srun python model_papysnet_bright_bigger.py