#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-BRIGHT
#SBATCH --output=/home/plnicolas/outputs/modelpapybright.txt
#
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net with all approaches..."
srun python model_papysnet.py --brightness 1