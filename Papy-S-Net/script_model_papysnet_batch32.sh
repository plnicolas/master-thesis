#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-BATCH32
#SBATCH --output=/home/plnicolas/outputs/modelpapybatch32.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net with batch size 32..."
srun python model_papysnet.py --batch_size 32