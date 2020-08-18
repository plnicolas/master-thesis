#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-MIX
#SBATCH --output=/home/plnicolas/outputs/modelpapymix.txt
#
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net with all approaches..."
srun python model_papysnet.py --size 224 --batch_size 32 --brightness 1