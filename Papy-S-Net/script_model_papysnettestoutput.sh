#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-TOUT
#SBATCH --output=/home/plnicolas/outputs/modelpapytestoutput.txt
#
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net..."
srun python model_papysnet_testoutput.py