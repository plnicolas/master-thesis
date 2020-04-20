#!/bin/bash
#
#SBATCH --job-name=TRAIN-PAPY-NO-DA
#SBATCH --output=/home/plnicolas/outputs/modelpapynoDA.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running Papy-S-Net..."
srun python model_papysnet_no_DA.py