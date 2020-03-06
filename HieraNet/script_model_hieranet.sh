#!/bin/bash
#
#SBATCH --job-name=TRAIN-HIERA
#SBATCH --output=/home/plnicolas/outputs/modelhieranet.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running HieraNet..."
srun python model_hieranet.py