#!/bin/bash
#
#SBATCH --job-name=EVAL-PAPY
#SBATCH --output=/home/plnicolas/outputs/evaluatepapy.txt
#
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Running the evaluation of Papy-S-Net..."
srun python evaluate_crop.py