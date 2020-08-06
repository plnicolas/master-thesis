#!/bin/bash
#
#SBATCH --job-name=EVAL-PAPY-CROPVARIANTS
#SBATCH --output=/home/plnicolas/outputs/evaluatepapycropvariants.txt
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

echo "Evaluating Papy-S-Net..."
srun python EvaluateCropBatch32.py
srun python EvaluateCropBigger.py
srun python EvaluateCropBright.py