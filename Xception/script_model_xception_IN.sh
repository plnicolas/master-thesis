#!/bin/bash
#
#SBATCH --job-name=TRAIN-XCEPTION-IN
#SBATCH --output=/home/plnicolas/outputs/modelxception_IN.txt
#
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1

echo "Running Xception..."
srun python model_xception_IN.py