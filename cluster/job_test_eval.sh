#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=ai6103_test_eval
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

module load anaconda/25.5.1
cd "$HOME/ai6103-homework"
python scripts/evaluate_test.py runs/mixup_0.2/best.pt --device cuda
