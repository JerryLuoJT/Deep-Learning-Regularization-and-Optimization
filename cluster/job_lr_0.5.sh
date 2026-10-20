#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=ai6103_lr_0.5
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

module load anaconda/25.5.1
cd "$HOME/ai6103-homework"
python -m src.train --config experiments/lr_0.5.json
