#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=ai6103_smi
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
nvidia-smi
