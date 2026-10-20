#!/bin/bash
# Run on the head node to queue the three Day-3 jobs.
sbatch job_lr_0.5.sh
sbatch job_lr_0.05.sh
sbatch job_lr_0.001.sh
squeue -u "$(whoami)"
