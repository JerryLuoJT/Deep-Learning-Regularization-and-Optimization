#!/bin/bash
# Run locally (with NTU VPN if off campus) to check Day 4-5 progress.
ssh ccds 'cd ~/ai6103-homework/cluster && date +%H:%M:%S && squeue -u jingtao003 | tail -n +2; echo "const:   $(grep epoch slurm_ai6103_sched_const_41692.out 2>/dev/null | tail -1)"; echo "cosine:  $(grep epoch slurm_ai6103_sched_cosine_41693.out 2>/dev/null | tail -1)"'
