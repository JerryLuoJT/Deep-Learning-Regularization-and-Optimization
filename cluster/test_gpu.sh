#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ai6103_gputest
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

module load anaconda/25.5.1
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", props.major, props.minor)
    print("memory GB:", round(props.total_memory / 1e9, 1))
    x = torch.randn(128, 3, 100, 100, device="cuda")
    print("cuda matmul ok:", torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device="cuda")).shape)
EOF
