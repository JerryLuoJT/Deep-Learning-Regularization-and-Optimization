#!/bin/bash
# Run once on the CCDS head node (10.96.189.12) to create the conda env.
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^ai6103 "; then
    echo "conda env ai6103 already exists, skipping creation"
else
    conda create -n ai6103 python=3.10 -y
fi

conda activate ai6103
python -m pip install --upgrade pip
pip install torch torchvision numpy pillow matplotlib tqdm
echo "done: conda env ai6103 ready"
