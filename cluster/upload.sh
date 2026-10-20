#!/bin/bash
# Run on the Mac: upload code + dataset to the cluster home directory.
set -e
cd "$(dirname "$0")/.."

HOST="10.96.189.12"
USER="${NTU_USER:-}"
if [ -z "$USER" ]; then
    echo "usage: NTU_USER=<your-ntu-account> ./cluster/upload.sh"
    exit 1
fi

ssh "$USER@$HOST" "mkdir -p ~/ai6103-homework/data"
rsync -avz --exclude '.venv' --exclude 'runs' --exclude '__pycache__' \
    ./src ./scripts ./experiments ./cluster ./PLAN.md \
    "$USER@$HOST:~/ai6103-homework/"
rsync -avz data/stats.json data/train_labels.csv data/val_labels.csv data/test_labels.csv \
    data/train data/val data/test \
    "$USER@$HOST:~/ai6103-homework/data/"
echo "upload done"
