# AI6103 Homework - Day 1-2

AI6103 homework: study how initial learning rate, learning rate schedule, weight
decay, and mixup affect EfficientNetV2-S trained from scratch on the Dog Breed
Identification dataset (120 breeds).

## Data and model (links from the assignment PDF)

- Model: EfficientNetV2-S from torchvision
  <https://docs.pytorch.org/vision/stable/models/efficientnetv2.html>
  (built-in, no manual download; trained from scratch with random weights)
- Dataset: Dog Breed Identification, Google Drive
  <https://drive.google.com/file/d/1PmfiGy1D1ssa8NFnwH8UDJJYCCj8NkHt/view?usp=sharing>
  (file ID `1PmfiGy1D1ssa8NFnwH8UDJJYCCj8NkHt`, downloaded by
  `scripts/download_data.py`)

Fixed settings required by the assignment: EfficientNetV2-S from torchvision
(random initialization), SGD with momentum 0.9, batch size 128, images resized
to 100x100, 12-pixel zero padding + random crop + horizontal flip (p=0.5) for
augmentation, per-channel mean/std computed on the training set for
normalization.

## Project layout

```
ai6103-homework/
├── src/                 # reusable training framework
│   ├── config.py        # experiment config dataclass
│   ├── dataset.py       # dataset, augmentation, channel statistics
│   ├── model.py         # EfficientNetV2-S from scratch
│   ├── train.py         # training entry point (SGD, cosine, weight decay, mixup)
│   ├── plot.py          # loss/accuracy curves
│   └── utils.py         # seeding, device, json helpers
├── scripts/
│   ├── download_data.py # Google Drive download + unzip
│   ├── inspect_data.py  # verify splits against 7155/1533/1534
│   ├── compute_stats.py # mean/std on the training set (Section 2)
│   └── smoke_test.py    # verify the pipeline without a full run
├── experiments/         # experiment configs and commands (Day 3+)
├── data/                # dataset and stats.json (gitignored)
└── runs/                # one folder per experiment (gitignored)
```

## Setup

```bash
cd ~/ai6103-homework
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU recommended for the 300-epoch runs. On Apple Silicon the code falls back to
MPS automatically.

## Day 1-2 workflow

1. Download and extract the dataset (Google Drive link from the assignment):

   ```bash
   python scripts/download_data.py
   ```

2. Verify the extracted layout and split sizes (train 7155 / val 1533 / test 1534):

   ```bash
   python scripts/inspect_data.py
   ```

   If the layout differs from the three supported in `src/dataset.py`
   (class folders per split / flat split folders with a per-split labels CSV /
   single images folder + split column), `inspect_data.py` output is enough to adapt
   `src/dataset.py` in a few lines.

   The course zip extracts to `data/{train,val,test}/images/` plus
   `data/{split}_labels.csv` (columns `id,breed`); `src/dataset.py` already
   handles this layout.

3. Compute normalization statistics on the training set (report these numbers):

   ```bash
   python scripts/compute_stats.py
   ```

   This saves `data/stats.json`; the training script reuses it and recomputes it
   automatically if missing.

   Computed on the extracted dataset (report these in Section 2):

   ```
   mean = [0.4772, 0.4528, 0.3928]
   std  = [0.2534, 0.2480, 0.2523]
   ```

4. Smoke test the whole pipeline (few steps, tiny batch, mixup enabled):

   ```bash
   python scripts/smoke_test.py
   ```

5. Run a real experiment (example: Section 3, lr=0.05, 15 epochs):

   ```bash
   python -m src.train --name lr_0.05 --lr 0.05 --epochs 15
   ```

   Each run writes to `runs/<name>/`: `config.json`, `history.csv`, `best.pt`,
   `last.pt`, `summary.json`, and `loss_curves.png` / `acc_curves.png` /
   `training_curves.png`.

## Day 1-2 checklist

- [x] Dataset downloaded and extracted
- [x] `inspect_data.py` confirms 7155/1533/1534 split sizes
- [x] Mean/std per channel computed on the training set and saved to `data/stats.json`
- [x] `smoke_test.py` passes: loading, augmentation, mixup, forward/backward
- [x] One short training run completes and produces both curve plots (Day 3: `runs/lr_0.5`, `runs/lr_0.001`; lr=0.05 pending rerun)

See `experiments/README.md` for the Day 3+ experiment commands.

中文版作业计划表见 `PLAN.md`。
