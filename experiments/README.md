# Experiment plan (Day 3+)

Every experiment uses `python -m src.train` from the project root. Fixed
defaults: EfficientNetV2-S from scratch, SGD momentum 0.9, batch size 128,
random crop + horizontal flip augmentation, normalization with `data/stats.json`.

| Section | Experiment | Command | Points |
|---|---|---|---|
| 3 Learning rate | lr = 0.5 | `python -m src.train --name lr_0.5 --lr 0.5 --epochs 15` | 20% |
| 3 Learning rate | lr = 0.05 | `python -m src.train --name lr_0.05 --lr 0.05 --epochs 15` | 20% |
| 3 Learning rate | lr = 0.001 | `python -m src.train --name lr_0.001 --lr 0.001 --epochs 15` | 20% |
| 4 Schedule | lr=0.05, constant | `python -m src.train --config experiments/sched_const.json` | 15% |
| 4 Schedule | lr=0.05, cosine | `python -m src.train --config experiments/sched_cosine.json` | 5% + 15% |
| 5 Weight decay | lr=0.05 + cosine, λ=5e-4 | `python -m src.train --config experiments/wd_5e-4.json` | 20% |
| 5 Weight decay | lr=0.05 + cosine, λ=1e-4 | `python -m src.train --config experiments/wd_1e-4.json` | 20% |
| 6 Mixup | best setup, α=0.2 | `python -m src.train --name mixup --lr <best> --epochs 300 --schedule cosine --weight-decay <best-wd> --mixup-alpha 0.2` | 5% + 20% |
| 6 Test set | best model on test split | final evaluation script (to be added on Day 8-9) | 5% |

## Order of operations

1. Pick the best initial learning rate from Section 3 by training loss
   (also report training/validation accuracy behavior).
2. For Section 4, describe cosine annealing with equations in the report;
   run constant vs cosine with the best lr for 300 epochs.
3. For Section 5, fix cosine annealing and add λ = 5e-4 and 1e-4.
4. For Section 6, add mixup α=0.2 to the best configuration; draw the
   Beta(0.2, 0.2) density in the report.
5. Only after all tuning, evaluate the best checkpoint once on the hold-out
   test set (5%).

## Report reminders

- AAAI double-column format, body <= 6 pages excluding references.
- Show the preprocessing code; plot train/val loss and train/val accuracy vs
  epochs for every experiment.
- Describe findings in words and relate them to lecture material.
- Do not answer the "Takeaway Message" question in the report.
