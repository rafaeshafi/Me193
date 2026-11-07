# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 809, validation pictures: 272 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 285,894

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 60 | 80.3 % | 75.4 % | 72.8 % | 1.003 |
| no augmentation | 60 | 97.4 % | 79.0 % | 78.3 % | 1.366 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
