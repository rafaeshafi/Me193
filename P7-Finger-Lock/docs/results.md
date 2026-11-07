# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 809, validation pictures: 272 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 285,894

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 100 | 76.9 % | 82.4 % | 79.8 % | 0.805 |
| no augmentation | 100 | 99.0 % | 79.0 % | 75.0 % | 2.159 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
