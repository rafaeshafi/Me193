# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 574, validation pictures: 193 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 285,894

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 60 | 86.4 % | 78.2 % | 76.2 % | 0.78 |
| no augmentation | 60 | 99.5 % | 77.7 % | 75.1 % | 1.582 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
