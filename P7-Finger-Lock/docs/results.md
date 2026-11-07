# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 809, validation pictures: 272 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 1,072,326

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 100 | 80.5 % | 83.8 % | 83.8 % | 0.72 |
| no augmentation | 100 | 99.8 % | 72.8 % | 71.7 % | 3.043 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
