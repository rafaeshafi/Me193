# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 627, validation pictures: 208 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 1,072,326

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 100 | 98.9 % | 94.7 % | 92.8 % | 0.235 |
| no augmentation | 100 | 100.0 % | 97.1 % | 87.0 % | 2.098 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
