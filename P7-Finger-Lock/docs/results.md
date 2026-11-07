# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 656, validation pictures: 217 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 1,072,326

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 100 | 94.2 % | 100.0 % | 99.5 % | 0.024 |
| no augmentation | 100 | 100.0 % | 97.7 % | 92.6 % | 0.802 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
