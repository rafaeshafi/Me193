# Results

*Written by `train/train.py` on 2026-11-07. Real webcam pictures.*

- training pictures: 616, validation pictures: 204 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 1,072,326

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 100 | 90.9 % | 100.0 % | 99.0 % | 0.028 |
| no augmentation | 100 | 100.0 % | 97.1 % | 92.2 % | 0.827 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
