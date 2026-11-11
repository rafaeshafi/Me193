# Results

*Written by `train/train.py` on 2026-11-10. Real webcam pictures.*

- training pictures: 672, validation pictures: 224 (held-out sessions: last 25 % of every burst (same sessions as training))
- FingerNet parameters: 1,072,326

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 100 | 96.0 % | 100.0 % | 100.0 % | 0.0 |
| no augmentation | 100 | 100.0 % | 100.0 % | 100.0 % | 0.0 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
