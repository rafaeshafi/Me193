# Results

*Written by `train/train.py` on 2026-11-04. **Synthetic hands** (pipeline check), not real pictures.*

- training pictures: 600, validation pictures: 300 (held-out sessions: synth2)
- FingerNet parameters: 285,894

| run | epochs | final train acc | best val acc | final val acc | final val loss |
|---|---|---|---|---|---|
| augmented | 40 | 71.8 % | 89.3 % | 88.3 % | 0.378 |
| no augmentation | 40 | 94.2 % | 96.3 % | 96.0 % | 0.111 |

Per-class results of the deployed run are in the confusion matrix (`docs/figures/confusion.png`); the curves are in `docs/figures/train_curves.png`.
