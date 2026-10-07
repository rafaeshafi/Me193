"""Spin from the swing itself: StandardScaler + LogisticRegression on the 12 swing features.

Three classes, learned per player from a handful of labelled swings:
  flat  a straight push through the ball
  top   brush up over it (the wrist rolls forward-up)
  back  chop down under it
The features are the ones the swing detector already computes at every IMPACT: the unit direction of
the gyro peak, of the net rotation and of the linear acceleration, the log of the peak rate, the
stroke duration and the backswing ratio.  predict_proba gives continuous topspin / backspin through
shot.spin_from_probs (an uncertain swing, max probability < 0.5, is simply flat).

Honesty rule: the accuracy shown is cross-validated (every swing is predicted by a model that never
saw it) and the model is only used in the game if it reaches SHIP_ACCURACY; below that the game stays
flat and says so.  With ~12 swings per class and one player it is an estimate, not a guarantee.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pingpong import profile

CLASSES = ("flat", "top", "back")
MIN_PER_CLASS = 6
SHIP_ACCURACY = 0.75


class SpinModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.classes = [str(c) for c in pipeline.named_steps["clf"].classes_]

    def probs(self, feat):
        """{"flat": p, "top": p, "back": p} for one swing's 12 features."""
        p = self.pipeline.predict_proba(np.asarray(feat, dtype=float).reshape(1, -1))[0]
        return {name: float(value) for name, value in zip(self.classes, p)}


def train(X, y, *, seed=0):
    """-> (SpinModel fitted on every swing, report with the cross-validated accuracy and the ship verdict)."""
    y = [str(label) for label in y]
    counts = {c: y.count(c) for c in CLASSES}
    for c in CLASSES:
        if counts[c] == 0:
            raise ValueError(f"no swings recorded for the {c!r} class")
        if counts[c] < MIN_PER_CLASS:
            raise ValueError(f"need at least {MIN_PER_CLASS} swings per class, {c!r} has {counts[c]}")
    X = np.asarray(X, dtype=float)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed))])
    folds = min(4, *counts.values())
    predicted = cross_val_predict(pipe, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    accuracy = float(np.mean(np.asarray(predicted) == np.asarray(y)))
    confusion = {a: {b: int(sum(1 for t, p in zip(y, predicted) if t == a and p == b)) for b in CLASSES}
                 for a in CLASSES}
    pipe.fit(X, y)
    report = {"n_per_class": counts, "cv_accuracy": accuracy, "cv_folds": folds, "confusion": confusion,
              "ship": accuracy >= SHIP_ACCURACY, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    return SpinModel(pipe), report


def _dir(name, root):
    return Path(root or profile.default_root()) / profile.slug(name)


def save(name, model, report, root=None):
    folder = _dir(name, root)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model.pipeline, "report": report}, folder / "spin.joblib")
    (folder / "spin.json").write_text(json.dumps(report, indent=2) + "\n")
    return folder / "spin.joblib"


def load_for(name, root=None, *, force=False):
    """The player's spin model, or None if there is none (or it did not reach the bar and force is off)."""
    path = _dir(name, root) / "spin.joblib"
    if not path.exists():
        return None
    try:
        data = joblib.load(path)
        model = SpinModel(data["pipeline"])
    except Exception as exc:
        raise ValueError(f"the spin model file {path} is damaged ({exc}); retrain with ./pp train_spin") from exc
    return model if force or data["report"]["ship"] else None
