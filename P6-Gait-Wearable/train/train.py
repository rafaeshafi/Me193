"""Train both decision trees from the recorded CSVs and export them as JSON for the board.

  .venv/bin/python train/train.py            # data/*.csv -> models/model1.json, models/model2.json
  .venv/bin/python train/train.py --peer     # also report leave-one-person-out accuracy (who = filename suffix)

Windows are cut exactly as on the board (2 s, 0.5 s hop) with app/python/features.py, so there is no train/live
mismatch. Model 1: depth-3 tree on peak_x only. Model 2: depth-5 tree on MODEL2_FEATURES. Hold-out is by session so a
tree never sees windows from the session it is scored on; the exported trees are refit on everything.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "python"))
from features import MODEL1_FEATURES, MODEL2_FEATURES, RATE, compute  # noqa: E402

LABELS = ["STAND", "WALK", "RUN", "BIKE"]
WINDOW_S, HOP_S = 2.0, 0.5
MODELS = {"model1": (MODEL1_FEATURES, 3, 5), "model2": (MODEL2_FEATURES, 5, 5)}  # name: (features, depth, min leaf)


def load_session(path):
    """-> [(t, label, (ax, ay, az, gx, gy, gz)), ...] in file order."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), r["label"], tuple(float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz"))))
    return rows


def runs(rows):
    """Split rows at a real break in t (FIFO overflow or re-probe, 0.8 s or more). Batches are stamped on arrival, so ordinary jitter makes t step back a few ms; that must not split."""
    run = []
    for row in rows:
        if run and not (-0.3 <= row[0] - run[-1][0] <= 0.3):
            yield run
            run = []
        run.append(row)
    if run:
        yield run


def windows(path, rate=RATE):
    """Yield (label, features) for every 2 s window (0.5 s hop) whose rows all carry one real label."""
    n, hop = int(WINDOW_S * rate), int(HOP_S * rate)
    for rows in runs(load_session(path)):
        for start in range(0, len(rows) - n + 1, hop):
            chunk = rows[start:start + n]
            labels = {lab for _, lab, _ in chunk}
            if len(labels) != 1:
                continue
            label = labels.pop()
            if label not in LABELS:
                continue
            yield label, compute([s for _, _, s in chunk], rate)


def person_of(name):
    """<stamp>-<kind>[-<who>].csv -> who, or 'me' when there is no suffix."""
    stem = re.sub(r"-2$", "", Path(name).stem)      # a second take in the same second gets a -2 suffix
    parts = stem.split("-", 3)
    return parts[3] if len(parts) >= 4 else "me"


def dataset(paths):
    feats, y, groups, persons = [], [], [], []
    for p in paths:
        for label, f in windows(p):
            feats.append(f)
            y.append(label)
            groups.append(Path(p).name)
            persons.append(person_of(p))
    return feats, y, groups, persons


def matrix(feats, names):
    return np.array([[f[n] for n in names] for f in feats], dtype=float)


def fit(feats, y, names, depth, leaf):
    X = matrix(feats, names)
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y)
    return clf, X


def export_tree(clf, names):
    t = clf.tree_

    def node(i):
        if t.children_left[i] == -1:
            return {"label": str(clf.classes_[int(np.argmax(t.value[i]))])}
        return {"f": names[t.feature[i]], "t": float(t.threshold[i]),
                "l": node(t.children_left[i]), "r": node(t.children_right[i])}

    return {"features": list(names), "classes": [str(c) for c in clf.classes_], "root": node(0)}


def heldout_predictions(X, y, groups, depth, leaf):
    """Every window predicted by a tree that never saw its own session (leave-one-session-out), pooled."""
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    return cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut())


def report(title, y_true, y_pred):
    print("\n%s: accuracy %.1f%% on %d windows" % (title, 100 * accuracy_score(y_true, y_pred), len(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("true \\ pred  " + "".join("%7s" % l for l in LABELS))
    for lab, row in zip(LABELS, cm):
        print("%-12s" % lab + "".join("%7d" % v for v in row))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    print("recall  " + "  ".join("%s %.2f" % (l, r) for l, r in zip(LABELS, recall)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--peer", action="store_true", help="also score leave-one-person-out")
    ap.add_argument("--force", action="store_true", help="export even when some activities are missing")
    args = ap.parse_args()
    paths = sorted(p for p in Path(args.data).glob("*.csv") if not p.name.startswith("rolling-"))
    if not paths:
        print("no recordings in", args.data, "(run scripts/pull_data.sh first)")
        return 1
    feats, y, groups, persons = dataset(paths)
    if len(y) == 0:
        print("no labelled windows")
        return 1
    print("%d windows from %d sessions: %s" % (len(y), len(paths), dict(Counter(y))))
    if len(set(y)) < 4:
        print("WARNING: not all four activities are recorded yet")
        if not args.force:
            print("refusing to overwrite the deployed models; pass --force")
            return 1
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for name, (names, depth, leaf) in MODELS.items():
        X = matrix(feats, names)
        y_arr = np.array(y)
        if len(set(groups)) >= 2:
            pred = heldout_predictions(X, y_arr, groups, depth, leaf)
            report("%s leave-one-session-out" % name, y_arr, pred)
            heldout = accuracy_score(y_arr, pred)
        else:
            print("only one session: no held-out estimate")
            heldout = None
        if args.peer and len(set(persons)) >= 2:
            ppl = np.array(persons)
            for who in sorted(set(persons)):
                tr, te = ppl != who, ppl == who
                clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X[tr], y_arr[tr])
                report("%s tested on %s (trained on everyone else)" % (name, who), y_arr[te], clf.predict(X[te]))
        clf, X = fit(feats, y, names, depth, leaf)
        spec = export_tree(clf, names)
        spec["trained_on_windows"] = len(y)
        spec["train_accuracy"] = round(float(accuracy_score(y, clf.predict(X))), 4)
        spec["heldout_accuracy"] = None if heldout is None else round(float(heldout), 4)
        out = Path(args.out) / ("%s.json" % name)
        out.write_text(json.dumps(spec, indent=1))
        print("wrote", out, "(depth %d, %d leaves)" % (clf.get_depth(), clf.get_n_leaves()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
