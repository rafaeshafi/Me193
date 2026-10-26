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
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "python"))
from features import MODEL1_FEATURES, MODEL2_FEATURES, RATE, STILL_GYRO, compute  # noqa: E402

LABELS = ["STAND", "WALK", "RUN", "BIKE"]
WINDOW_S, HOP_S = 2.0, 0.5
# name: (features, depth of the learned tree, min samples per leaf, motion gate). With the gate, the exported tree's root
# is the fixed split gyro_rms <= STILL_GYRO -> STAND and the learned tree only sorts the moving classes: a still leg is
# decided by motion, never by a lean or a tilt that happened to separate standing from walking in one session.
MODELS = {"model1": (MODEL1_FEATURES, 3, 5, False), "model2": (MODEL2_FEATURES, 4, 5, True)}
CCP_ALPHA = 0.005


def load_session(path):
    """-> [(t, label, (ax, ay, az, gx, gy, gz)), ...] in file order."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:  # a session cut short (power pulled, file pulled while open) ends in a truncated row: skip it
                rows.append((float(r["t"]), r["label"], tuple(float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz"))))
            except (TypeError, ValueError):
                continue
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


def moving(feats):
    return np.array([f["gyro_rms"] > STILL_GYRO for f in feats])


def fit(feats, y, names, depth, leaf, gate=False):
    """-> (clf, X). With the gate the tree is fit on the moving windows only (X still covers every window)."""
    X = matrix(feats, names)
    keep = moving(feats) if gate else np.ones(len(feats), dtype=bool)
    # ccp_alpha prunes splits whose two leaves carry the same label (cost-complexity pruning), nothing else at this size
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, ccp_alpha=CCP_ALPHA, random_state=0).fit(X[keep], np.array(y)[keep])
    return clf, X


def predict(clf, feats, names, gate=False):
    """Labels for every window, applying the motion gate in front of the tree when the model has one."""
    out = clf.predict(matrix(feats, names)).astype(object)
    if gate:
        out[~moving(feats)] = "STAND"
    return out


def export_tree(clf, names, gate=False):
    t = clf.tree_

    def node(i):
        if t.children_left[i] == -1:
            return {"label": str(clf.classes_[int(np.argmax(t.value[i]))])}
        return {"f": names[t.feature[i]], "t": float(t.threshold[i]),
                "l": node(t.children_left[i]), "r": node(t.children_right[i])}

    root = node(0)
    classes = [str(c) for c in clf.classes_]
    if gate:
        root = {"f": "gyro_rms", "t": STILL_GYRO, "l": {"label": "STAND"}, "r": root}
        classes = sorted(set(classes) | {"STAND"})
    return {"features": list(names), "classes": classes, "root": root}


def heldout_predictions(feats, y, groups, names, depth, leaf, gate=False):
    """Every window predicted by a tree that never saw its own session (leave-one-session-out), pooled."""
    y = np.array(y)
    groups = np.array(groups)
    pred = np.empty(len(y), dtype=object)
    for held in sorted(set(groups)):
        tr = groups != held
        clf, _ = fit([f for f, k in zip(feats, tr) if k], y[tr], names, depth, leaf, gate)
        te = [f for f, k in zip(feats, tr) if not k]
        pred[~tr] = predict(clf, te, names, gate)
    return pred


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

    still = [f["gyro_rms"] for f, lab in zip(feats, y) if lab == "STAND"]
    active = [f["gyro_rms"] for f, lab in zip(feats, y) if lab != "STAND"]
    if still and active and (max(still) > STILL_GYRO or min(active) < STILL_GYRO):
        print("WARNING: motion gate %.0f deg/s is not clean: STAND reaches %.0f, moving starts at %.0f"
              % (STILL_GYRO, max(still), min(active)))
    for name, (names, depth, leaf, gate) in MODELS.items():
        y_arr = np.array(y)
        if len(set(groups)) >= 2:
            pred = heldout_predictions(feats, y_arr, groups, names, depth, leaf, gate)
            report("%s leave-one-session-out" % name, y_arr, pred)
            heldout = accuracy_score(y_arr, pred)
        else:
            print("only one session: no held-out estimate")
            heldout = None
        if args.peer and len(set(persons)) >= 2:
            ppl = np.array(persons)
            for who in sorted(set(persons)):
                tr = ppl != who
                clf, _ = fit([f for f, k in zip(feats, tr) if k], y_arr[tr], names, depth, leaf, gate)
                te = [f for f, k in zip(feats, tr) if not k]
                report("%s tested on %s (trained on everyone else)" % (name, who), y_arr[~tr], predict(clf, te, names, gate))
        clf, X = fit(feats, y, names, depth, leaf, gate)
        spec = export_tree(clf, names, gate)
        spec["trained_on_windows"] = len(y)
        spec["train_accuracy"] = round(float(accuracy_score(y, predict(clf, feats, names, gate))), 4)
        spec["heldout_accuracy"] = None if heldout is None else round(float(heldout), 4)
        out = Path(args.out) / ("%s.json" % name)
        out.write_text(json.dumps(spec, indent=1))
        print("wrote", out, "(depth %d, %d leaves%s)" % (clf.get_depth() + (1 if gate else 0), clf.get_n_leaves() + (1 if gate else 0), ", motion gate first" if gate else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
