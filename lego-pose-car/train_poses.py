"""Train a pose classifier on the examples recorded by collect_poses.py.

    python train_poses.py

Reads pose_data.csv, fits a classifier, reports how well it actually did, and
writes pose_model.joblib for pose_car.py --model to drive from.

The model is deliberately small. Each sample is only 18 numbers describing the
shape of your upper body, the classes are a handful of poses you chose to be
easy to tell apart, and there are a few hundred examples. Logistic regression
suits that: it trains instantly, it reports calibrated probabilities (which is
what lets the car refuse to act on an uncertain frame), and when it gets
something wrong you can see why.
"""

import argparse
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pose_features as pf


def load_dataset(path):
    """Read the CSV, refusing it if its columns don't match today's features."""
    if not path.exists():
        raise SystemExit(f"No data at {path}. Record some first:\n"
                         f"  my_env/bin/python collect_poses.py")

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"{path.name} is empty.")
        expected = ["label", *pf.FEATURE_NAMES]
        if header != expected:
            # Silently training on mismatched columns would produce a model that
            # loads fine and predicts nonsense, so stop here instead.
            raise SystemExit(
                f"{path.name} was recorded with a different feature set.\n"
                f"  file:     {len(header) - 1} features\n"
                f"  expected: {len(expected) - 1} features\n"
                f"Delete it and re-record, or check pose_features.py.")
        labels, rows = [], []
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
            rows.append([float(value) for value in row[1:]])

    return np.array(rows, dtype=np.float64), np.array(labels)


def print_confusion(matrix, classes):
    width = max(len(name) for name in classes) + 1
    print(f"\n{'actual \\ predicted':>{width + 18}}")
    print(" " * (width + 2) + "  ".join(f"{name[:6]:>6}" for name in classes))
    for name, row in zip(classes, matrix):
        cells = "  ".join(f"{count:>6}" for count in row)
        print(f"  {name:<{width}}{cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=lambda p: pf.HERE / p, default=pf.DATA_PATH)
    parser.add_argument("--out", type=lambda p: pf.HERE / p, default=pf.CLASSIFIER_PATH)
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="fraction held out for testing (default 0.25)")
    args = parser.parse_args()

    features, labels = load_dataset(args.data)
    counts = Counter(labels)
    classes = sorted(counts)

    print(f"{len(labels)} samples, {len(classes)} classes, {features.shape[1]} features each")
    for name in classes:
        print(f"  {name:<10} {counts[name]:>5}")

    if len(classes) < 2:
        raise SystemExit("\nNeed at least 2 classes to train a classifier.")
    smallest = min(counts.values())
    if smallest < 10:
        raise SystemExit(f"\nToo few samples for '{min(counts, key=counts.get)}' "
                         f"({smallest}). Record more with collect_poses.py.")

    unknown = [name for name in classes if name not in pf.CLASS_SPEEDS]
    if unknown:
        print(f"\nNote: {', '.join(unknown)} have no entry in CLASS_SPEEDS "
              f"(pose_features.py), so the car won't know what to do with them.")

    # Standardize, then fit. Scaling matters here: features are offsets in
    # shoulder-widths, and the ones further from the shoulders (wrists) swing
    # over a much wider range than the ones near them (nose).
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, stratify=labels, random_state=0)
    model.fit(x_train, y_train)

    print(f"\nHeld-out test set ({len(y_test)} samples):")
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, zero_division=0))
    print_confusion(confusion_matrix(y_test, predictions, labels=classes), classes)

    # One split of a few hundred samples is a noisy estimate, so cross-validate
    # for a number worth quoting.
    folds = min(5, smallest)
    scores = cross_val_score(model, features, labels,
                             cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    print(f"\n{folds}-fold cross-validation: "
          f"{scores.mean():.1%} accuracy (+/- {scores.std():.1%})")
    if scores.mean() > 0.995:
        print("  Near-perfect usually means the classes are very easy to tell apart,\n"
              "  and sometimes means your samples are too similar to each other.\n"
              "  Record from a few distances and angles to find out which.")

    # Refit on everything for the model that actually ships: the split above was
    # for measuring, and there is no reason to throw away a quarter of the data.
    model.fit(features, labels)
    joblib.dump({"pipeline": model,
                 "classes": list(model.classes_),
                 "feature_names": list(pf.FEATURE_NAMES)}, args.out)
    print(f"\nSaved {args.out.name} ({len(labels)} samples, refit on all of them)")
    print(f"Drive with it:  my_env/bin/python pose_car.py --model {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
