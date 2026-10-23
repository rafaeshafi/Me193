"""The Notion figures: both tree diagrams, the two confusion matrices side by side, and the feature-space picture
that shows why Model 1 cannot see biking.   .venv/bin/python train/plots.py   -> docs/figures/*.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from train import LABELS, MODELS, dataset, fit, matrix  # noqa: E402

COLORS = {"STAND": "#888888", "WALK": "#16a806", "RUN": "#ff7800", "BIKE": "#0093e7"}


def make_figures(paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats, y, groups, _ = dataset(paths)
    y_arr = np.array(y)
    written = []

    # 1 + 2: the trees, fit on everything (what the board runs)
    for name, (names, depth, leaf) in MODELS.items():
        clf, _ = fit(feats, y, names, depth, leaf)
        fig, ax = plt.subplots(figsize=(14 if name == "model2" else 8, 6))
        plot_tree(clf, feature_names=names, class_names=list(clf.classes_), filled=True, rounded=True,
                  impurity=False, proportion=True, fontsize=9, ax=ax)
        ax.set_title("%s: %s" % (name, "peak thigh-axis g only" if name == "model1" else "all six IMU channels"))
        p = out_dir / ("tree_%s.png" % name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 3: confusion matrices on held-out sessions (or on everything when there is only one session)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, (names, depth, leaf)) in zip(axes, MODELS.items()):
        X = matrix(feats, names)
        if len(set(groups)) >= 2:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y_arr, groups))
        else:
            tr = te = np.arange(len(y_arr))
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X[tr], y_arr[tr])
        cm = confusion_matrix(y_arr[te], clf.predict(X[te]), labels=LABELS)
        ax.imshow(cm, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        ax.set_xticks(range(4), LABELS)
        ax.set_yticks(range(4), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc = np.trace(cm) / max(1, cm.sum())
        ax.set_title("%s: %.0f%% on held-out windows" % (name, 100 * acc))
    fig.tight_layout()
    p = out_dir / "confusion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 4: feature space. Left: peak_x alone (all Model 1 ever sees). Right: peak_x against gyro RMS.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rng = np.random.default_rng(0)
    for lab in LABELS:
        m = y_arr == lab
        px = np.array([f["peak_x"] for f, keep in zip(feats, m) if keep])
        gr = np.array([f["gyro_rms"] for f, keep in zip(feats, m) if keep])
        if len(px) == 0:
            continue
        axes[0].scatter(px, rng.uniform(-0.3, 0.3, len(px)) + LABELS.index(lab), s=12, c=COLORS[lab], label=lab)
        axes[1].scatter(px, gr, s=12, c=COLORS[lab], label=lab)
    axes[0].set_yticks(range(4), LABELS)
    axes[0].set_xlabel("peak thigh-axis acceleration (g)")
    axes[0].set_title("Model 1 sees only this axis")
    axes[1].set_xlabel("peak thigh-axis acceleration (g)")
    axes[1].set_ylabel("gyro RMS (deg/s)")
    axes[1].set_title("Model 2 also sees rotation: biking separates")
    axes[1].legend()
    fig.tight_layout()
    p = out_dir / "features.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


def main():
    paths = sorted(p for p in (ROOT / "data").glob("*.csv") if not p.name.startswith("rolling-"))
    if not paths:
        print("no recordings in data/ (run scripts/pull_data.sh first)")
        return 1
    for p in make_figures(paths, ROOT / "docs" / "figures"):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
