"""The Notion figures: both tree diagrams, the two confusion matrices side by side, and the feature-space picture
that shows why Model 1 cannot see biking.   .venv/bin/python train/plots.py   -> docs/figures/*.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "app" / "python"))
from features import MODEL2_FEATURES  # noqa: E402
from train import LABELS, MODELS, dataset, fit, heldout_predictions, predict  # noqa: E402
from features import MODEL2_FEATURES, STILL_GYRO  # noqa: E402

COLORS = {"STAND": "#888888", "WALK": "#16a806", "RUN": "#ff7800", "BIKE": "#0093e7"}


def make_figures(paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats, y, groups, _ = dataset(paths)
    y_arr = np.array(y)
    written = []

    # 1 + 2: the trees, fit on everything (what the board runs)
    for name, (names, depth, leaf, gate) in MODELS.items():
        clf, _ = fit(feats, y, names, depth, leaf, gate)
        if name == "model2":
            clf2 = clf
        fig, ax = plt.subplots(figsize=(max(14, 1.3 * clf.get_n_leaves()), 2 + 1.8 * clf.get_depth()))
        plot_tree(clf, feature_names=names, class_names=list(clf.classes_), filled=True, rounded=True,
                  impurity=False, proportion=False, fontsize=7, ax=ax)
        if gate:
            ax.set_title("%s (deployed): root split gyro_rms <= %.0f deg/s -> STAND (leg not moving); otherwise this tree"
                         % (name, STILL_GYRO))
        else:
            ax.set_title("%s (tree deployed: fit on all sessions)" % name)
        fig.text(0.01, 0.01, "class order in the boxes: " + ", ".join(clf.classes_), fontsize=7)
        p = out_dir / ("tree_%s.png" % name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 3: confusion matrices on held-out sessions (or on everything when there is only one session)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, (names, depth, leaf, gate)) in zip(axes, MODELS.items()):
        if len(set(groups)) >= 2:
            pred = heldout_predictions(feats, y_arr, groups, names, depth, leaf, gate)
            suffix = "leave-one-session-out"
        else:
            clf, _ = fit(feats, y_arr, names, depth, leaf, gate)
            pred = predict(clf, feats, names, gate)
            suffix = "(training data, one session only)"
        cm = confusion_matrix(y_arr, pred, labels=LABELS)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "%d\n%.0f%%" % (cm[i, j], 100 * norm[i, j]), ha="center", va="center",
                        color="black" if norm[i, j] < 0.5 else "white")
        ax.set_xticks(range(4), LABELS)
        ax.set_yticks(range(4), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc = np.trace(cm) / max(1, cm.sum())
        ax.set_title("%s: %.0f%% %s" % (name, 100 * acc, suffix))
    fig.tight_layout()
    p = out_dir / "confusion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 4: feature space. Left: peak_x alone (all Model 1 ever sees). Right: peak_x against Model 2's first learned split.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rng = np.random.default_rng(0)
    root = MODEL2_FEATURES[clf2.tree_.feature[0]]
    for lab in LABELS:
        m = y_arr == lab
        px = np.array([f["peak_x"] for f, keep in zip(feats, m) if keep])
        gr = np.array([f[root] for f, keep in zip(feats, m) if keep])
        if len(px) == 0:
            continue
        axes[0].scatter(px, rng.uniform(-0.3, 0.3, len(px)) + LABELS.index(lab), s=12, c=COLORS[lab], label=lab)
        axes[1].scatter(px, gr, s=12, c=COLORS[lab], label=lab)
    axes[0].set_yticks(range(4), LABELS)
    axes[0].set_xlabel("peak thigh-axis acceleration (g)")
    axes[0].set_title("Model 1 sees only peak thigh-axis g")
    axes[1].set_xlabel("peak thigh-axis acceleration (g)")
    axes[1].set_ylabel(root)
    axes[1].set_title("Model 2 also sees %s (after the motion gate)" % root)
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
