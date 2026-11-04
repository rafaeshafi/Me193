"""Figures for the write-up: training curves, confusion matrix, sample grid, augmentation grid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import augment

CLASS_NAMES = ["0 (none)", "1", "2", "3", "4", "5"]


def curves(history, path, title="FingerNet training"):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    if any(h.get("val_loss") is not None for h in history):
        ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("cross-entropy loss"), ax[0].legend(), ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [100 * h["train_acc"] for h in history], label="train")
    if any(h.get("val_acc") is not None for h in history):
        ax[1].plot(ep, [100 * h["val_acc"] for h in history], label="validation")
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("accuracy (%)"), ax[1].set_ylim(0, 101), ax[1].legend(), ax[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(cm, path, title="Confusion (held-out sessions)"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(6), CLASS_NAMES), ax.set_yticks(range(6), CLASS_NAMES)
    ax.set_xlabel("predicted fingers"), ax.set_ylabel("true fingers"), ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def samples(xs, labels, path, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(6, per_class, figsize=(per_class * 1.4, 6 * 1.5))
    for label in range(6):
        idx = np.flatnonzero(labels == label)
        pick = rng.choice(idx, min(per_class, len(idx)), replace=False) if len(idx) else []
        for j in range(per_class):
            ax = axes[label, j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(xs[pick[j]], cmap="gray", vmin=0, vmax=1)
            if j == 0:
                ax.set_title(f"{label} fingers", fontsize=9, loc="left")
    fig.suptitle("What the net sees (64x64 grayscale crops)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def augmentations(x, path, n=8, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 1.4, 1.8))
    axes[0].imshow(x, cmap="gray", vmin=0, vmax=1), axes[0].set_title("original", fontsize=8), axes[0].axis("off")
    for ax in axes[1:]:
        ax.imshow(augment(x, rng), cmap="gray", vmin=0, vmax=1), ax.axis("off")
    fig.suptitle("One picture, eight augmented copies (rotate, shift, scale, flip, light)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confusion_matrix(y_true, y_pred, n=6):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm
