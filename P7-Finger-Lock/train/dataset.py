"""Pictures on disk -> training tensors. data/raw/<label>/<session>-<n>.png, label 0..5, session = the capture run.

Validation holds out whole sessions (neighbouring frames of one burst are near-duplicates, so a random split would
flatter the number). Augmentation runs on the fly with OpenCV warps: rotation, shift, scale, flip, brightness,
contrast, noise. The input is grayscale 64x64 in 0..1, standardised with the training mean and std.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lock.preprocess import to_input

NAME = re.compile(r"^(?P<session>.+?)-(?P<n>\d+)\.(png|jpg|jpeg)$")


def scan(root):
    """-> list of (path, label, session) for every picture under root/<label>/."""
    items = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        label = int(d.name)
        for p in sorted(d.iterdir()):
            m = NAME.match(p.name)
            if m:
                items.append((p, label, m.group("session")))
    return items


def load_gray(items, size):
    xs = np.empty((len(items), size, size), dtype=np.float32)
    for i, (p, _, _) in enumerate(items):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"unreadable picture {p}")
        xs[i] = to_input(img, size)
    return xs


def split_by_session(items, val_sessions=None, val_fraction=0.25, seed=0):
    """Hold out whole sessions. Named sessions if given, else the last quarter of them (sorted by name = by time)."""
    sessions = sorted({s for _, _, s in items})
    if val_sessions is None:
        k = max(1, int(round(len(sessions) * val_fraction))) if len(sessions) > 1 else 0
        val_sessions = sessions[len(sessions) - k:] if k else []
    val_sessions = set(val_sessions)
    train = [it for it in items if it[2] not in val_sessions]
    val = [it for it in items if it[2] in val_sessions]
    return train, val, sorted(val_sessions)


def augment(x, rng, max_rot=15, max_shift=0.08, scale=(0.85, 1.15), flip=True, brightness=0.25, contrast=(0.8, 1.25),
            noise=0.02):
    """x (size, size) float 0..1 -> a randomly rotated, shifted, scaled, flipped, re-lit copy."""
    size = x.shape[0]
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale)
    tx, ty = rng.uniform(-max_shift, max_shift, 2) * size
    m = cv2.getRotationMatrix2D((size / 2, size / 2), ang, sc)
    m[:, 2] += (tx, ty)
    y = cv2.warpAffine(x, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if flip and rng.random() < 0.5:
        y = y[:, ::-1]
    y = (y - 0.5) * rng.uniform(*contrast) + 0.5 + rng.uniform(-brightness, brightness)
    if noise:
        y = y + rng.normal(0, noise, y.shape).astype(np.float32)
    return np.clip(y, 0, 1).astype(np.float32)


class Fingers(Dataset):
    def __init__(self, xs, labels, mean, std, augmented=False, seed=0):
        self.xs, self.labels, self.mean, self.std, self.augmented = xs, np.asarray(labels), mean, std, augmented
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x = self.xs[i]
        if self.augmented:
            x = augment(x, self.rng)
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x))[None], int(self.labels[i])


def stats(xs):
    return float(xs.mean()), float(xs.std() + 1e-6)
