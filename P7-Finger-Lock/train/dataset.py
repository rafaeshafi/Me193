"""Pictures on disk -> training tensors. data/raw/<label>/<session>-<n>.png, label 0..5, session = the capture run.

Validation holds out whole sessions (neighbouring frames of one burst are near-duplicates, so a random split would
flatter the number). Augmentation runs on the fly with OpenCV warps: rotation, shift, scale, flip, brightness,
contrast, noise, a hidden patch. The input is a grayscale square (config.json input_size) in 0..1, standardised with the training mean and std.
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


def split_within_session(items, val_fraction=0.25):
    """Hold out the last quarter (by frame number) of every label's every burst. For when a label exists in one
    session only, so whole sessions cannot be held out; frames neighbour their training frames, so the accuracy
    this gives is optimistic."""
    groups = {}
    for it in items:
        groups.setdefault((it[1], it[2]), []).append(it)
    train, val = [], []
    for g in groups.values():
        g.sort(key=lambda it: int(NAME.match(Path(it[0]).name).group("n")) if it[0] is not None else 0)
        k = max(1, int(round(len(g) * val_fraction))) if len(g) > 3 else 0
        train += g[:len(g) - k] if k else g
        val += g[len(g) - k:] if k else []
    return train, val, ["last 25 % of every burst (same sessions as training)"]


def augment(x, rng, max_rot=30, max_shift=0.12, scale=(0.75, 1.25), flip=True, brightness=0.25, contrast=(0.8, 1.25),
            noise=0.02, cutout=0.3, strength=1.0):
    """x (size, size) float 0..1 -> a randomly rotated, shifted, scaled, flipped, re-lit copy with, half the time, a
    patch hidden. strength scales every amount (1 = the defaults, 0.5 = half as wild); the flip stays. The wide
    rotation, shift and scale are there because one capture burst holds the hand in one pose: live, a finger pointing
    another way read as a different count. The hidden patch stops the net leaning on the watch, the sleeve or a shadow."""
    size = x.shape[0]
    max_rot, max_shift, brightness, noise = max_rot * strength, max_shift * strength, brightness * strength, noise * strength
    cutout = cutout * strength
    scale = (1 - (1 - scale[0]) * strength, 1 + (scale[1] - 1) * strength)
    contrast = (1 - (1 - contrast[0]) * strength, 1 + (contrast[1] - 1) * strength)
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale)
    tx, ty = rng.uniform(-max_shift, max_shift, 2) * size
    m = cv2.getRotationMatrix2D((size / 2, size / 2), ang, sc)
    m[:, 2] += (tx, ty)
    y = cv2.warpAffine(x, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if flip and rng.random() < 0.5:
        y = y[:, ::-1]
    if cutout and rng.random() < 0.5:
        side = max(1, int(size * rng.uniform(0.1, cutout)))
        cx, cy = rng.integers(0, size - side + 1, 2)
        y = y.copy(); y[cy:cy + side, cx:cx + side] = float(y.mean())
    y = (y - 0.5) * rng.uniform(*contrast) + 0.5 + rng.uniform(-brightness, brightness)
    if noise:
        y = y + rng.normal(0, noise, y.shape).astype(np.float32)
    return np.clip(y, 0, 1).astype(np.float32)


class Fingers(Dataset):
    def __init__(self, xs, labels, mean, std, augmented=False, seed=0, strength=1.0):
        self.xs, self.labels, self.mean, self.std, self.augmented = xs, np.asarray(labels), mean, std, augmented
        self.rng, self.strength = np.random.default_rng(seed), strength

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x = self.xs[i]
        if self.augmented:
            x = augment(x, self.rng, strength=self.strength)
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x))[None], int(self.labels[i])


def stats(xs):
    return float(xs.mean()), float(xs.std() + 1e-6)
