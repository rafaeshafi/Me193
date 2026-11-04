"""The trained net in numpy, for the board (no PyTorch there). Reads models/fingers.json written by train/export.py.

Layers: conv (3x3, pad 1, stride 1, as im2col matmul), relu, maxpool (2x2), flatten, linear. The forward pass matches
PyTorch to float32 rounding (tests/test_cnn.py checks 1e-4).
"""
import json

import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def im2col(x, k, pad):
    """x (C, H, W) -> (H*W, C*k*k) rows of receptive fields, for stride 1 and 'same' padding."""
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((c, k, k, h, w), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, i, j] = xp[:, i:i + h, j:j + w]
    return cols.reshape(c * k * k, h * w).T


def conv2d(x, w, b):
    """x (C, H, W), w (O, C, k, k), b (O,) -> (O, H, W)."""
    o, c, k, _ = w.shape
    cols = im2col(x, k, k // 2)
    out = cols @ w.reshape(o, c * k * k).T + b
    return out.T.reshape(o, x.shape[1], x.shape[2])


def maxpool2(x):
    c, h, w = x.shape
    return x[:, :h - h % 2, :w - w % 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def relu(x):
    return np.maximum(x, 0)


class NumpyNet:
    def __init__(self, spec):
        self.spec = spec
        self.layers = []
        for layer in spec["layers"]:
            t = layer["type"]
            if t in ("conv", "linear"):
                self.layers.append((t, np.asarray(layer["w"], dtype=np.float32), np.asarray(layer["b"], dtype=np.float32)))
            else:
                self.layers.append((t, None, None))
        self.mean, self.std = float(spec["input"]["mean"]), float(spec["input"]["std"])
        self.size = int(spec["input"]["size"])
        self.classes = list(spec["classes"])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def logits(self, x):
        """x: (size, size) float in 0..1 (the preprocess output) -> the six scores."""
        x = ((np.asarray(x, dtype=np.float32) - self.mean) / self.std)[None]
        for t, w, b in self.layers:
            if t == "conv":
                x = conv2d(x, w, b)
            elif t == "relu":
                x = relu(x)
            elif t == "maxpool":
                x = maxpool2(x)
            elif t == "flatten":
                x = x.reshape(-1)
            elif t == "linear":
                x = w @ x + b
            elif t == "dropout":
                pass  # identity at inference
            else:
                raise ValueError(f"unknown layer {t}")
        return x

    def predict(self, x):
        """-> (class, probabilities list)."""
        p = softmax(self.logits(x))
        return int(p.argmax()), [float(v) for v in p]

    def n_params(self):
        return sum(w.size + b.size for t, w, b in self.layers if w is not None)
