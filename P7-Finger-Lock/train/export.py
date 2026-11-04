"""models/fingers.pt -> models/fingers.json: the weights as plain lists for the board's numpy engine.

    .venv/bin/python train/export.py [--pt models/fingers.pt] [--out models/fingers.json]
"""
import argparse
import json
from pathlib import Path

import torch

from model import CLASSES, FingerNet, layer_specs

ROOT = Path(__file__).resolve().parents[1]


def to_spec(model, mean, std, extra=None):
    spec = {"input": {"size": model.size, "mean": float(mean), "std": float(std)}, "classes": CLASSES,
            "channels": list(model.channels), "layers": layer_specs(model)}
    spec.update(extra or {})
    return spec


def save_checkpoint(model, mean, std, path, extra=None):
    torch.save({"state_dict": model.state_dict(), "size": model.size, "channels": list(model.channels),
                "mean": float(mean), "std": float(std), **(extra or {})}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = FingerNet(size=ck["size"], channels=tuple(ck["channels"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def export(pt, out):
    model, ck = load_checkpoint(pt)
    spec = to_spec(model, ck["mean"], ck["std"], {"trained": ck.get("trained"), "val_accuracy": ck.get("val_accuracy")})
    with open(out, "w") as f:
        json.dump(spec, f)
    return spec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=ROOT / "models" / "fingers.pt")
    ap.add_argument("--out", default=ROOT / "models" / "fingers.json")
    a = ap.parse_args()
    spec = export(a.pt, a.out)
    print(f"wrote {a.out}: {len(spec['layers'])} layers, input {spec['input']['size']}")
