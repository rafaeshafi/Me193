"""Train FingerNet on data/raw (or synthetic hands), make the figures, write the results and export for the board.

    .venv/bin/python train/train.py                   # real pictures in data/raw, 30 epochs
    .venv/bin/python train/train.py --synth           # synthetic hands (pipeline check, no camera needed)
    .venv/bin/python train/train.py --no-aug          # same net without augmentation (for the comparison)
    .venv/bin/python train/train.py --ablate          # both runs, one results table
    options: --epochs N --val-session NAME [NAME ...] --seed N --out-tag TAG

Outputs: models/fingers.pt, models/fingers.json, models/meta.json, docs/figures/*.png, docs/results.md.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plots  # noqa: E402
import synth  # noqa: E402
from dataset import split_within_session, Fingers, load_gray, scan, split_by_session, stats  # noqa: E402
from export import save_checkpoint, to_spec  # noqa: E402
from model import FingerNet  # noqa: E402

MODELS, FIGS, DOCS = ROOT / "models", ROOT / "docs" / "figures", ROOT / "docs"


def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, loss_sum, preds, trues = 0, 0, 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss_sum += loss_fn(out, y).item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += len(y)
            preds += p.tolist()
            trues += y.tolist()
    return loss_sum / max(total, 1), correct / max(total, 1), np.asarray(trues), np.asarray(preds)


def fit(xs_tr, y_tr, xs_va, y_va, size, epochs, augmented, seed, lr=1e-3, batch=64, log=print, strength=1.0):
    torch.manual_seed(seed)
    mean, std = stats(xs_tr)
    tr = DataLoader(Fingers(xs_tr, y_tr, mean, std, augmented, seed, strength), batch_size=batch, shuffle=True)
    va = DataLoader(Fingers(xs_va, y_va, mean, std), batch_size=256) if len(xs_va) else None
    model = FingerNet(size=size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # a little decay against memorising one burst
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, correct, total = time.time(), 0.0, 0, 0
        for x, y in tr:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        h = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total, "val_loss": None, "val_acc": None}
        if va is not None:
            h["val_loss"], h["val_acc"], _, _ = evaluate(model, va, loss_fn)
        history.append(h)
        log(f"epoch {epoch:3d}  loss {h['train_loss']:.3f}  acc {100 * h['train_acc']:5.1f}%"
            + (f"  val loss {h['val_loss']:.3f}  val acc {100 * h['val_acc']:5.1f}%" if va is not None else "")
            + f"  ({time.time() - t0:.1f} s)")
    cm = None
    if va is not None:
        _, _, trues, preds = evaluate(model, va, loss_fn)
        cm = plots.confusion_matrix(trues, preds)
    return model, mean, std, history, cm


def results_md(runs, n_train, n_val, val_sessions, n_params, synthetic):
    lines = ["# Results", "", f"*Written by `train/train.py` on {date.today()}. "
             + ("**Synthetic hands** (pipeline check), not real pictures.*" if synthetic else "Real webcam pictures.*"), "",
             f"- training pictures: {n_train}, validation pictures: {n_val} (held-out sessions: {', '.join(val_sessions) or 'none'})",
             f"- FingerNet parameters: {n_params:,}", "", "| run | epochs | final train acc | best val acc | final val acc | final val loss |",
             "|---|---|---|---|---|---|"]
    for name, h in runs.items():
        last, best = h[-1], max((x["val_acc"] or 0) for x in h)
        lines.append(f"| {name} | {len(h)} | {100 * last['train_acc']:.1f} % | {100 * best:.1f} % | "
                     f"{100 * (last['val_acc'] or 0):.1f} % | {last['val_loss'] if last['val_loss'] is None else round(last['val_loss'], 3)} |")
    cm_name = "confusion.png"
    lines += ["", "Per-class results of the deployed run are in the confusion matrix (`docs/figures/" + cm_name + "`); the curves are in "
              "`docs/figures/train_curves.png`."]
    return "\n".join(lines) + "\n"


def per_class(cm):
    return {str(i): (round(cm[i, i] / cm[i].sum(), 3) if cm[i].sum() else None) for i in range(cm.shape[0])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw")
    ap.add_argument("--synth", action="store_true", help="train on generated hands instead of data/raw")
    ap.add_argument("--synth-per-class", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="train with and without augmentation; deploy the augmented one")
    ap.add_argument("--val-session", nargs="*", default=None)
    ap.add_argument("--split", choices=("session", "within"), default="session",
                    help="session: hold out whole capture sessions (honest); within: the last quarter of every burst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug-strength", type=float, default=1.0, help="scale of the augmentation amounts (0.5 = milder)")
    ap.add_argument("--sessions", nargs="*", default=None, help="train on these capture sessions only (default: all)")
    ap.add_argument("--deploy", choices=("augmented", "best"), default="augmented",
                    help="with --ablate: export the augmented run (default) or whichever run validates best")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out-dir", default=None, help="write models/figures/results somewhere else (tests)")
    a = ap.parse_args(argv)
    out_root = Path(a.out_dir) if a.out_dir else ROOT
    models, figs, docs = out_root / "models", out_root / "docs" / "figures", out_root / "docs"
    for d in (models, figs, docs):
        d.mkdir(parents=True, exist_ok=True)

    if a.synth:
        imgs, labels, sessions = synth.dataset(a.synth_per_class, a.seed, sessions=3)
        from lock.preprocess import to_input
        xs = np.stack([to_input(im, a.size) for im in imgs])
        items = [(None, int(l), str(s)) for l, s in zip(labels, sessions)]
    else:
        items = scan(a.data)
        if a.sessions:
            items = [it for it in items if it[2] in set(a.sessions)]
        if not items:
            sys.exit(f"no pictures under {a.data}: run tools/capture.py first (or --synth)")
        xs = load_gray(items, a.size)
        labels = np.asarray([l for _, l, _ in items])
    idx = np.arange(len(items))
    if a.split == "within":
        tr_items, va_items, val_sessions = split_within_session(items)
        va_ids = {id(it) for it in va_items}
        tr_idx = [i for i in idx if id(items[i]) not in va_ids]
        va_idx = [i for i in idx if id(items[i]) in va_ids]
    else:
        tr_items, va_items, val_sessions = split_by_session(items, a.val_session)
        tr_idx = [i for i in idx if items[i][2] not in set(val_sessions)]
        va_idx = [i for i in idx if items[i][2] in set(val_sessions)]
    xs_tr, y_tr, xs_va, y_va = xs[tr_idx], labels[tr_idx], xs[va_idx], labels[va_idx]
    print(f"{len(tr_idx)} training / {len(va_idx)} validation pictures; validation sessions: {val_sessions}")
    counts = {int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}
    print("per class:", counts)

    runs, deployed = {}, None
    variants = [("augmented", True), ("no augmentation", False)] if a.ablate else [("no augmentation", False)] if a.no_aug else [("augmented", True)]
    for name, aug in variants:
        print(f"--- {name} ---")
        model, mean, std, history, cm = fit(xs_tr, y_tr, xs_va, y_va, a.size, a.epochs, aug, a.seed, strength=a.aug_strength)
        runs[name] = history
        better = deployed is None or (aug if a.deploy == "augmented" else
                                      (history[-1]["val_acc"] or 0) > (deployed[4][-1]["val_acc"] or 0))
        if better:
            deployed = (name, model, mean, std, history, cm)
        plots.curves(history, figs / ("train_curves.png" if aug or not a.ablate else "train_curves_noaug.png"),
                     f"FingerNet training ({name})")
        if cm is not None:
            plots.confusion(cm, figs / ("confusion.png" if aug or not a.ablate else "confusion_noaug.png"))

    name, model, mean, std, history, cm = deployed
    plots.samples(xs, labels, figs / "samples.png")
    plots.augmentations(xs[int(np.flatnonzero(labels == 3)[0]) if (labels == 3).any() else 0], figs / "augment.png")
    val_acc = history[-1]["val_acc"]
    extra = {"trained": str(date.today()), "val_accuracy": val_acc, "run": name}
    save_checkpoint(model, mean, std, models / "fingers.pt", extra)
    with open(models / "fingers.json", "w") as f:
        json.dump(to_spec(model, mean, std, extra), f)
    meta = {"trained": extra["trained"], "run": name, "synthetic": a.synth, "epochs": a.epochs, "n_train": len(tr_idx),
            "n_val": len(va_idx), "val_sessions": val_sessions, "per_class_counts": counts, "n_params": model.n_params(),
            "val_accuracy": val_acc, "per_class_val_accuracy": per_class(cm) if cm is not None else None,
            "history": {k: v for k, v in runs.items()}, "input": {"size": a.size, "mean": mean, "std": std}}
    with open(models / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    (docs / "results.md").write_text(results_md(runs, len(tr_idx), len(va_idx), val_sessions, model.n_params(), a.synth))
    print(f"deployed run: {name}; val acc {val_acc}; wrote {models / 'fingers.json'}")
    return meta


if __name__ == "__main__":
    main()
