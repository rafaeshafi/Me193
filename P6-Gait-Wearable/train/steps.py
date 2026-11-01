"""Train StepNet from the counted takes and measure how good it is.

  .venv/bin/python train/steps.py                 # fit on data/*-steps-*-n<N>*.csv -> models/stepnet.json
  .venv/bin/python train/steps.py --report        # + leave-one-recording-out, 5 seeds, demo takes, figures, docs/steps_results.md

A take's true count is in its file name (recorder.parse_counted). Takes whose `who` is "demo" or ends in "-demo" are
never trained on: they are the fresh takes the video shows and are reported separately. A take with a break in t over
0.8 s (samples lost) is skipped. Decimation is the board's own (stepnet.decimate), normalisation is fit on the training
fold only. Loss: smooth-L1 between the integrated rate and the typed count; augmentation keeps the count true (pairing
two takes, frame rotation, gain, and on every pass a time-stretch by U(0.7, 1.3): the same count over a new duration,
which is what makes the rate follow the cadence instead of the time spent moving).
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "python"))
sys.path.insert(0, str(ROOT / "train"))
from recorder import parse_counted  # noqa: E402
from stepnet import decimate  # noqa: E402
from stepnet_torch import StepNetTorch  # noqa: E402

DATA, MODELS, FIGS = ROOT / "data", ROOT / "models", ROOT / "docs" / "figures"
EPOCHS, LR, LR_DROP_AT, SEEDS, BOOT = 400, 3e-3, 0.75, 5, 2000
P_AUG, STRETCH = 0.5, (0.7, 1.3)   # pairing/rotation/gain each half the time; a time-stretch in STRETCH on every pass
MAX_GAP_S = 0.8    # a break in t this long (FIFO overflow, re-probe; train.py's threshold) loses steps: skip the take


@dataclass
class Recording:
    name: str
    activity: str
    steps: int
    who: str
    x: np.ndarray      # (6, T) decimated raw channels
    demo: bool


def load_rows(path):
    """-> (t list, [(ax, ay, az, gx, gy, gz), ...]) in file order."""
    ts, out = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                row = float(r["t"]), tuple(float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz"))
            except (TypeError, ValueError):
                continue   # a truncated last row (power pulled)
            ts.append(row[0])
            out.append(row[1])
    return ts, out


def to_array(samples):
    """100 Hz samples -> (6, T) decimated array, exactly as the board does it."""
    dec, _ = decimate(samples)
    return np.array(dec, dtype=np.float32).T


def is_demo(who):
    w = who.lower()
    return w == "demo" or w.endswith("-demo")


def load_recordings(data_dir=DATA, log=print):
    recs = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        parsed = parse_counted(p.name)
        if not parsed:
            continue
        act, n, who = parsed
        ts, rows = load_rows(p)
        gap = float(np.diff(ts).max()) if len(ts) > 1 else 0.0
        if gap > MAX_GAP_S:
            log("skipping %s: a %.1f s gap in t (samples lost, the count would not match)" % (p.name, gap))
            continue
        x = to_array(rows)
        if x.size == 0:
            continue
        recs.append(Recording(p.name, act, n, who, x, is_demo(who)))
    return recs


# --- augmentation (count-preserving) -----------------------------------------------------------------------------
def rotate(x, rng, max_deg=15.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return np.vstack([R @ x[:3], R @ x[3:]]).astype(np.float32)


def gain(x, rng):
    return (x * rng.uniform(0.9, 1.1)).astype(np.float32)


def stretch(x, rng, lo=STRETCH[0], hi=STRETCH[1]):
    f = rng.uniform(lo, hi)
    T = x.shape[1]
    T2 = max(2, int(round(T * f)))
    src, dst = np.linspace(0, 1, T), np.linspace(0, 1, T2)
    return np.vstack([np.interp(dst, src, ch) for ch in x]).astype(np.float32)


def augment_pair(a, b):
    return np.concatenate([a.x, b.x], axis=1), a.steps + b.steps


def normalisation(recs):
    allx = np.concatenate([r.x for r in recs], axis=1)
    return allx.mean(axis=1), np.maximum(allx.std(axis=1), 1e-3)


def tensor(x, mean, std):
    return torch.tensor((x - mean[:, None]) / std[:, None], dtype=torch.float32)[None]


# --- fitting and predicting ---------------------------------------------------------------------------------------
def fit(recs, epochs=EPOCHS, seed=0, augment=True, log=None):
    """-> (model, per-epoch mean loss, mean, std). mean/std = normalisation(recs), the inputs the model expects."""
    mean, std = normalisation(recs)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StepNetTorch()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for ep in range(epochs):
        if ep == int(epochs * LR_DROP_AT):
            for g in opt.param_groups:
                g["lr"] = LR / 10
        total = 0.0
        for i in rng.permutation(len(recs)):
            x, y = recs[i].x, recs[i].steps
            if augment:
                if rng.random() < P_AUG:
                    x, y = augment_pair(recs[i], recs[int(rng.integers(len(recs)))])
                if rng.random() < P_AUG:
                    x = rotate(x, rng)
                if rng.random() < P_AUG:
                    x = gain(x, rng)
                x = stretch(x, rng)   # always: the same count over a new duration forces rate ~ cadence
            loss = F.smooth_l1_loss(model.count(tensor(x, mean, std)), torch.tensor([float(y)]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(recs))
        if log and (ep % 25 == 0 or ep == epochs - 1):
            log("epoch %3d  loss %.3f" % (ep, losses[-1]))
    return model, losses, mean, std


def predict(model, mean, std, x):
    with torch.no_grad():
        return model.count(tensor(x, mean, std)).item()


def rates(model, mean, std, x):
    with torch.no_grad():
        return model(tensor(x, mean, std))[0, 0].numpy()


def loro(recs, epochs=EPOCHS, seed=0, log=None):
    """Leave-one-recording-out: fit without each take and predict it. -> [{name, activity, who, true, pred}]."""
    out = []
    for i, held in enumerate(recs):
        rest = recs[:i] + recs[i + 1:]
        model, _, mean, std = fit(rest, epochs=epochs, seed=seed)
        pred = predict(model, mean, std, held.x)
        out.append({"name": held.name, "activity": held.activity, "who": held.who, "true": held.steps, "pred": round(pred, 2)})
        if log:
            log("LORO %2d/%d %-45s true %3d  pred %6.1f" % (i + 1, len(recs), held.name, held.steps, pred))
    return out


def seed_spread(recs, n=SEEDS, epochs=EPOCHS):
    """Fit n times with different seeds on all takes; -> per-take std of the predicted count (mean over takes)."""
    preds = []
    for s in range(n):
        model, _, mean, std = fit(recs, epochs=epochs, seed=s)
        preds.append([predict(model, mean, std, r.x) for r in recs])
    return float(np.mean(np.std(np.array(preds), axis=0)))


def metrics(results, rng=None):
    rng = rng or np.random.default_rng(0)
    err = np.array([r["pred"] - r["true"] for r in results])
    nz = np.array([r["true"] > 0 for r in results])
    pct = 100 * err[nz] / np.array([r["true"] for r in results])[nz] if nz.any() else np.array([0.0])
    boots = [rng.choice(pct, len(pct)).mean() for _ in range(BOOT)]
    by = {}
    for act in sorted({r["activity"] for r in results}):
        e = np.array([r["pred"] - r["true"] for r in results if r["activity"] == act])
        by[act] = {"n": int(len(e)), "mae_steps": round(float(np.abs(e).mean()), 2), "bias_steps": round(float(e.mean()), 2)}
    return {"n": len(results), "mae_steps": round(float(np.abs(err).mean()), 2), "mape_pct": round(float(np.abs(pct).mean()), 2),
            "bias_steps": round(float(err.mean()), 2), "ci95_pct": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
            "pm_pct95": round(float(np.percentile(np.abs(pct), 95)), 2) if nz.any() else None, "by_activity": by}


# --- the report ---------------------------------------------------------------------------------------------------
def write_results(path, m, results, demo, seed_std, losses, n_train):
    lines = ["# Step counter results", "", "Generated by `train/steps.py --report` on %s from %d counted takes." % (time.strftime("%Y-%m-%d %H:%M"), n_train), "",
             "## Leave-one-recording-out", "", "| metric | value |", "|---|---|",
             "| takes | %d |" % m["n"], "| mean absolute error | %.2f steps |" % m["mae_steps"], "| mean absolute percentage error | %.1f %% |" % m["mape_pct"],
             "| bias (predicted − true) | %+.2f steps |" % m["bias_steps"], "| 95 %% bootstrap interval of the mean %% error | %+.1f %% to %+.1f %% |" % tuple(m["ci95_pct"]),
             "| 95th percentile of the absolute %% error (shown live as ±) | %s |" % ("n/a (no take with steps)" if m["pm_pct95"] is None else "%.1f %%" % m["pm_pct95"]), "| seed-to-seed spread of a prediction (5 fits) | %.2f steps |" % seed_std, "",
             "| activity | takes | MAE (steps) | bias (steps) |", "|---|---|---|---|"]
    lines += ["| %s | %d | %.2f | %+.2f |" % (a, v["n"], v["mae_steps"], v["bias_steps"]) for a, v in m["by_activity"].items()]
    lines += ["", "| take | activity | true | predicted | error |", "|---|---|---|---|---|"]
    lines += ["| %s | %s | %d | %.1f | %+.1f |" % (r["name"], r["activity"], r["true"], r["pred"], r["pred"] - r["true"]) for r in results]
    if demo:
        lines += ["", "## Fresh takes the net never saw (`-demo`)", "", "| take | true | predicted | error |", "|---|---|---|---|"]
        lines += ["| %s | %d | %.1f | %+.1f |" % (r["name"], r["true"], r["pred"], r["pred"] - r["true"]) for r in demo]
    lines += ["", "Final fit: %d epochs, last-epoch loss %.3f." % (len(losses), losses[-1]), ""]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="LORO + seeds + demo + figures + docs/steps_results.md")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args(argv)
    recs = load_recordings(a.data)
    train, demo = [r for r in recs if not r.demo], [r for r in recs if r.demo]
    print("%d counted takes (%d training, %d demo)" % (len(recs), len(train), len(demo)))
    if len(train) < 3:
        sys.exit("need at least 3 counted takes in %s (names like ...-steps-walk-n42-Rafae.csv)" % a.data)
    meta = {"trained": time.strftime("%Y-%m-%dT%H:%M:%S"), "recordings": len(train), "epochs": a.epochs, "loro": None, "seed_std_steps": None}
    results = None
    if a.report:
        if len(train) < 4:
            sys.exit("--report needs at least 4 training takes")
        results = loro(train, epochs=a.epochs, seed=a.seed, log=print)
        m = metrics(results)
        meta["loro"] = m
        meta["seed_std_steps"] = round(seed_spread(train, epochs=a.epochs), 2)
        print(json.dumps(m, indent=1))
    else:  # keep the last measured half-width so the live ± does not fall back to 100 %, and say which run measured it
        try:
            old = json.loads((MODELS / "stepnet.json").read_text())["meta"]
            meta["loro"] = old.get("loro")
            if meta["loro"]:
                meta["loro_from"] = old.get("trained")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    model, losses, mean, std = fit(train, epochs=a.epochs, seed=a.seed, log=print)
    demo_res = [{"name": r.name, "true": r.steps, "pred": round(predict(model, mean, std, r.x), 2)} for r in demo]
    model.export(MODELS / "stepnet.json", mean, std, meta)
    print("wrote", MODELS / "stepnet.json")
    if a.report:   # the results first, so a figure error cannot lose them
        write_results(ROOT / "docs" / "steps_results.md", m, results, demo_res, meta["seed_std_steps"], losses, len(train))
        print("wrote docs/steps_results.md")
        from steps_plots import figures
        figures(FIGS, results, m, train, model, mean, std, losses)
        print("wrote docs/figures/steps_*.png")
    for r in demo_res:
        print("demo %-45s true %3d  pred %6.1f" % (r["name"], r["true"], r["pred"]))


if __name__ == "__main__":
    main()
