"""Headless timing of the two engines on this machine, for the 'UNO Q vs computer' table.

    .venv/bin/python tools/bench.py [--n 300] [--camera 1]   # --camera also times real capture frames

Writes docs/bench_mac.json; the board writes the same shape of numbers at http://<board>:8080/stats.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "train"):
    sys.path.insert(0, str(ROOT / sub))
from lock import config  # noqa: E402
from lock.engine import make_engine  # noqa: E402
from lock.preprocess import crop, to_input  # noqa: E402
from lock.timing import Stats, Timer  # noqa: E402


def bench_engine(engine, xs, stats_name, stats):
    for x in xs:
        with Timer(stats, stats_name):
            engine.predict(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--camera", type=int, default=None, help="also time real frames from this camera")
    ap.add_argument("--out", default=ROOT / "docs" / "bench_mac.json")
    a = ap.parse_args(argv)
    cfg = config.load()
    rng = np.random.default_rng(0)
    xs = [rng.random((cfg["input_size"], cfg["input_size"]), dtype=np.float32) for _ in range(a.n)]
    stats = Stats(keep=10_000)
    engines = {}
    for kind in ("numpy", "torch"):
        try:
            engines[kind] = make_engine(kind, ROOT / "models")
        except Exception as e:  # no torch on the board; the numpy JSON is always there once trained
            print(f"{kind}: not available ({e})")
    for kind, eng in engines.items():
        bench_engine(eng, xs[:10], f"warmup_{kind}", stats)
        bench_engine(eng, xs, f"inference_{kind}", stats)
    if a.camera is not None:
        import cv2
        cap = cv2.VideoCapture(a.camera)
        t_prev = None
        for _ in range(min(a.n, 100)):
            now = time.monotonic()
            if t_prev is not None:
                stats.add("loop", (now - t_prev) * 1000)
            t_prev = now
            with Timer(stats, "capture"):
                ok, frame = cap.read()
            if not ok:
                break
            with Timer(stats, "preprocess"):
                x = to_input(crop(frame, cfg["roi"]), cfg["input_size"])
            for kind, eng in engines.items():
                with Timer(stats, f"inference_{kind}"):
                    eng.predict(x)
        cap.release()
    summary = stats.summary()
    json.dump(summary, open(a.out, "w"), indent=1)
    for k, v in summary.items():
        if isinstance(v, dict) and "mean_ms" in v:
            print(f"{k:18s} n={v['n']:4d}  mean {v['mean_ms']:7.2f} ms  p50 {v['p50_ms']:7.2f}  p95 {v['p95_ms']:7.2f}  max {v['max_ms']:7.2f}")
    print(f"written to {a.out}")
    return summary


if __name__ == "__main__":
    main()
