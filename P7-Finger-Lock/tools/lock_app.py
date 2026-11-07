"""The lock on the Mac: the USB webcam, the trained net, the same policy as the safe, and the board's outputs drawn on
screen (two LEDs, the 13x8 matrix, the servo). Press b for the button.

    .venv/bin/python tools/lock_app.py [--camera 1] [--engine torch|numpy] [--saliency]

Keys: b button   s saliency overlay on/off (torch engine)   r clear failed attempts   q quit
      p save what the net sees right now (the crop + its 64x64 input + the prediction) to data/live/ for debugging
Timings go to docs/bench_mac.json when you quit (the 'UNO Q vs computer' table).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "train"):
    sys.path.insert(0, str(ROOT / sub))
from lock import config, matrix  # noqa: E402
from lock.engine import make_engine  # noqa: E402
from lock.policy import LISTENING, Lock  # noqa: E402
from lock.preprocess import crop, roi_box, to_input  # noqa: E402
from lock.timing import Stats, Timer  # noqa: E402

INTRUDERS = ROOT / "data" / "intruders"


def save_intruder(info, where=INTRUDERS):
    """The picture of the hand that failed for the third time, for the owner to look at later."""
    img = info.get("image")
    if img is None:
        return None
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{''.join(map(str, info['entered']))}.png"
    cv2.imwrite(str(path), img)
    return path


def draw_matrix(view, frame13x8, x, y, cell=14):
    for r, row in enumerate(frame13x8):
        for c, v in enumerate(row):
            shade = int(40 + 215 * v / 7)
            cv2.circle(view, (x + c * cell + cell // 2, y + r * cell + cell // 2), cell // 2 - 2, (0, shade // 3, shade), -1)
    cv2.rectangle(view, (x - 4, y - 4), (x + 13 * cell + 4, y + 8 * cell + 4), (90, 90, 90), 1)


def draw_leds(view, red, green, x, y):
    cv2.circle(view, (x, y), 14, (0, 0, 255) if red else (0, 0, 60), -1)
    cv2.circle(view, (x + 40, y), 14, (0, 255, 0) if green else (0, 60, 0), -1)
    cv2.putText(view, "LEDs", (x - 18, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_probs(view, probs, cls, x, y, w=140):
    for i, p in enumerate(probs or [0] * 6):
        yy = y + i * 20
        cv2.rectangle(view, (x + 18, yy), (x + 18 + int(w * p), yy + 14), (0, 200, 255) if i == cls else (120, 120, 120), -1)
        cv2.putText(view, str(i), (x, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(view, f"{100 * p:3.0f}%", (x + 22 + w, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def saliency_overlay(view, sal, box):
    x0, y0, x1, y1 = box
    heat = cv2.applyColorMap((255 * cv2.resize(sal, (x1 - x0, y1 - y0))).astype(np.uint8), cv2.COLORMAP_JET)
    view[y0:y1, x0:x1] = cv2.addWeighted(view[y0:y1, x0:x1], 0.5, heat, 0.5, 0)


def render(frame, cfg, snap, cls, probs, engine_name, stats, sal=None, servo_open=False, t=0.0):
    view = frame.copy()
    box = roi_box(frame.shape, cfg["roi"])
    if sal is not None:
        saliency_overlay(view, sal, box)
    colour = {"IDLE": (160, 160, 160), "LISTENING": (0, 255, 255), "UNLOCKED": (0, 255, 0), "DENIED": (0, 0, 255),
              "LOCKOUT": (0, 0, 255)}[snap["state"]]
    cv2.rectangle(view, box[:2], box[2:], colour, 2)
    cv2.putText(view, f"{snap['state']}   entered {' '.join(map(str, snap['entered'])) or '-'}   attempts {snap['attempts']}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    if snap["remaining"] is not None:
        cv2.putText(view, f"{snap['remaining']:.0f} s", (frame.shape[1] - 70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    draw_probs(view, probs, cls, 10, 60)
    draw_matrix(view, matrix.picture(dict(snap, timeout_total=cfg["entry_timeout_s"]), t), 10, 200)
    draw_leds(view, *snap["leds"], x=30, y=340)
    cv2.putText(view, "SERVO OPEN" if servo_open else "servo closed", (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if servo_open else (120, 120, 120), 2)
    s = stats.summary()
    inf = s.get("inference", {}).get("mean_ms")
    cv2.putText(view, f"{engine_name}  inference {inf if inf is not None else '-'} ms  {s['fps'] or '-'} fps",
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(view, "b button   s saliency   r clear attempts   q quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--engine", choices=("torch", "numpy"), default="torch")
    ap.add_argument("--saliency", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    engine = make_engine(a.engine, ROOT / "models")
    stats = Stats()
    servo_until = [0.0]

    def on_event(kind, info):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {kind} {dict((k, v) for k, v in info.items() if k != 'image')}")
        if kind == "unlock":
            servo_until[0] = time.monotonic() + cfg["servo_open_ms"] / 1000.0
        elif kind == "intruder":
            p = save_intruder(info)
            print(f"intruder picture saved: {p}")

    lock = Lock(cfg, on_event=on_event)
    cap = cv2.VideoCapture(cfg["camera"] if a.camera is None else a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_size"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_size"][1])
    if not cap.isOpened():
        sys.exit("camera did not open (tools/capture.py --list shows the indices)")
    show_sal, t_loop, t0 = a.saliency, None, time.monotonic()
    print(f"engine {engine.name}; code has {len(cfg['code'])} digits; press b")
    while True:
        now = time.monotonic()
        if t_loop is not None:
            stats.add("loop", (now - t_loop) * 1000)
        t_loop = now
        with Timer(stats, "capture"):
            ok, frame = cap.read()
        if not ok:
            print("no frame")
            break
        with Timer(stats, "preprocess"):
            roi_img = crop(frame, cfg["roi"])
            x = to_input(roi_img, cfg["input_size"])
        with Timer(stats, "inference"):
            cls, probs = engine.predict(x)
        lock.observe(probs, image=roi_img if lock.state == LISTENING else None)
        lock.tick()
        sal = engine.saliency(x) if show_sal else None
        view = render(frame, cfg, lock.snapshot(), cls, probs, engine.name, stats, sal,
                      servo_open=now < servo_until[0], t=now - t0)
        cv2.imshow("finger lock", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            lock.press_button()
        elif key == ord("s"):
            show_sal = not show_sal
        elif key == ord("r"):
            lock.attempts = 0
        elif key == ord("p"):
            live = ROOT / "data" / "live"
            live.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%H%M%S")
            cv2.imwrite(str(live / f"{stamp}-pred{cls}-crop.png"), roi_img)
            cv2.imwrite(str(live / f"{stamp}-pred{cls}-input.png"), (x * 255).astype(np.uint8))
            print(f"saved data/live/{stamp}-pred{cls}-*.png  probs {[round(v, 2) for v in probs]}")
    cap.release()
    cv2.destroyAllWindows()
    out = ROOT / "docs" / f"bench_mac_{engine.name}.json"
    out.parent.mkdir(exist_ok=True)
    summary = dict(stats.summary(), engine=engine.name, unlocks=lock.unlocks, denials=lock.denials)
    json.dump(summary, open(out, "w"), indent=1)
    print(f"timings: {json.dumps({k: v for k, v in summary.items() if k in ('capture', 'preprocess', 'inference', 'loop', 'fps')})}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
