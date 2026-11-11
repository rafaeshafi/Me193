"""Take the training pictures with the USB webcam (never the laptop or phone camera).

    .venv/bin/python tools/capture.py --list            # which index is the USB webcam
    .venv/bin/python tools/capture.py --camera 1        # (default from config.json)

Keys: 0..5 pick the label (fingers showing; 0 = no hand or fist)   r  burst on/off (5 pictures a second)
      c one picture    arrows move the square, -/= resize it, w writes the square to config.json    q quit
Pictures go to data/raw/<label>/<session>-<n>.png as 128x128 colour crops of the square; the session is this run.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lock import config  # noqa: E402
from lock.preprocess import SAVE_SIZE, crop, resize, roi_box  # noqa: E402

KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2, 0, 3, 1  # OpenCV on macOS (waitKeyEx gives 63234.. ; waitKey & 0xFF gives these)


def list_cameras(max_index=6):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ok, frame = cap.read()
        if ok:
            found.append((i, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


def open_camera(index, size):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        sys.exit(f"camera {index} did not open; try --list")
    return cap


def counts(out_dir):
    return {label: len(list((out_dir / str(label)).glob("*.png"))) if (out_dir / str(label)).exists() else 0 for label in range(6)}


def overlay(frame, roi, label, burst, n_session, total, msg=""):
    x0, y0, x1, y1 = roi_box(frame.shape, roi)
    view = frame.copy()
    cv2.rectangle(view, (x0, y0), (x1, y1), (0, 0, 255) if burst else (0, 255, 0), 2)
    cv2.putText(view, f"label {label}   {'REC' if burst else 'paused'}   this session {n_session}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if burst else (255, 255, 255), 2)
    cv2.putText(view, "per class " + " ".join(f"{k}:{v}" for k, v in total.items()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(view, "0-5 label  r burst  c one  arrows/-/= square  w save square  q quit", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if msg:
        cv2.putText(view, msg, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=ROOT / "data" / "raw")
    ap.add_argument("--rate", type=float, default=5.0, help="pictures per second in a burst")
    a = ap.parse_args()
    if a.list:
        for i, w, h in list_cameras():
            print(f"camera {i}: {w}x{h}")
        return
    cfg = config.load()
    roi = list(cfg["roi"])
    cap = open_camera(cfg["camera"] if a.camera is None else a.camera, cfg["frame_size"])
    out_dir, session = Path(a.out), datetime.now().strftime("%Y%m%d-%H%M%S")
    label, burst, n_session, last_shot, msg, msg_until = 1, False, 0, 0.0, "", 0.0
    total = counts(out_dir)
    print(f"session {session}; pictures go to {out_dir}")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break
        now = time.monotonic()
        key = cv2.waitKey(1) & 0xFF
        shoot = False
        if key == ord("q"):
            break
        elif ord("0") <= key <= ord("5"):
            label = key - ord("0")
        elif key == ord("r"):
            burst = not burst
        elif key == ord("c"):
            shoot = True
        elif key in (KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, ord("-"), ord("=")):
            dx = {KEY_LEFT: -0.01, KEY_RIGHT: 0.01}.get(key, 0)
            dy = {KEY_UP: -0.01, KEY_DOWN: 0.01}.get(key, 0)
            ds = {ord("-"): -0.02, ord("="): 0.02}.get(key, 0)
            size = min(max(roi[2] + ds, 0.1), 1)
            roi = [min(max(roi[0] + dx, 0), 1 - size), min(max(roi[1] + dy, 0), 1 - size), size]
        elif key == ord("w"):
            import json
            path = ROOT / "config.json"
            data = json.load(open(path))
            data["roi"] = [round(v, 3) for v in roi]
            json.dump(data, open(path, "w"), indent=2)
            msg, msg_until = f"square saved to config.json {data['roi']}", now + 2
        if burst and now - last_shot >= 1.0 / a.rate:
            shoot = True
        if shoot:
            last_shot = now
            d = out_dir / str(label)
            d.mkdir(parents=True, exist_ok=True)
            n_session += 1
            cv2.imwrite(str(d / f"{session}-{n_session:04d}.png"), resize(crop(frame, roi), SAVE_SIZE))
            total[label] += 1
        cv2.imshow("capture (USB webcam)", overlay(frame, roi, label, burst, n_session, total, msg if now < msg_until else ""))
    cap.release()
    cv2.destroyAllWindows()
    print(f"session {session}: {n_session} pictures; totals {total}")


if __name__ == "__main__":
    main()
