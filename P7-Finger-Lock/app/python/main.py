"""Finger lock entry point on the UNO Q; App Lab runs this inside the app container (cwd /app/python).

Loop: webcam frame -> crop the desk square -> 64x64 gray -> numpy FingerNet -> probabilities -> lock policy -> MCU
(matrix picture, LEDs, servo) + the web page on port 8080. deploy.sh puts lock/, models/fingers.json and config.json
next to this file's parent, so the policy here is byte-for-byte the one tested on the Mac.
"""
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # /app on the board = ~/ArduinoApps/finger-lock on its Linux host
# deploy.sh puts lock/, models/ and config.json under /app; run from the repo (tests) they are one level up.
FILES = ROOT if (ROOT / "config.json").exists() else ROOT.parent
sys.path.insert(0, str(FILES if FILES != ROOT else ROOT / "python"))
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(DATA / "app.log")])
log = logging.getLogger("lock")

import cv2  # noqa: E402  (in the App Lab container; the camera peripheral uses it too)
from arduino.app_peripherals.camera import Camera  # noqa: E402
from arduino.app_utils import App, Bridge  # noqa: E402

from lock import config, matrix  # noqa: E402
from lock.engine import NumpyEngine  # noqa: E402
from lock.policy import LISTENING, Lock  # noqa: E402
from lock.preprocess import crop, to_input  # noqa: E402
from lock.timing import Stats, Timer  # noqa: E402
from web import serve  # noqa: E402

cfg = config.load(FILES / "config.json")
engine = NumpyEngine(FILES / "models" / "fingers.json")
stats = Stats()


class Outputs:
    """What the MCU shows; only changes are sent over the Bridge (a call costs a few ms)."""

    def __init__(self):
        self.leds_sent, self.frame_sent, self.failures = None, None, 0

    def _call(self, name, *args):
        try:
            Bridge.call(name, *args, timeout=2)
            return True
        except Exception as exc:  # the MCU is still starting, or busy
            self.failures += 1
            if self.failures % 50 == 1:
                log.warning("bridge %s failed (%s)", name, exc)
            return False

    def leds(self, red, green):
        mask = (1 if red else 0) | (2 if green else 0)
        if mask != self.leds_sent and self._call("leds", mask):
            self.leds_sent = mask

    def draw(self, frame):
        b = matrix.to_bytes(frame)
        if b != self.frame_sent and self._call("draw", b):
            self.frame_sent = b

    def unlock(self, ms):
        self._call("unlock", int(ms))


class Safe:
    def __init__(self):
        self.lock = Lock(cfg, on_event=self.on_event)
        self.out = Outputs()
        self.cls, self.probs, self.roi_jpg, self.t0 = 0, None, None, time.monotonic()
        self.camera_ok, self.frames, self._lock = False, 0, threading.Lock()

    def on_event(self, kind, info):
        log.info("%s %s", kind, {k: v for k, v in info.items() if k != "image"})
        if kind == "unlock":
            self.out.unlock(cfg["servo_open_ms"])
        elif kind == "intruder" and info.get("image") is not None:
            path = DATA / "intruders" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{''.join(map(str, info['entered']))}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), info["image"])
            log.info("intruder picture %s", path)

    def button(self, *_):
        """The physical button (from the MCU) or POST /button on the page."""
        started = self.lock.press_button()
        log.info("button -> %s", self.lock.state)
        return started

    def on_frame(self, frame):
        with Timer(stats, "preprocess"):
            roi_img = crop(frame, cfg["roi"])
            x = to_input(roi_img, cfg["input_size"])
        with Timer(stats, "inference"):
            cls, probs = engine.predict(x)
        with self._lock:
            self.cls, self.probs, self.frames = cls, probs, self.frames + 1
            self.lock.observe(probs, image=roi_img if self.lock.state == LISTENING else None)
            if self.frames % 3 == 0:  # the page's picture, 3 times a second is plenty
                ok, buf = cv2.imencode(".jpg", roi_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.roi_jpg = buf.tobytes() if ok else self.roi_jpg

    def tick(self):
        with self._lock:
            self.lock.tick()
            snap = self.lock.snapshot()
        self.out.leds(*snap["leds"])
        self.out.draw(matrix.picture(dict(snap, timeout_total=cfg["entry_timeout_s"]), time.monotonic() - self.t0))

    def snapshot(self):
        with self._lock:
            snap = self.lock.snapshot()
        s = stats.summary()
        snap.update({"cls": self.cls, "fps": s.get("fps"), "inference_ms": s.get("inference", {}).get("mean_ms"),
                     "camera": self.camera_ok, "frames": self.frames, "engine": engine.name, "code_len": len(cfg["code"])})
        return snap

    def stats(self):
        return dict(stats.summary(), engine=engine.name, frames=self.frames, unlocks=self.lock.unlocks, denials=self.lock.denials)


safe = Safe()
Bridge.provide("button", safe.button)
serve(safe, port=cfg["web_port"])
log.info("model %s (%d parameters), code of %d digits, web on %d", engine.name, engine.net.n_params(), len(cfg["code"]), cfg["web_port"])

camera = None
_t_prev, _next_open, _last_log = None, 0.0, 0.0


def open_camera():
    global camera
    try:
        camera = Camera(cfg.get("board_camera", "usb:0"), resolution=tuple(cfg["frame_size"]), fps=cfg["fps"])
        camera.start()
        safe.camera_ok = True
        log.info("camera open: %s", cfg.get("board_camera", "usb:0"))
    except Exception as exc:
        camera, safe.camera_ok = None, False
        log.warning("no camera yet (%s); retrying", exc)


def loop():
    global _t_prev, _next_open, _last_log
    now = time.monotonic()
    if camera is None:
        if now >= _next_open:
            _next_open = now + 3.0
            open_camera()
        safe.tick()
        time.sleep(0.05)
        return
    with Timer(stats, "capture"):
        frame = camera.capture()  # blocks to the configured fps
    if frame is None:
        safe.tick()
        time.sleep(0.02)
        return
    if _t_prev is not None:
        stats.add("loop", (now - _t_prev) * 1000)
    _t_prev = now
    safe.on_frame(frame)
    safe.tick()
    if now - _last_log > 10:
        _last_log = now
        s = safe.snapshot()
        log.info("state=%s entered=%s cls=%s p=%s fps=%s inf=%s ms", s["state"], s["entered"], s["cls"],
                 None if s["probs"] is None else max(s["probs"]), s["fps"], s["inference_ms"])


App.run(user_loop=loop)
