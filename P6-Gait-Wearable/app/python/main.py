"""Gait wearable entry point on the UNO Q; App Lab runs this inside the app container (cwd /app/python)."""
import logging
import time
from pathlib import Path

from arduino.app_utils import App, Bridge

from models import Tree
from pipeline import Gait
from recorder import Recorder
from web import serve

ROOT = Path(__file__).resolve().parent.parent  # /app on the board = ~/ArduinoApps/gait-wearable on its Linux host
MODELS, DATA = ROOT / "models", ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(DATA / "app.log")])
log = logging.getLogger("gait")

gait = Gait(Tree.load(MODELS / "model1.json"), Tree.load(MODELS / "model2.json"), Recorder(DATA))
Bridge.provide("imu", gait.on_batch)
serve(gait, port=8080)
log.info("models loaded (depth %d and %d), web on 8080, data in %s", gait.m1.depth(), gait.m2.depth(), DATA)

_last_payload = None
_last_sent = 0.0
_last_log = 0.0


def loop():
    global _last_payload, _last_sent, _last_log
    gait.tick()
    p = gait.payload()
    if p != _last_payload or time.monotonic() - _last_sent > 5:   # resend so a reset MCU gets the screen back
        try:
            Bridge.call("tft", p, timeout=2)
            _last_payload = p
            _last_sent = time.monotonic()
        except Exception as e:  # the sketch may still be compiling or resetting; keep going
            log.warning("tft: %s", e)
    wave = gait.take_wave()
    if wave:
        try:
            Bridge.notify("tfth", wave)
        except Exception as e:
            log.warning("tfth: %s", e)
    now = time.monotonic()
    if now - _last_log > 10:
        _last_log = now
        s = gait.state
        log.info("act=%s m1=%s cad=%s pk=%s dropped=%d imu=%s", s["act"], s["m1"], s["cad"],
                 None if s["pk"] is None else round(s["pk"], 2), s["dropped"], s["imu"])
    time.sleep(0.05)


App.run(user_loop=loop)
