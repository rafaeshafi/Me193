"""Every tunable in one flat file (P3 style).

Defaults live here.  The bench tools write what they MEASURE into
config_local.json (never hand-copied) and it overlays these defaults at import.
Environment overrides: PP_BROKER=host[:port], PP_RECORD_SCOPE=<scope>.
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_PATH = HERE / "config_local.json"

# --- MQTT ----------------------------------------------------------------
BROKER_HOST = "test.mosquitto.org"
BROKER_PORT = 1883
KEEPALIVE_S = 30
SCORE_TOPIC = "ME193/Rogers/RafaeShafi"          # the assigned topic, exact casing
# Extras live OUTSIDE ME193/ so a ME193/# or ME193/Rogers/# listener never sees them.
EXTRAS_PREFIX = "ME193-pp/RafaeShafi"
STATUS_TOPIC = EXTRAS_PREFIX + "/status"
DEMO_SCORE_TOPIC = EXTRAS_PREFIX + "/demo/score"
SELFTEST_TOPIC = EXTRAS_PREFIX + "/selftest"
RECORD_SCOPE = "record_session"                   # record_session | record_alltime | live_streak
RECORD_SCOPES = ("record_session", "record_alltime", "live_streak")

# --- LEGO Double Motor (the paddle) ----------------------------------------
CARD_COLOR = None        # e.g. "green"  (read it off the Connection Card)
CARD_SERIAL = None       # a STRING, e.g. "0997" (leading zeros matter)
NOTIFY_MS = 15           # 15 ms is the library minimum (~66 Hz)
ACCEL_PER_G = 1000.0     # hypothesis (milli-g); bench P2 measures it
GYRO_PER_DPS = 10.0      # hypothesis (decidegrees/s); bench P2 measures it
HUB_FS_RAW = 32767       # raw full scale; bench P3 sets it from clipping plateaus
HUB_RATE_HZ = None       # measured by env_check / bench_hub
HUB_WORST_GAP_MS = None
HUB_P999_GAP_MS = None
STALE_MS = 300.0         # silence beyond this pauses the game (set from the histogram)

# --- Haptics (actuator limits) ---------------------------------------------
MAX_WRITES_PER_S = 10
BLANK_AFTER_PULSE_S = 0.12   # IMU blanking after a motor pulse (bench P6 measures it)
DUTY_CAP = 0.25              # max motor-on fraction ...
DUTY_WINDOW_S = 2.0          # ... per rolling window

# --- Where the time goes (see pingpong/latency.py): typical values, tunable ----------------------------------
LAT_IMU_S = 0.040        # hub -> Mac: a sample is stamped on ARRIVAL, ~this long after the hand did it
LAT_STROKE_S = 0.14      # from the gyro's peak to the end of the forward stroke (live recordings: 0.10-0.22 s)
LAT_DISPLAY_S = 0.050    # a frame drawn -> light from the screen
LAT_AUDIO_S = 0.025      # a sound written -> heard
LAT_HAPTIC_S = 0.050     # a motor command written -> the hub's motors move

# --- Vision ------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_LAG_S = 0.10          # camera-vs-IMU lag; bench P8 measures it

_TUNABLE = (
    "BROKER_HOST", "BROKER_PORT", "KEEPALIVE_S", "RECORD_SCOPE",
    "CARD_COLOR", "CARD_SERIAL", "NOTIFY_MS", "ACCEL_PER_G", "GYRO_PER_DPS", "HUB_FS_RAW",
    "HUB_RATE_HZ", "HUB_WORST_GAP_MS", "HUB_P999_GAP_MS", "STALE_MS",
    "MAX_WRITES_PER_S", "BLANK_AFTER_PULSE_S", "DUTY_CAP", "DUTY_WINDOW_S",
    "CAMERA_INDEX", "CAMERA_LAG_S", "LAT_IMU_S", "LAT_STROKE_S", "LAT_DISPLAY_S", "LAT_AUDIO_S", "LAT_HAPTIC_S",
)


def parse_broker(text):
    """'host' or 'host:port' -> (host, port)."""
    host, sep, port = text.partition(":")
    if not host:
        raise ValueError(f"bad broker {text!r}")
    return host, int(port) if sep else 1883


def parse_record_scope(text):
    if text not in RECORD_SCOPES:
        raise ValueError(f"PP_RECORD_SCOPE must be one of {RECORD_SCOPES}, got {text!r}")
    return text


def merge_overrides(defaults, overrides):
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _defaults():
    return {name: globals()[name] for name in _TUNABLE}


def _check(overrides):
    serial = overrides.get("CARD_SERIAL")
    if serial is not None and not isinstance(serial, str):
        raise ValueError("CARD_SERIAL must be a string like '0997' (an int loses leading zeros)")
    if "RECORD_SCOPE" in overrides:
        parse_record_scope(overrides["RECORD_SCOPE"])


def load_local(path=LOCAL_PATH):
    """Validated overrides from config_local.json ({} if the file is absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    merge_overrides(_defaults(), data)   # raises on unknown keys
    _check(data)
    return data


def write_local(updates, path=LOCAL_PATH):
    """Merge bench results into config_local.json, keeping earlier results."""
    path = Path(path)
    merge_overrides(_defaults(), updates)
    _check(updates)
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(updates)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


MEASURED = set()      # the settings config_local.json held at start-up: what a bench (or the player) actually wrote


def is_measured(name):
    """True if `name` came from config_local.json rather than being this file's unmeasured default."""
    return name in MEASURED


def _apply():
    if os.environ.get("PP_NO_LOCAL") != "1":      # tests / ./pp ready ignore measured bench numbers
        local = load_local(LOCAL_PATH)
        globals().update(local)
        MEASURED.update(local)
    if os.environ.get("PP_BROKER"):
        globals()["BROKER_HOST"], globals()["BROKER_PORT"] = parse_broker(os.environ["PP_BROKER"])
    if os.environ.get("PP_RECORD_SCOPE"):
        globals()["RECORD_SCOPE"] = parse_record_scope(os.environ["PP_RECORD_SCOPE"])


_apply()
