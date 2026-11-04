"""The shared settings file (config.json at the repo root, copied to the board by deploy.sh)."""
import json
from pathlib import Path

DEFAULTS = {
    "code": [2, 5, 3], "camera": 1, "frame_size": [640, 480], "fps": 10, "roi": [0.3, 0.2, 0.55], "input_size": 64,
    "min_confidence": 0.85, "hold_frames": 8, "gap_frames": 3, "entry_timeout_s": 20, "unlock_s": 5, "deny_s": 2,
    "max_attempts": 3, "lockout_s": 30, "servo_open_ms": 5000, "web_port": 8080,
}
DIGITS = (1, 2, 3, 4, 5)
CODE_LEN = 3


def validate_code(code):
    """The code is three different digits, each 1..5 (the assignment); anything else is refused at load time."""
    code = [int(d) for d in code]
    if len(code) != CODE_LEN:
        raise ValueError(f"code must have {CODE_LEN} digits, got {code}")
    if any(d not in DIGITS for d in code):
        raise ValueError(f"code digits must be 1..5, got {code}")
    if len(set(code)) != CODE_LEN:
        raise ValueError(f"code digits must all differ, got {code}")
    return code


def validate(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    out["code"] = validate_code(out["code"])
    x, y, s = out["roi"]
    if not (0 <= x and 0 <= y and 0 < s and x + s <= 1 and y + s <= 1):
        raise ValueError(f"roi [x, y, size] must sit inside the frame as fractions, got {out['roi']}")
    for key in ("hold_frames", "gap_frames", "max_attempts", "input_size"):
        if int(out[key]) < 1:
            raise ValueError(f"{key} must be at least 1")
    if not 0 < float(out["min_confidence"]) <= 1:
        raise ValueError("min_confidence must be in (0, 1]")
    return out


def load(path=None):
    """config.json next to the repo root (or the given path), validated, with defaults for anything missing."""
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "config.json"
    with open(path) as f:
        return validate(json.load(f))
