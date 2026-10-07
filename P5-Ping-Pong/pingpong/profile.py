"""A player's calibration: swing axis + strengths, reach box, shoulder width.

Saved as plain JSON under data/players/<name>/calibration.json (data/ is gitignored: player
names and calibrations stay local).  An unknown player has no calibration -- the caller
decides between "run the calibration tool" and Calibration.default() (flagged uncalibrated so
the HUD can say so).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from pingpong.calibration import SwingCalibration
from pingpong.paddle import ReachBox
from pingpong.tilt import TiltCalibration

VERSION = 1


def default_root():
    return config.HERE / "data" / "players"


def slug(name):
    """A player name as a safe folder name ("Rafae Shafi" -> "rafae-shafi")."""
    out = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    if not out:
        raise ValueError(f"player name {name!r} has no letters or digits")
    return out


@dataclass(frozen=True)
class Calibration:
    swing: SwingCalibration
    box: ReachBox
    shoulder_w: Optional[float] = None      # standing shoulder width as a fraction of the frame; feeds PoseLock
    hand: str = "right"
    calibrated: bool = True
    tilt: Optional[TiltCalibration] = None  # how the paddle turns with the hub; None: it stays upright on screen

    @classmethod
    def default(cls, source="imu"):
        """Plausible values to try the game with; nothing here is measured.  `source` "pose" = the camera as the gyro."""
        lo, hi = (300.0, 1200.0) if source == "imu" else (250.0, 800.0)
        return cls(swing=SwingCalibration(u_fwd=(1.0, 0.0, 0.0), omega_lo=lo, omega_hi=hi, source=source),
                   box=ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5), calibrated=False)

    @property
    def swing_source(self):
        return self.swing.source

    def swing_params(self, gyro_per_dps, accel_per_g, fs_raw):
        return self.swing.swing_params(gyro_per_dps, accel_per_g, fs_raw)

    def to_json(self):
        box = self.box
        data = {
            "version": VERSION, "hand": self.hand, "shoulder_w": self.shoulder_w,
            "swing": {"u_fwd": list(self.swing.u_fwd), "omega_lo": self.swing.omega_lo,
                      "omega_hi": self.swing.omega_hi, "source": self.swing.source},
            "box": {"u_min": box.u_min, "u_max": box.u_max, "v_min": box.v_min, "v_max": box.v_max},
        }
        if self.tilt is not None:
            data["tilt"] = json.loads(self.tilt.to_json())
        return json.dumps(data, indent=2) + "\n"

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        swing, box = d["swing"], d["box"]
        return cls(swing=SwingCalibration(u_fwd=tuple(swing["u_fwd"]), omega_lo=float(swing["omega_lo"]),
                                          omega_hi=float(swing["omega_hi"]), source=swing.get("source", "imu")),
                   box=ReachBox(box["u_min"], box["u_max"], box["v_min"], box["v_max"]),
                   shoulder_w=d.get("shoulder_w"), hand=d.get("hand", "right"),
                   tilt=TiltCalibration.from_json(d["tilt"]) if d.get("tilt") else None)


def _path(name, root, source="imu"):
    """A hub calibration and a camera calibration are different measurements: each has its own file."""
    return Path(root or default_root()) / slug(name) / ("calibration.json" if source == "imu"
                                                         else f"calibration-{source}.json")


def save(name, calibration, root=None):
    path = _path(name, root, calibration.swing.source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibration.to_json())
    return path


def load(name, root=None, source="imu"):
    """The player's calibration for this swing source, or None if they have none yet.  A damaged file is an error."""
    path = _path(name, root, source)
    if not path.exists():
        return None
    try:
        return Calibration.from_json(path.read_text())
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"calibration file {path} is damaged ({exc}); delete it and re-run "
                         "./pp calibrate_swing") from exc
