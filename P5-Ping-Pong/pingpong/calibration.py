"""Swing calibration math (the interactive flow lives in tools/calibrate_swing.py).

  * forward axis u_fwd = principal axis (SVD) of the gyro vectors at the PEAK of each
    calibration swing, sign-fixed so the forward stroke is positive (the backswing
    then projects negative and never arms the detector);
  * omega_lo / omega_hi = medians of the peak rates of the soft and the full swings,
    which scale every shot's speed; T_PK = T_PK_FACTOR * omega_lo is the weakest swing that counts.
Everything is in deg/s through gyro_per_dps, so a unit mistake is a one-constant fix.
"""

import dataclasses
import json
import math
from dataclasses import dataclass

import numpy as np

from pingpong.swing import SwingParams

SOURCES = ("imu", "pose")
T_PK_FACTOR = 0.7        # the weakest swing that counts, as a fraction of the soft-swing strength.  Measured on real
                         # takes: the weakest deliberate soft swing was 0.81 x the median soft one, a swing's own
                         # recovery lobe up to 0.65 x, so 0.7 sits between them
# A camera delivers 15-30 samples a second, so a fast swing rises from rest to its peak in one frame: the
# hub's one-sample spike gate would throw real swings away.  A single-frame landmark jump is caught by a
# longer minimum duration instead (a real swing needs two frames to rise).  The hand moves to the ball
# before it swings, and a quick reposition can look like a swing: a short refractory period keeps that
# false swing from swallowing the real stroke that follows it (the hub's 0.3 s guards against vibration
# ringing, which a camera does not have).  max_dur_ms stays at the old 400 ms: these settings were tuned on scripted
# hand motion, never on a real recording (the hub's gyro, whose real swings last 0.5-1.4 s, is the supported sensor).
POSE_SWING_SETTINGS = {"spike_ratio": 0.0, "min_dur_ms": 100.0, "max_dur_ms": 400.0, "refractory_s": 0.15}
# The shake lock keeps its pre-hub-recording band for the camera for the same reason: the hub's 1.2-8 Hz / 1.5 s
# band comes from real hub shakes, and the camera's reposition-then-swing rhythm (~2 Hz) falls inside it.
POSE_SHAKE_SETTINGS = {"band_hz": (3.0, 8.0), "window_s": 1.0}


def fit_forward_axis(takes, gyro_per_dps):
    """takes: lists of ImuSample (one swing each) -> (unit forward axis, peak rate per take in dps)."""
    if len(takes) < 3:
        raise ValueError("need at least 3 swings to fit the forward axis")
    peaks, vectors = [], []
    for samples in takes:
        g = np.array([s.g for s in samples], dtype=float) / gyro_per_dps
        k = int(np.argmax(np.linalg.norm(g, axis=1)))
        vectors.append(g[k])
        peaks.append(float(np.linalg.norm(g[k])))
    vectors = np.array(vectors)
    units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(units, full_matrices=False)
    axis = vt[0]
    if float(np.sum(vectors @ axis)) < 0:           # the peak of a swing is its forward stroke
        axis = -axis
    return tuple(float(c) for c in axis), peaks


def forward_peaks(takes, axis, gyro_per_dps, bias):
    """Strength of each take: its largest FORWARD rate, i.e. the gyro (minus its rest offset) projected on `axis`.

    The largest gyro magnitude is not the strength: real soft swings have a backswing about as big as the
    stroke, so the biggest lobe can be the wrong one.  `bias` (dps, 3 numbers) is the hub's rest offset.  It is
    never guessed from a take's own first samples: real takes begin mid-backswing, and that offset inflated
    the strength of a soft swing by half.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    offset = np.asarray(bias, dtype=float)
    return [float(((np.array([s.g for s in samples], dtype=float) / gyro_per_dps - offset) @ axis).max())
            for samples in takes]


def swing_strengths(soft, full):
    """Peak rates of the soft and full calibration swings -> omega_lo, omega_hi and T_PK (all dps)."""
    lo, hi = float(np.median(soft)), float(np.median(full))
    if hi < 1.5 * lo:
        raise ValueError(f"soft ({lo:.0f}) and full ({hi:.0f}) swings are not clearly separated: "
                         "swing noticeably harder for the full ones")
    return {"omega_lo": lo, "omega_hi": hi, "t_pk": T_PK_FACTOR * lo}


@dataclass(frozen=True)
class SwingCalibration:
    u_fwd: tuple
    omega_lo: float
    omega_hi: float
    source: str = "imu"          # "imu": the hub's gyro; "pose": the camera as a virtual gyro (pingpong.posegyro)

    def __post_init__(self):
        norm = math.sqrt(sum(c * c for c in self.u_fwd))
        if norm < 1e-9:
            raise ValueError("forward axis must be a non-zero vector")
        if self.source not in SOURCES:
            raise ValueError(f"swing source must be one of {SOURCES}, got {self.source!r}")
        object.__setattr__(self, "u_fwd", tuple(c / norm for c in self.u_fwd))

    @property
    def t_pk(self):
        return T_PK_FACTOR * self.omega_lo

    def swing_params(self, gyro_per_dps, accel_per_g, fs_raw):
        params = SwingParams(u_fwd=self.u_fwd, gyro_per_dps=gyro_per_dps, accel_per_g=accel_per_g,
                             t_pk=self.t_pk, fs_raw=fs_raw)
        return dataclasses.replace(params, **POSE_SWING_SETTINGS) if self.source == "pose" else params

    def to_json(self):
        return json.dumps({"u_fwd": list(self.u_fwd), "omega_lo": self.omega_lo, "omega_hi": self.omega_hi,
                           "source": self.source})

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        return cls(u_fwd=tuple(d["u_fwd"]), omega_lo=float(d["omega_lo"]), omega_hi=float(d["omega_hi"]),
                   source=d.get("source", "imu"))
