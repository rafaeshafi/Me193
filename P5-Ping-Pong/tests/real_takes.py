"""Real recordings from the actual Double Motor, for the swing detector's regression tests.

tests/data/real_hub_takes.jsonl holds 18 takes cut from the bench session of 2026-10-07 (a person swinging the
hub at the 1.8 m play position: soft, hard and max-effort forward swings, backswings only, shakes and a
fan-wave).  They are the reason the detector looks the way it does: real swings are slower and broader than the
150 ms pulses it was first written for.  The hub's units were measured on the same day.
"""

import dataclasses
from pathlib import Path

import numpy as np

from pingpong import calibration, fixtures
from pingpong.events import ImuSample
from pingpong.swing import SwingParams

DATA = Path(__file__).parent / "data" / "real_hub_takes.jsonl"
GPD, APG = 0.988, 1017.0            # counts per deg/s and per g, as measured on that hub
HZ = 64.0
REST_GYRO = (0, -2, 1)              # the hub's measured rest offset in raw counts (a few counts at most)
REST_DPS = np.array(REST_GYRO, dtype=float) / GPD
_ALL = fixtures.load_takes(DATA)


def takes(label):
    return [t for t in _ALL if t["label"] == label]


def source_indices(label):
    return [t["index"] for t in takes(label)]


def calibrated():
    """(axis, omega_lo, omega_hi) the way the calibration flow derives them: soft and hard takes as the 5 + 5."""
    samples = [t["samples"] for lab in ("soft", "hard") for t in takes(lab)]
    axis, _ = calibration.fit_forward_axis(samples, GPD)
    forward = calibration.forward_peaks(samples, axis, GPD, bias=REST_DPS)
    n = len(takes("soft"))
    strengths = calibration.swing_strengths(forward[:n], forward[n:])
    return axis, strengths["omega_lo"], strengths["omega_hi"]


def params(**kw):
    axis, lo, hi = calibrated()
    base = calibration.SwingCalibration(axis, lo, hi).swing_params(GPD, APG, 32767)
    return dataclasses.replace(base, **kw)


def rest(t0_ns, seconds, g=REST_GYRO, a=(0, 0, 1017), hz=HZ):
    return [ImuSample(t_ns=t0_ns + int(i * 1e9 / hz), g=g, a=a) for i in range(int(seconds * hz))]


def lead_in(take, seconds=0.7):
    """Rest samples before a take, so the detector's warm-up is done when the take starts (a person is still before a swing).

    The gyro at rest reads its small offset, not whatever the take starts with: some takes begin mid-backswing."""
    a = tuple(int(round(v)) for v in np.median([s.a for s in take["samples"][:6]], axis=0))
    return rest(take["samples"][0].t_ns - int(seconds * 1e9), seconds, REST_GYRO, a)


def true_forward_peak(take, axis):
    """(peak forward rate in dps, its time in ns from the take's first sample) -- measured independently of the detector.

    The offset is the hub's rest offset, not the take's first samples: most takes begin mid-backswing."""
    g = np.array([s.g for s in take["samples"]], dtype=float) / GPD
    forward = (g - REST_DPS) @ np.asarray(axis)
    k = int(forward.argmax())
    return float(forward[k]), take["samples"][k].t_ns - take["samples"][0].t_ns


def stitch(selected, rest_s=1.2):
    """Several takes as one continuous stream, a quiet stretch before each and after the last (what the flow sees)."""
    out, t = [], 5_000_000_000
    for take in selected:
        out += rest(t, rest_s)
        t = out[-1].t_ns + int(1e9 / HZ)
        base = take["samples"][0].t_ns
        shifted = [ImuSample(t_ns=t + (s.t_ns - base), g=s.g, a=s.a) for s in take["samples"]]
        out += shifted
        t = out[-1].t_ns + int(1e9 / HZ)
    out += rest(t, rest_s)
    return out
