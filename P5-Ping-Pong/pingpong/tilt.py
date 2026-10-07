"""Tilt: the paddle on screen turns when you turn the hub in your hand, side to side like a doorknob.

Where the hub's own axes point depends on how it sits in the fist, so the calibration measures three things once:

  * the AXIS you turn the hub about (the principal axis of the gyro while you turn it side to side),
  * which way is RIGHT: whatever you did first becomes a positive angle (clockwise as you see it), and
  * what UPRIGHT is: the accelerometer's direction while you hold the hub as the paddle, still.

TiltEstimator then follows the rotation about that axis: the gyro gives the fast change, gravity (the accelerometer
while the hub is not being thrown about) pulls the slow drift back.  Everything is in degrees.
"""

import json
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

MAX_DEG = 80.0               # the paddle is drawn leaning at most this far
MIN_SPEED_DPS = 25.0         # slower gyro readings are not part of turning the hub
MIN_RANGE_DEG = 15.0         # the turn has to reach this far each way
FIRST_MOVE_DEG = 12.0        # the first turn this big says which way is 'right'
AXIS_SHARE = 0.45            # of the turning energy that has to be about one axis (a wrist turn wanders; noise gives ~0.35)


class TiltError(Exception):
    pass


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class TiltCalibration:
    axis: tuple          # unit vector in hub coordinates: a clockwise turn (as the player sees it) is positive about it
    neutral: tuple       # unit vector: the accelerometer's direction with the paddle upright
    bias_dps: tuple      # the gyro's resting offset when this was measured (the estimator starts from it)

    def to_json(self):
        return json.dumps({"axis": list(self.axis), "neutral": list(self.neutral), "bias_dps": list(self.bias_dps)})

    @classmethod
    def from_json(cls, text):
        d = json.loads(text) if isinstance(text, str) else text
        return cls(axis=tuple(d["axis"]), neutral=tuple(d["neutral"]), bias_dps=tuple(d["bias_dps"]))


def fit(samples, *, neutral, gyro_per_dps, accel_per_g, bias_dps):
    """Samples of the hub turned side to side (right first) -> a TiltCalibration, or TiltError saying what to do.

    `neutral` is the accelerometer's direction while the hub was held upright and still (any length)."""
    t = np.array([s.t_ns for s in samples], dtype=float) / 1e9
    g = np.array([s.g for s in samples], dtype=float) / gyro_per_dps - np.asarray(bias_dps, dtype=float)
    moving = g[np.linalg.norm(g, axis=1) > MIN_SPEED_DPS]
    if len(moving) < 15:
        raise TiltError("the hub hardly turned: turn it side to side, further")
    _, sv, vt = np.linalg.svd(moving, full_matrices=False)
    if sv[0] ** 2 / float((sv ** 2).sum()) < AXIS_SHARE:
        raise TiltError("turn it about one axis, like a doorknob")
    axis = vt[0]
    rate = g @ axis
    angle = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2.0 * np.diff(t))])      # degrees, from the start
    first = next((a for a in angle if abs(a) >= FIRST_MOVE_DEG), None)
    if first is None:
        raise TiltError("the hub did not turn far enough: turn it side to side, further")
    if first < 0:                                               # the first move is 'right', positive by definition
        axis, angle = -axis, -angle
    if angle.max() - angle.min() < 2 * MIN_RANGE_DEG:
        raise TiltError("the hub did not turn far enough: turn it side to side, further")
    if angle.max() < MIN_RANGE_DEG or angle.min() > -MIN_RANGE_DEG:
        raise TiltError("turn it both ways: right, then left")
    return TiltCalibration(axis=tuple(float(c) for c in axis), neutral=tuple(float(c) for c in _unit(neutral)),
                           bias_dps=tuple(float(c) for c in bias_dps))


class TiltCapture:
    """The calibration step: hold the hub upright and steady, then turn it side to side -> a TiltCalibration.

    A held hub is never 'still' by a rate threshold, so steady means an unchanging orientation (the accelerometer's
    direction stays within a few degrees) for a second.  The player may also simply start turning once the hub has
    been steady for a little while (before the prompt changed): the steady stretch just before the turn is 'upright'.
    The turn starts when the hub's rotation rate, smoothed over 0.15 s, reaches `start_dps` (tremor does not)."""

    def __init__(self, *, accel_per_g, gyro_per_dps, settle_s=1.5, hold_s=1.0, min_steady_s=0.6, steady_deg=10.0,
                 start_dps=30.0, move_s=4.0):
        self.apg, self.gpd = accel_per_g, gyro_per_dps
        self.settle_ns, self.hold_ns, self.min_steady_ns = round(settle_s * 1e9), round(hold_s * 1e9), round(min_steady_s * 1e9)
        self.steady_deg, self.start_dps, self.move_ns = steady_deg, start_dps, round(move_s * 1e9)
        self.phase, self.result = "hold", None              # phase: "hold" until steady, then "turn"
        self._t0 = self._last_t = None
        self._sum = self._stretch_t0 = self._stretch_t1 = None
        self._motion, self._recent = deque(), deque()       # (t, rate): the last 0.15 s, and the last 2 s of smoothed rates
        self._move_t0, self._rows, self._neutral, self._bias, self._asked_ns = None, [], None, None, None

    @property
    def collecting(self):
        return self._move_t0 is not None

    @property
    def freezes_bias(self):
        """While the hub is being turned the gyro's resting offset must not follow it."""
        return self.phase == "turn" or self.collecting

    def feed(self, sample, rate, bias):
        """One sample (rate = |gyro - offset| in dps, bias = the offset) -> the notes to tell the player."""
        t = sample.t_ns
        self._t0 = t if self._t0 is None else self._t0
        self._last_t = t
        if self.collecting:
            self._rows.append(sample)
            return self._finish() if t - self._move_t0 >= self.move_ns else []
        motion = self._smooth(t, rate)
        self._follow_orientation(sample, t)
        settled = t - self._t0 >= self.settle_ns
        steady_ns = 0 if self._sum is None else self._stretch_t1 - self._stretch_t0
        if motion >= self.start_dps:                         # the turn begins
            notes = []
            if settled and steady_ns >= self.min_steady_ns:
                self._neutral, self._bias = _unit(self._sum), np.array(bias, dtype=float)
                self._move_t0, self._rows = t, [sample]
                return notes
            if settled and (self._asked_ns is None or t - self._asked_ns > 3e9):
                self._asked_ns = t
                notes.append("hold the hub steady for a second first, then turn it side to side")
            self._sum = None
            return notes
        if self.phase == "hold" and settled and steady_ns >= self.hold_ns:
            self.phase = "turn"
            return ["upright captured: now turn the hub side to side like a doorknob, right first"]
        return []

    def hint(self):
        """What the step is waiting for, for the window (a line under the prompt)."""
        if self.collecting:
            done = max(0.0, (self._last_t - self._move_t0) / 1e9)
            return f"turning... {min(done, self.move_ns / 1e9):.1f} of {self.move_ns / 1e9:.1f} s"
        if self.phase == "hold":
            elapsed = 0.0 if self._last_t is None else (self._last_t - self._t0 - self.settle_ns) / 1e9
            steady = 0.0 if self._sum is None else (self._stretch_t1 - self._stretch_t0) / 1e9
            return f"hold it steady: {max(0.0, min(steady, elapsed, self.hold_ns / 1e9)):.1f} of {self.hold_ns / 1e9:.1f} s"
        peak = max((m for _, m in self._recent), default=0.0)
        if 10.0 <= peak < self.start_dps:
            return f"turn it harder: you reached {peak:.0f} dps, it needs {self.start_dps:.0f}"
        return "now turn the hub side to side: right, then left"

    def _smooth(self, t, rate):
        self._motion.append((t, rate))
        while self._motion and self._motion[0][0] < t - 150_000_000:
            self._motion.popleft()
        smooth = sum(r for _, r in self._motion) / len(self._motion)
        self._recent.append((t, smooth))
        while self._recent and self._recent[0][0] < t - 2_000_000_000:
            self._recent.popleft()
        return smooth

    def _follow_orientation(self, sample, t):
        a = np.asarray(sample.a, dtype=float) / self.apg
        size = float(np.linalg.norm(a))
        if not 0.8 <= size <= 1.25:                          # thrown about, not held: leave the stretch alone
            return
        u = a / size
        if self._sum is not None and math.degrees(math.acos(float(np.clip(u @ _unit(self._sum), -1.0, 1.0)))) <= self.steady_deg:
            self._sum, self._stretch_t1 = self._sum + u, t
        else:
            self._sum, self._stretch_t0, self._stretch_t1 = u.copy(), t, t

    def _finish(self):
        rows, self._rows, self._move_t0, self._sum = self._rows, [], None, None
        try:
            self.result = fit(rows, neutral=self._neutral, gyro_per_dps=self.gpd, accel_per_g=self.apg,
                              bias_dps=tuple(float(c) for c in self._bias))
        except TiltError as exc:
            return [str(exc)]
        return ["tilt measured: the paddle on screen will turn with the hub"]


class TiltEstimator:
    """The paddle's angle from the hub's samples: gyro for the fast part, gravity for the slow part."""

    def __init__(self, calibration, gyro_per_dps, accel_per_g, *, nominal_hz=64.0, lost_s=0.12, tau_s=1.0,
                 accel_band=(0.9, 1.1), min_observable=0.4, leak_s=30.0):
        self.axis, self.neutral = _unit(calibration.axis), _unit(calibration.neutral)
        e1 = self.neutral - (self.neutral @ self.axis) * self.axis
        # the reference direction in the plane gravity turns in; none if 'upright' points along the axis itself
        self._e1 = _unit(e1) if np.linalg.norm(e1) > 0.2 else None
        self._e2 = None if self._e1 is None else np.cross(self.axis, self._e1)
        self.gpd, self.apg = gyro_per_dps, accel_per_g
        self.dt_nominal, self.lost_s = 1.0 / nominal_hz, lost_s
        self.tau_s, self.band, self.min_observable, self.leak_s = tau_s, accel_band, min_observable, leak_s
        self._bias = np.asarray(calibration.bias_dps, dtype=float)
        self.angle, self._t_ns = 0.0, None

    def feed(self, sample):
        """-> the paddle's angle in degrees after this sample (positive = clockwise as the player sees it)."""
        # The hub samples at a steady rate but BLE delivers the samples in bursts, and they are stamped on arrival: each
        # sample is one nominal period long, unless the gap is so long that samples were lost (then the time passed counts).
        gap = 0.0 if self._t_ns is None else (sample.t_ns - self._t_ns) / 1e9
        dt = 0.0 if self._t_ns is None else (self.dt_nominal if gap < self.lost_s else min(0.3, gap))
        self._t_ns = sample.t_ns
        w = np.asarray(sample.g, dtype=float) / self.gpd
        if np.linalg.norm(w - self._bias) < 5.0:                # at rest: follow the gyro's own drift, slowly
            self._bias = self._bias + 0.004 * (w - self._bias)
        self.angle += float((w - self._bias) @ self.axis) * dt
        a = np.asarray(sample.a, dtype=float) / self.apg
        magnitude = float(np.linalg.norm(a))
        if self._e1 is not None and self.band[0] <= magnitude <= self.band[1]:
            perp = a - (a @ self.axis) * self.axis
            if np.linalg.norm(perp) / magnitude >= self.min_observable:
                seen = -math.degrees(math.atan2(perp @ self._e2, perp @ self._e1))      # gravity turns the other way
                error = (seen - self.angle + 180.0) % 360.0 - 180.0
                self.angle += min(1.0, dt / self.tau_s) * error
                return self._clip()
        self.angle -= self.angle * min(1.0, dt / self.leak_s)       # gravity cannot tell: let it relax to upright
        return self._clip()

    def _clip(self):
        self.angle = max(-MAX_DEG, min(MAX_DEG, self.angle))
        return self.angle
