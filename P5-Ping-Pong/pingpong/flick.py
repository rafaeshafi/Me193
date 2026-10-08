"""The flick of the wrist: how the hub turns around the moment the paddle meets the ball decides the spin.

The hub's axes come from the tilt calibration: "up" is what its accelerometer reads when it is held upright, "forward" is the
doorknob axis (turning right is clockwise as the player sees it, so it points away from the player) made horizontal, and "right"
completes the frame.  Tipping the hub's front down is topspin, up is backspin; turning it (about up) or rolling it (about
forward) to the right is sidespin to the right.

A stroke always turns the hub somewhat (the hand arrives, the forearm rolls), so what counts is the turn beyond what this
player's strokes usually do: the first hits of a session decide that, so the way you normally hit is no spin and a flick is
the extra.  Pure functions over (arrival time, gyro in dps) samples.
"""

import math
from dataclasses import dataclass

MIN_SAMPLES = 3
MIN_PERP = 0.3               # the doorknob axis must be at least this far (sine) from vertical to tell forward from up


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return None if n < 1e-9 else tuple(c / n for c in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class WristFrame:
    up: tuple
    fwd: tuple
    right: tuple


def wrist_frame(neutral, tilt_axis):
    """The hub's up / forward / right in its own axes, or None without a tilt calibration."""
    if neutral is None or tilt_axis is None:
        return None
    up, axis = _unit(neutral), _unit(tilt_axis)
    if up is None or axis is None:
        return None
    along = _dot(axis, up)
    fwd = _unit(tuple(a - along * u for a, u in zip(axis, up)))
    if fwd is None or math.sqrt(max(0.0, 1.0 - along * along)) < MIN_PERP:
        return None
    right = (fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2], fwd[0] * up[1] - fwd[1] * up[0])
    return WristFrame(up=up, fwd=fwd, right=right)


def window_ns(contact_ns, imu_delay_ns, model):
    """The arrival times of the gyro samples that tell how the hub turned around the contact (they arrive imu_delay late)."""
    return (contact_ns + imu_delay_ns - round(model.flick_before_s * 1e9),
            contact_ns + imu_delay_ns + round(model.flick_after_s * 1e9))


def mean_rate(samples, lo_ns, hi_ns):
    """The mean gyro vector (dps) of the samples that arrived in [lo, hi], or None when there are too few to say."""
    window = [g for t, g in samples if lo_ns <= t <= hi_ns]
    if len(window) < MIN_SAMPLES:
        return None
    return tuple(sum(g[k] for g in window) / len(window) for k in range(3))


class FlickBaseline:
    """What the player's strokes usually do at the contact: the median turn of the first `warmup` hits, then fixed."""

    def __init__(self, warmup=6):
        self.warmup, self._seen, self._usual = warmup, [], None

    def deviation(self, rate):
        """The turn beyond the usual (dps, per axis); zero while the usual is still being learnt."""
        if rate is None:
            return (0.0, 0.0, 0.0)
        if self._usual is None:
            self._seen.append(rate)
            if len(self._seen) >= self.warmup:
                self._usual = tuple(self._median([r[k] for r in self._seen]) for k in range(3))
            return (0.0, 0.0, 0.0)
        return tuple(rate[k] - self._usual[k] for k in range(3))

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def _amount(rate_dps, model):
    """Signed 0..1: nothing inside the dead zone, then growing to 1 at flick_ref_dps."""
    size = abs(rate_dps)
    if size <= model.flick_min_dps:
        return 0.0
    full = max(1.0, model.flick_ref_dps - model.flick_min_dps)
    return math.copysign(min(1.0, (size - model.flick_min_dps) / full), rate_dps)


def spin_from_flick(dev, frame, model):
    """(topspin, sidespin), each -1..1, from the turn beyond the usual (a gyro vector in the hub's axes)."""
    if frame is None:
        return 0.0, 0.0
    pitch, yaw, roll = _dot(dev, frame.right), _dot(dev, frame.up), _dot(dev, frame.fwd)
    top = max(-1.0, min(1.0, model.k_flick_top * _amount(-pitch, model)))
    side = max(-1.0, min(1.0, model.k_flick_side * _amount(roll - yaw, model)))
    return top, side


def read_flick_detail(samples, contact_ns, imu_delay_ns, model, frame, baseline):
    """What the wrist did around contact_ns: the mean turn (dps) the samples show, how far it is beyond the player's usual,
    and the spin that makes (rate and dev are None when the hub told nothing)."""
    lo, hi = window_ns(contact_ns, imu_delay_ns, model)
    rate = mean_rate(samples, lo, hi)
    dev = baseline.deviation(rate)
    top, side = spin_from_flick(dev, frame, model)
    return {"rate": rate, "dev": None if rate is None else dev, "topspin": top, "sidespin": side}


def read_flick(samples, contact_ns, imu_delay_ns, model, frame, baseline):
    """(topspin, sidespin) of the stroke whose paddle met the ball at contact_ns, from the gyro samples around it."""
    detail = read_flick_detail(samples, contact_ns, imu_delay_ns, model, frame, baseline)
    return detail["topspin"], detail["sidespin"]
