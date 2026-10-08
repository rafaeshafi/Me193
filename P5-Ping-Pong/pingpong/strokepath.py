"""How the ball leaves the paddle: the path of the hand and the twist of the hub decide where it goes and how it spins.

  * The PATH of the hand is its velocity over the stroke, a straight line through the pose readings around the gyro's peak
    (the camera sees the hand across and up, not forward: the forward speed is the gyro's, and it sets the ball's speed).
    Sideways: the ball is placed further that way and carries sidespin; upwards: a higher arc and topspin (a stroke that
    lifts); downwards: a flat arc and backspin (a chop).
  * The TWIST of the hub is the rotation it made over the stroke about the doorknob direction (the tilt calibration's axis,
    made perpendicular to the stroke axis, "turn right" positive).  A stroke always twists it somewhat, so it is measured
    against what THIS player's strokes usually do (the first hits of a session decide that), and what is beyond it is
    sidespin: twist the hub more to the right than usual and the ball curves right.

The gains are one dataclass (`--set shot.k_top=0.6` while playing).  Pure functions: the game feeds them what it saw.
"""

import math
from dataclasses import dataclass

WINDOW_S = (-0.30, 0.10)     # the path is read from the pose track from this long before the gyro's peak to this long after
MIN_SAMPLES = 4
MIN_SPAN_S = 0.12            # ... and has to cover at least this much time to give a slope worth using
MIN_CONF = 0.5
MIN_PERP = 0.3               # the doorknob axis must be at least this far (sine) from the stroke axis to tell a twist from a stroke
Z_CAP = 2.5


@dataclass(frozen=True)
class ShotModel:
    ref_speed_sw_s: float = 2.5     # a hand speed that counts as 1.0 (a normal stroke: the median of 62 real ones is 2.4)
    speed_cap: float = 1.5          # a hand speed counts up to this many times the reference
    k_aim: float = 0.20             # landing point shift (0..1 across the table) per 1.0 of sideways speed
    k_loft_m: float = 0.14          # extra height of the first arc, metres, per 1.0 of upward speed
    k_top: float = 0.45             # topspin per 1.0 of upward speed (downwards: backspin)
    k_side_path: float = 0.35       # sidespin per 1.0 of sideways speed
    k_side_twist: float = 0.30      # sidespin per 1.0 of twist beyond the player's usual
    twist_sd_deg: float = 30.0      # degrees of twist beyond the usual that count as 1.0 (the spread of real strokes)
    twist_warmup: int = 6           # hits of a session that decide what the usual twist is


@dataclass(frozen=True)
class HandPath:
    vu: float                       # shoulder widths a second, towards the player's right
    vv: float                       # ... and up
    n: int


@dataclass(frozen=True)
class ReturnShape:
    aim_shift: float                # added to the landing point (0..1 across the table)
    loft_m: float                   # added to the height of the first arc
    topspin: float                  # -1 backspin .. +1 topspin
    sidespin: float                 # -1 .. +1, positive: towards the player's right
    vu_norm: float = 0.0            # what it was made from, for the report
    vv_norm: float = 0.0
    twist_z: float = 0.0

    @property
    def amplitude(self):
        return min(1.0, math.hypot(self.topspin, self.sidespin))


def hand_path(poses, t_peak_ns, *, window_s=WINDOW_S, min_conf=MIN_CONF):
    """The hand's velocity over the stroke around the gyro's peak, or None when there are too few readings to say."""
    lo, hi = t_peak_ns + round(window_s[0] * 1e9), t_peak_ns + round(window_s[1] * 1e9)
    pts = [(p.t_scene_ns / 1e9, p.u, p.v) for p in poses if lo <= p.t_scene_ns <= hi and p.conf >= min_conf]
    if len(pts) < MIN_SAMPLES:
        return None
    t = [x[0] for x in pts]
    if max(t) - min(t) < MIN_SPAN_S:
        return None
    n = len(pts)
    tm, um, vm = sum(t) / n, sum(x[1] for x in pts) / n, sum(x[2] for x in pts) / n
    sxx = sum((ti - tm) ** 2 for ti in t)
    return HandPath(vu=sum((ti - tm) * (x[1] - um) for ti, x in zip(t, pts)) / sxx,
                    vv=sum((ti - tm) * (x[2] - vm) for ti, x in zip(t, pts)) / sxx, n=n)


def wrist_axis(u_fwd, tilt_axis):
    """The doorknob axis made perpendicular to the stroke axis (a unit vector, its sign kept), or None."""
    if tilt_axis is None:
        return None
    norm = math.sqrt(sum(c * c for c in tilt_axis))
    if norm < 1e-9:
        return None
    t = tuple(c / norm for c in tilt_axis)
    along = sum(a * b for a, b in zip(t, u_fwd))
    perp = tuple(a - along * b for a, b in zip(t, u_fwd))
    size = math.sqrt(sum(c * c for c in perp))
    return None if size < MIN_PERP else tuple(c / size for c in perp)


def twist_deg(net_rot_deg, axis):
    """The rotation of the hub over the stroke about the wrist axis, degrees (None without an axis)."""
    if axis is None:
        return None
    return sum(a * b for a, b in zip(net_rot_deg, axis))


class TwistBaseline:
    """What this player's strokes usually do about the wrist axis: the median of the first `warmup` hits, then fixed.

    z(twist) is the twist beyond that, in units of sd_deg; 0 while the usual is still being learnt."""

    def __init__(self, sd_deg=30.0, warmup=6):
        self.sd_deg, self.warmup, self._seen, self._usual = sd_deg, warmup, [], None

    def z(self, twist):
        if twist is None:
            return 0.0
        if self._usual is None:
            self._seen.append(twist)
            if len(self._seen) >= self.warmup:
                ordered = sorted(self._seen)
                mid = len(ordered) // 2
                self._usual = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
            return 0.0
        return max(-Z_CAP, min(Z_CAP, (twist - self._usual) / self.sd_deg))


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def shape_return(path, twist_z, model):
    """Aim shift, loft and spin of the return from the hand's path (None: not seen) and the hub's twist beyond usual."""
    un = vn = 0.0
    if path is not None:
        un = _clip(path.vu / model.ref_speed_sw_s, -model.speed_cap, model.speed_cap)
        vn = _clip(path.vv / model.ref_speed_sw_s, -model.speed_cap, model.speed_cap)
    z = _clip(twist_z or 0.0, -Z_CAP, Z_CAP)
    return ReturnShape(aim_shift=model.k_aim * un, loft_m=model.k_loft_m * max(0.0, vn), topspin=_clip(model.k_top * vn),
                       sidespin=_clip(model.k_side_path * un + model.k_side_twist * z), vu_norm=un, vv_norm=vn, twist_z=z)
