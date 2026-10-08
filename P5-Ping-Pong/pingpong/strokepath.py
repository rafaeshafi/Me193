"""How the ball leaves the paddle: the path of the hand decides where it goes, how high, and how hard.

  * The PATH of the hand is its velocity over the stroke, a straight line through the pose readings (the camera sees the hand
    across and up, not forward).  Sideways: the ball is placed further that way; upwards: a higher arc.
  * HOW HARD: when the hand meeting the ball is the hit, the hand's speed (a hand that barely moves blocks the ball, a fast
    one hits it hard); a swing that hits it has the gyro's strength instead.
  * The SPIN is the wrist's: flick.py reads how the hub turned around the moment of contact.

The gains are one dataclass (`--set shot.k_aim=0.3` while playing).  Pure functions: the game feeds them what it saw.
"""

import math
from dataclasses import dataclass

WINDOW_S = (-0.30, 0.10)     # the path is read from the pose track from this long before the stroke's peak to this long after
MIN_SAMPLES = 4
MIN_SPAN_S = 0.12            # ... and has to cover at least this much time to give a slope worth using
MIN_CONF = 0.5


@dataclass(frozen=True)
class ShotModel:
    # the path of the hand
    ref_speed_sw_s: float = 2.5     # a hand speed that counts as 1.0 (a normal stroke: the median of 62 real ones is 2.4)
    speed_cap: float = 1.5          # a hand speed counts up to this many times the reference
    k_aim: float = 0.20             # landing point shift (0..1 across the table) per 1.0 of sideways speed
    k_loft_m: float = 0.14          # extra height of the first arc, metres, per 1.0 of upward speed
    # a hand that meets the ball is the hit: how hard from how fast the hand goes
    hand_lo_sw_s: float = 0.6       # a hand this slow only blocks the ball (the softest return) ...
    hand_hi_sw_s: float = 4.5       # ... and this fast hits it as hard as it goes
    hold_s: float = 0.11            # the ball stays on the paddle this long after they meet, so the flick can be seen
    # the flick of the wrist (flick.py)
    flick_before_s: float = 0.04    # the turn is read from this long before the contact ...
    flick_after_s: float = 0.06     # ... to this long after it (plus the hub's delay)
    flick_min_dps: float = 80.0     # a turn beyond the player's usual smaller than this is no flick
    flick_ref_dps: float = 350.0    # ... and one this big gives full spin
    k_flick_top: float = 1.0        # a negative gain flips the direction: tipping the hub's front down is topspin
    k_flick_side: float = 1.0       # ... turning or rolling it to the right is sidespin right
    flick_warmup: int = 6           # hits of a session that decide what the player's usual turn is


@dataclass(frozen=True)
class HandPath:
    vu: float                       # shoulder widths a second, towards the player's right
    vv: float                       # ... and up
    n: int


@dataclass(frozen=True)
class ReturnShape:
    aim_shift: float                # added to the landing point (0..1 across the table)
    loft_m: float                   # added to the height of the first arc
    topspin: float                  # -1 backspin .. +1 topspin (the flick's)
    sidespin: float                 # -1 .. +1, positive: towards the player's right (the flick's)
    vu_norm: float = 0.0            # what it was made from, for the report
    vv_norm: float = 0.0

    @property
    def amplitude(self):
        return min(1.0, math.hypot(self.topspin, self.sidespin))


def hand_path(poses, t_ns, *, window_s=WINDOW_S, min_conf=MIN_CONF):
    """The hand's velocity over the stroke around t_ns, or None when there are too few readings to say."""
    lo, hi = t_ns + round(window_s[0] * 1e9), t_ns + round(window_s[1] * 1e9)
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


def hand_strength(path, model):
    """0..1 from the hand's speed: how hard a hand that meets the ball hits it (None: nobody saw it move: a block)."""
    if path is None:
        return 0.0
    speed = math.hypot(path.vu, path.vv)
    span = model.hand_hi_sw_s - model.hand_lo_sw_s
    return max(0.0, min(1.0, (speed - model.hand_lo_sw_s) / span)) if span > 0 else float(speed >= model.hand_hi_sw_s)


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def shape_return(path, topspin, sidespin, model):
    """Aim shift and loft from the hand's path (None: not seen), and the flick's spin as given (clipped to -1..1)."""
    un = vn = 0.0
    if path is not None:
        un = _clip(path.vu / model.ref_speed_sw_s, -model.speed_cap, model.speed_cap)
        vn = _clip(path.vv / model.ref_speed_sw_s, -model.speed_cap, model.speed_cap)
    return ReturnShape(aim_shift=model.k_aim * un, loft_m=model.k_loft_m * max(0.0, vn), topspin=_clip(topspin),
                       sidespin=_clip(sidespin), vu_norm=un, vv_norm=vn)
