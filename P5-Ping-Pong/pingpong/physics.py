"""Analytic ball legs: exact arrival time and point, stylised spin, no integrator.

A Leg is one flight across the table (3 m).  Time to the bounce (55% of the flat
flight) is fixed; after the bounce topspin shortens and backspin lengthens the
remaining time, and the final flight is never shorter than 0.30 s (the floor is
applied to the FINAL time by stretching the whole timeline).  Sidespin bends the
path en route but the ball always arrives at the aimed point.  Net and out faults
end early / long.  This is arcade physics on purpose (Magnus is not modelled).
"""

import math
from dataclasses import dataclass

D_M = 3.0
P_BOUNCE = 0.55
MIN_FLIGHT_S = 0.30
HALF_WIDTH_M = 0.7625
SPIN_TIME_GAIN = 0.25
NET_FRACTION = 0.5       # of the flight: where a net fault ends
OUT_FACTOR = 1.10        # of the flight: how long an out fault carries on


def x_of_a(a):
    """Reach-box coordinate a (0..1) -> lateral meters across the table."""
    return max(-1.0, min(1.0, (a - 0.5) * 2 * 0.8)) * HALF_WIDTH_M


@dataclass(frozen=True)
class Leg:
    t0_ns: int
    v: float
    flight_s: float          # final flight time (>= 0.30 s)
    raw_s: float             # flight time before the floor
    p_bounce: float
    spin_factor: float
    x_start: float
    x_end: float
    aim_ab: tuple
    topspin: float
    sidespin: float
    apex_m: float
    terminal: str            # "arrive" | "net" | "out"

    @property
    def arrival_ns(self):
        return self.t0_ns + round(self.flight_s * 1e9)

    @property
    def end_ns(self):
        scale = {"arrive": 1.0, "net": NET_FRACTION, "out": OUT_FACTOR}[self.terminal]
        return self.t0_ns + round(self.flight_s * scale * 1e9)

    def progress(self, t_ns):
        """Fraction of the way across the table (1.0 at arrival; > 1 for a long ball)."""
        tau = max(0.0, (min(t_ns, self.end_ns) - self.t0_ns) / 1e9)
        tau *= self.raw_s / self.flight_s            # undo the floor stretch
        base = D_M / self.v
        t_bounce = base * self.p_bounce
        if tau <= t_bounce:
            return tau / base
        return self.p_bounce + (tau - t_bounce) * self.spin_factor / base

    def position(self, t_ns):
        """(lateral meters, progress, height meters) for rendering."""
        p = self.progress(t_ns)
        pc = min(p, 1.0)
        x = self.x_start + (self.x_end - self.x_start) * p
        x += 0.25 * self.sidespin * math.sin(math.pi * pc) * HALF_WIDTH_M
        if pc > self.p_bounce:
            q = (pc - self.p_bounce) / (1.0 - self.p_bounce)
            x += 0.15 * self.sidespin * HALF_WIDTH_M * math.sin(math.pi * q)
        return x, p, self.apex_m * math.sin(math.pi * pc)


def plan_leg(t0_ns, v, x_start, aim_ab, topspin=0.0, sidespin=0.0, fault=None):
    if v <= 0:
        raise ValueError("ball speed must be positive")
    base = D_M / v
    f = max(0.75, min(1.25, 1.0 + SPIN_TIME_GAIN * topspin))
    raw = base * (P_BOUNCE + (1.0 - P_BOUNCE) / f)
    return Leg(
        t0_ns=int(t0_ns), v=float(v), flight_s=max(MIN_FLIGHT_S, raw), raw_s=raw, p_bounce=P_BOUNCE,
        spin_factor=f, x_start=float(x_start), x_end=x_of_a(aim_ab[0]), aim_ab=tuple(aim_ab),
        topspin=float(topspin), sidespin=float(sidespin),
        apex_m=min(0.30, 0.10 + 0.4 * raw ** 2),
        terminal={None: "arrive", "net": "net", "out": "out"}[fault],
    )
