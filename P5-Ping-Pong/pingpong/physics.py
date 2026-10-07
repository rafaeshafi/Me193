"""Analytic ball legs in 3-D: exact arrival time and point, a real bounce, stylised spin, no integrator.

World, in metres: x across the table (positive to the player's right), y up from the table top, z along the table
from the player's edge (0) to the computer's (2.74); the net stands at z = 1.37.

A Leg is one flight: the computer's strike (z = CPU_Z) to the sweet spot over the player's end (z = HIT_Z), or back.
It is two parabolic arcs joined by one bounce on the RECEIVER's half.  The arcs are drawn by their apex heights (not
by gravity), so the net is cleared by construction at every speed.  Time to the bounce is a fixed share of the flat
flight; after the bounce topspin shortens and backspin lengthens the rest, and the final flight is never shorter than
0.30 s (the floor is applied to the FINAL time by stretching the whole timeline).  Sidespin bends the path en route
but the ball always arrives at the aimed point.  Net and out faults end early / long.  This is arcade physics on
purpose (Magnus is not modelled).
"""

import math
from dataclasses import dataclass

D_M = 3.0                  # nominal distance: a ball of speed v takes D_M / v seconds (the levels' speeds mean this)
TABLE_LEN_M = 2.74
HALF_WIDTH_M = 0.7625
NET_Z_M, NET_H_M = TABLE_LEN_M / 2, 0.1525
CPU_Z_M = 2.89             # the computer strikes the ball just behind its end of the table ...
HIT_Z_M = 0.30             # ... you meet it over your end: the ball is here when the timing is perfect ...
STRIKE_Y_M = 0.22          # ... and both of you strike it this high above the table
MIN_FLIGHT_S = 0.30
SPIN_TIME_GAIN = 0.25
NET_CLEAR_M = 0.07         # the ball passes at least this far above the net's tape
APEX_MIN_M = 0.40          # the first arc peaks at least this high above the table
REBOUND_APEX_M = 0.30      # the arc after the bounce peaks this high (and at least 0.05 above where it ends)
OUT_FACTOR = 1.10          # of the flight: how long an out fault carries on
FLY_ON_S = 1.0             # a ball nobody hits keeps flying this long past the sweet spot (it is a miss long before)
FLOOR_Y_M = -0.6           # ... and falls no lower than this below the table top
LONG_BOUNCE = 1.12         # an out ball would land this far along (a share of the ground distance): past the far end
RETURN_BOUNCE = 0.72       # a return bounces this far along its way to the computer


def x_of_a(a):
    """Reach-box coordinate a (0..1) -> lateral meters across the table."""
    return max(-1.0, min(1.0, (a - 0.5) * 2 * 0.8)) * HALF_WIDTH_M


def a_of_x(x):
    """Lateral meters -> reach-box coordinate a (0..1), the inverse of x_of_a within the box."""
    return max(0.0, min(1.0, 0.5 + x / (2 * 0.8 * HALF_WIDTH_M)))


def arc_y(s, ya, yb, apex):
    """Height at s (0..1) along the parabola from height ya to height yb that peaks at `apex` (>= both)."""
    m = math.sqrt(apex - ya) + math.sqrt(apex - yb)
    return ya + 2.0 * m * math.sqrt(apex - ya) * s - m * m * s * s


def _clearing_apex(ya, s_net):
    """The lowest apex (>= APEX_MIN_M) whose arc from height ya down to the table at s = 1 clears the tape at s_net."""
    apex = max(APEX_MIN_M, ya)
    while arc_y(s_net, ya, 0.0, apex) < NET_H_M + NET_CLEAR_M and apex < 3.0:
        apex += 0.02
    return apex


@dataclass(frozen=True)
class Leg:
    t0_ns: int
    v: float
    flight_s: float          # final flight time (>= 0.30 s)
    raw_s: float             # flight time before the floor
    p_bounce: float          # time structure: the share of the flat flight spent before the bounce
    p_land: float            # geometry: where the bounce is, as a share of the ground distance (> 1: beyond the end)
    spin_factor: float
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    z_start: float
    z_end: float
    aim_ab: tuple
    topspin: float
    sidespin: float
    apex1_m: float
    apex2_m: float
    terminal: str            # "arrive" | "net" | "out"

    @property
    def arrival_ns(self):
        return self.t0_ns + round(self.flight_s * 1e9)

    @property
    def p_net(self):
        """How far along the ground distance the net is."""
        return (NET_Z_M - self.z_start) / (self.z_end - self.z_start)

    @property
    def end_ns(self):
        if self.terminal == "arrive":
            return self.arrival_ns
        if self.terminal == "out":
            return self.t0_ns + round(self.flight_s * OUT_FACTOR * 1e9)
        return self.time_at_progress(self.p_net)

    @property
    def bounce_ns(self):
        return self.time_at_progress(self.p_land)

    def progress(self, t_ns):
        """Fraction of the way along the ground distance (1.0 at arrival; > 1 for a long ball)."""
        limit = self.arrival_ns + round(FLY_ON_S * 1e9) if self.terminal == "arrive" else self.end_ns
        tau = max(0.0, (min(t_ns, limit) - self.t0_ns) / 1e9)
        tau *= self.raw_s / self.flight_s            # undo the floor stretch
        base = D_M / self.v
        t_bounce = base * self.p_bounce
        if tau <= t_bounce:
            return tau / base
        return self.p_bounce + (tau - t_bounce) * self.spin_factor / base

    def time_at_progress(self, p):
        """The moment (ns) the ball has covered the share p of the ground distance (the inverse of progress)."""
        base = D_M / self.v
        tau = p * base if p <= self.p_bounce else base * self.p_bounce + (p - self.p_bounce) * base / self.spin_factor
        return self.t0_ns + round(tau * self.flight_s / self.raw_s * 1e9)

    def time_at_z(self, z):
        """The moment (ns) the ball is over the depth z (for an incoming ball: later for a smaller z, also past the arrival)."""
        return self.time_at_progress((z - self.z_start) / (self.z_end - self.z_start))

    def position(self, t_ns):
        """(x, y, z) in metres: lateral, height above the table, distance from the player's edge."""
        p = self.progress(t_ns)
        pc = min(p, 1.0)
        x = self.x_start + (self.x_end - self.x_start) * p
        x += 0.25 * self.sidespin * math.sin(math.pi * pc) * HALF_WIDTH_M
        if pc > self.p_bounce:
            q = (pc - self.p_bounce) / (1.0 - self.p_bounce)
            x += 0.15 * self.sidespin * HALF_WIDTH_M * math.sin(math.pi * q)
        return x, self._height(p), self.z_start + (self.z_end - self.z_start) * p

    def _height(self, p):
        if p <= self.p_land:
            return arc_y(p / self.p_land, self.y_start, 0.0, self.apex1_m)
        s = (p - self.p_land) / (1.0 - self.p_land)               # beyond 1 the arc carries on: past the sweet spot, off the table
        return max(FLOOR_Y_M, arc_y(s, 0.0, self.y_end, self.apex2_m))


def _plan(t0_ns, v, start, end, p_bounce, aim_ab, topspin, sidespin, fault):
    if v <= 0:
        raise ValueError("ball speed must be positive")
    base = D_M / v
    f = max(0.75, min(1.25, 1.0 + SPIN_TIME_GAIN * topspin))
    raw = base * (p_bounce + (1.0 - p_bounce) / f)
    p_net = (NET_Z_M - start[2]) / (end[2] - start[2])
    p_land, apex1 = p_bounce, None
    if fault == "net":
        p_land, apex1 = 0.75 * p_net, start[1]          # it lands on its own side and dies in the net
    elif fault == "out":
        p_land = LONG_BOUNCE                            # it sails over the far end
    if apex1 is None:
        apex1 = _clearing_apex(start[1], p_net / p_land)
    return Leg(
        t0_ns=int(t0_ns), v=float(v), flight_s=max(MIN_FLIGHT_S, raw), raw_s=raw, p_bounce=p_bounce, p_land=p_land,
        spin_factor=f, x_start=float(start[0]), x_end=float(end[0]), y_start=float(start[1]), y_end=float(end[1]),
        z_start=float(start[2]), z_end=float(end[2]), aim_ab=tuple(aim_ab), topspin=float(topspin),
        sidespin=float(sidespin), apex1_m=apex1, apex2_m=max(end[1] + 0.05, REBOUND_APEX_M),
        terminal={None: "arrive", "net": "net", "out": "out"}[fault])


def plan_leg(t0_ns, v, x_start, aim_ab, topspin=0.0, sidespin=0.0, fault=None):
    """The computer's ball to you: struck at its end, bouncing on your half (aim b: 0 short .. 1 deep), over the sweet
    spot at the lateral point aim a."""
    return _plan(t0_ns, v, (x_start, STRIKE_Y_M, CPU_Z_M), (x_of_a(aim_ab[0]), STRIKE_Y_M, HIT_Z_M),
                 0.64 + 0.16 * aim_ab[1], aim_ab, topspin, sidespin, fault)


def plan_return(t0_ns, v, start, aim_ab, topspin=0.0, sidespin=0.0, fault=None):
    """Your ball to the computer: from where you struck it, bouncing on its half, to its strike point at lateral a."""
    return _plan(t0_ns, v, tuple(start), (x_of_a(aim_ab[0]), STRIKE_Y_M, CPU_Z_M), RETURN_BOUNCE, aim_ab, topspin,
                 sidespin, fault)
