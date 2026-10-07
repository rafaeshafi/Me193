"""Signed forward-axis swing detector (pure; samples in, SwingEvents out).

Why signed: a magnitude-only FSM fires on the BACKSWING (reviewer simulation:
100% of backswing-only takes).  Here the gyro vector (minus its rest offset) is
projected on the learned forward axis u_fwd, so s is the FORWARD rate: a backswing
projects NEGATIVE and never arms, and a forward lobe arms, is tracked to its peak
on the raw signed rate (never EMA: smoothing delays the peak), and fires once the
rate has fallen back from it:

    IDLE --s > ARM and rising--> FWD --peak >= T_PK and s <= (1 - fire_drop) * peak--> COOL

("or the peak is confirm_s old and s is below it": a real stroke builds for ~0.5 s with shoulders and
hesitations, so a one-sample dip is not its peak.)  An oscillation guard (>2 sign reversals of s in the
previous 0.8 s among lobes >= lobe_fraction * peak) suppresses waving; a spike gate drops one-sample
vibration spikes; samples inside a haptic blank window are ignored.  Every threshold is in dps via
gyro_per_dps, so a wrong unit guess is a one-constant fix.

Written for 150 ms scripted pulses, then reworked on the real hub (2026-10-07): real strokes last
0.5-1.4 s, the backswing is as big as the stroke, and the swing is reported 33-150 ms after its peak.
"""

import math
from collections import deque
from dataclasses import dataclass

from pingpong.events import SwingEvent


@dataclass(frozen=True)
class SwingParams:
    u_fwd: tuple = (1.0, 0.0, 0.0)
    gyro_per_dps: float = 10.0       # raw counts per deg/s (bench P2)
    accel_per_g: float = 1000.0
    t_pk: float = 250.0              # dps: a fraction of the soft-swing strength (calibration.T_PK_FACTOR)
    fs_raw: int = 32767              # raw full scale (bench P3)
    arm_factor: float = 0.4          # ARM = arm_factor * t_pk
    start_factor: float = 0.5        # SWING_START at 0.5 * t_pk
    max_swing_s: float = 1.2         # from the forward rate crossing ARM to the impact (real swings ramp up for ~0.5 s)
    min_swing_s: float = 0.04
    min_dur_ms: float = 60.0
    max_dur_ms: float = 2000.0       # 2 x (onset -> peak): real soft swings measured 0.5-1.4 s
    fire_drop: float = 0.35          # the impact fires once the forward rate has fallen this far below its peak ...
    confirm_s: float = 0.12          # ... or the peak is this old and the rate is below it (a hesitation is not a peak)
    refractory_s: float = 0.30
    cool_w_dps: float = 100.0
    idle_w_dps: float = 25.0
    reversal_window_s: float = 0.8
    max_reversals: int = 2
    lobe_fraction: float = 0.25       # a lobe counts toward the oscillation guard from this fraction of the swing's peak
    spike_ratio: float = 0.4
    warmup_s: float = 0.5


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n > 1e-9 else (0.0, 0.0, 0.0)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def swing_features(g_peak, net_rot, a_lin, w_pk, dur_ms, back_ratio):
    """The 12 spin-classifier features: 3 + 3 + 3 unit vectors, log peak, duration, back ratio."""
    return (*_unit(g_peak), *_unit(net_rot), *_unit(a_lin),
            math.log(max(w_pk, 1e-6)), dur_ms / 400.0, back_ratio)


def count_reversals(history, onset, current_sign, params, peak):
    """Sign reversals of s among lobes (>= lobe_fraction * peak) in the window before this swing."""
    lo = onset - params.reversal_window_s
    lobes, sign, best = [], 0, 0.0
    for t, s in history:
        if t < lo or t >= onset:
            continue
        sg = 1 if s > 0 else -1 if s < 0 else 0
        if sg == 0:
            continue
        if sg != sign:
            if sign != 0 and best >= params.lobe_fraction * peak:
                lobes.append(sign)
            sign, best = sg, 0.0
        best = max(best, abs(s))
    if sign != 0 and best >= params.lobe_fraction * peak:
        lobes.append(sign)
    lobes.append(current_sign)
    return sum(1 for a, b in zip(lobes, lobes[1:]) if a != b)


class SwingDetector:
    def __init__(self, params):
        n = math.sqrt(sum(c * c for c in params.u_fwd))
        if n < 1e-9:
            raise ValueError("u_fwd must be a non-zero vector")
        self.p = params
        self.u = tuple(c / n for c in params.u_fwd)
        self._arm = params.arm_factor * params.t_pk
        self._blanks = []
        self._history = deque()
        self._state = "IDLE"
        self._bias = (0.0, 0.0, 0.0)
        self._gravity = None
        self._t_first = None
        self._warm = []
        self._prev = None            # (t, s, w)
        self._last_low_t = None
        self._quiet_s = 0.0
        self._cool_until = 0.0
        self._fired_t = 0.0
        self._src = "hub"

    # --- blank windows (haptic pulses) --------------------------------------------
    def blank(self, start_ns, end_ns):
        self._blanks.append((start_ns, end_ns))

    def _blanked(self, t_ns):
        self._blanks = [b for b in self._blanks if b[1] >= t_ns - 5_000_000_000]
        return any(a <= t_ns <= b for a, b in self._blanks)

    def trace(self, seconds):
        """[(t_seconds, signed forward rate in dps)] for the last `seconds` -- the HUD's swing trace."""
        if not self._history:
            return []
        cutoff = self._history[-1][0] - seconds
        return [(t, s) for t, s in self._history if t >= cutoff]

    # --- main entry ------------------------------------------------------------------
    def feed(self, sample):
        if self._blanked(sample.t_ns):
            return []
        t = sample.t_ns / 1e9
        self._src = sample.src
        g = tuple(v / self.p.gyro_per_dps for v in sample.g)
        if self._gravity is None:
            self._gravity = tuple(float(v) for v in sample.a)
        if self._t_first is None:
            self._t_first = t
        warming = (t - self._t_first) < self.p.warmup_s
        if warming:
            self._warm.append(g)
        elif self._warm:
            self._finish_warmup()
        gu = tuple(g[k] - self._bias[k] for k in range(3))
        s, w = _dot(gu, self.u), math.sqrt(sum(c * c for c in gu))
        self._history.append((t, s))
        while self._history and self._history[0][0] < t - 2.0:
            self._history.popleft()
        if self._state != "FWD" and s <= 0.2 * self._arm:
            self._last_low_t = t                  # in the cool-down too: a swing may arm the moment it ends

        events = []
        prev = self._prev
        if not warming:
            if self._state == "IDLE":
                events += self._idle(sample, t, g, gu, s, w, prev)
            elif self._state == "FWD":
                events += self._forward(sample, t, gu, s, w, prev)
            elif self._state == "COOL" and t >= self._cool_until and (
                    w < self.p.cool_w_dps or (self._last_low_t or 0.0) > self._fired_t):
                self._state = "IDLE"          # settled, or the forward rate has dropped since the impact: stroke over
        self._prev = (t, s, w)
        return events

    def _finish_warmup(self):
        n = len(self._warm)
        mean = tuple(sum(g[k] for g in self._warm) / n for k in range(3))
        spread = max(math.sqrt(sum((g[k] - mean[k]) ** 2 for g in self._warm) / n) for k in range(3))
        self._bias = mean if spread < 15.0 else (0.0, 0.0, 0.0)   # moving at start-up: assume no offset
        self._warm = []

    # --- IDLE -------------------------------------------------------------------------
    def _idle(self, sample, t, g, gu, s, w, prev):
        if w < self.p.idle_w_dps:
            self._quiet_s += (t - prev[0]) if prev else 0.0
            if self._quiet_s >= 0.3:
                self._bias = tuple(self._bias[k] + 0.05 * (g[k] - self._bias[k]) for k in range(3))
                self._gravity = tuple(0.9 * self._gravity[k] + 0.1 * sample.a[k] for k in range(3))
        else:
            self._quiet_s = 0.0
        if prev is not None and s > self._arm and s > prev[1]:
            self._begin(sample, t, gu, s, w, prev)
            return self._forward(sample, t, gu, s, w, prev, first=True)
        return []

    def _begin(self, sample, t, gu, s, w, prev):
        self._state = "FWD"
        self._onset = self._last_low_t if self._last_low_t is not None else prev[0]
        self._t_arm = t
        self._started = False
        self._net = [0.0, 0.0, 0.0]
        self._clip_run = self._clip_max = 0
        self._pk = {"w": s, "t": t, "prev_w": prev[1], "prev_t": prev[0], "next_w": None, "next_t": None,
                    "g": gu, "a": sample.a}

    # --- FWD ---------------------------------------------------------------------------
    def _forward(self, sample, t, gu, s, w, prev, first=False):
        events = []
        p, pk = self.p, self._pk
        if not first:
            dt = t - prev[0]
            self._net = [self._net[k] + gu[k] * dt for k in range(3)]
        clipped_now = any(abs(v) >= p.fs_raw for v in sample.g)
        self._clip_run = self._clip_run + 1 if clipped_now else 0
        self._clip_max = max(self._clip_max, self._clip_run)
        if not self._started and s > p.start_factor * p.t_pk:
            self._started = True
            events.append(self._event("SWING_START", sample.t_ns, s, 0.0, 0, gu, (0.0,) * 3, (0.0,) * 3, False))
        if s >= pk["w"]:                                   # the peak is the FORWARD rate: a wobble about another axis
            self._pk = pk = {"w": s, "t": t, "prev_w": prev[1], "prev_t": prev[0], "next_w": None,
                             "next_t": None, "g": gu, "a": sample.a}   # must not look like a swing
        elif pk["next_w"] is None:
            pk["next_w"], pk["next_t"] = s, t

        if (t - self._t_arm) > p.max_swing_s or (s < 0.3 * self._arm and pk["w"] < p.t_pk):
            self._state = "IDLE"
            return events
        # Fire when the swing is clearly past its peak: the forward rate lost fire_drop of it (a quick hesitation on
        # the way up, which real swings have, loses far less), or the peak is confirm_s old and the rate is below it.
        over = s <= (1.0 - p.fire_drop) * pk["w"] or (t - pk["t"] >= p.confirm_s and s < pk["w"])
        if pk["w"] >= p.t_pk and over and (t - self._onset) >= p.min_swing_s:
            events += self._fire(sample, t)
        return events

    def _fire(self, sample, t):
        p, pk = self.p, self._pk
        self._state, self._cool_until, self._fired_t = "COOL", t + p.refractory_s, t
        spike = pk["prev_w"] < p.spike_ratio * pk["w"] or (
            pk["next_w"] is not None and pk["next_w"] < p.spike_ratio * pk["w"])
        t_pk, w_pk = self._interpolate_peak()
        dur_ms = 2.0 * (t_pk - self._onset) * 1000.0
        if spike or not (p.min_dur_ms <= dur_ms <= p.max_dur_ms):
            return []
        reversals = count_reversals(self._history, self._onset, 1, p, w_pk)
        if reversals > p.max_reversals:
            return []
        back = max((-s for tt, s in self._history if self._onset - 0.5 <= tt < self._onset and s < 0), default=0.0)
        a_lin = tuple((pk["a"][k] - self._gravity[k]) / p.accel_per_g for k in range(3))
        feat = swing_features(pk["g"], self._net, a_lin, w_pk, dur_ms, min(2.0, back / w_pk))
        return [self._event("IMPACT", int(round(t_pk * 1e9)), w_pk, dur_ms, reversals, pk["g"], self._net,
                            a_lin, self._clip_max >= 3, feat)]

    def _interpolate_peak(self):
        pk = self._pk
        y1, y0, y2 = pk["w"], pk["prev_w"], pk["next_w"]
        if y2 is None:
            return pk["t"], y1
        denom = y0 - 2.0 * y1 + y2
        if denom >= -1e-9:                       # flat top / plateau: no interpolation
            return pk["t"], y1
        h = 0.5 * ((pk["t"] - pk["prev_t"]) + (pk["next_t"] - pk["t"]))
        offset = 0.5 * (y0 - y2) / denom
        offset = max(-1.0, min(1.0, offset))
        # the peak lies between its neighbours: with uneven spacing (a Bluetooth gap) the vertex can fall outside
        t_peak = max(pk["prev_t"], min(pk["next_t"], pk["t"] + offset * h))
        return t_peak, y1 - 0.25 * (y0 - y2) * offset

    def _event(self, kind, t_ns, w_pk, dur_ms, reversals, g, net, a_lin, clipped, feat=None):
        return SwingEvent(kind=kind, t_ns=t_ns, w_pk=float(w_pk), dur_ms=float(dur_ms), n_reversals=reversals,
                          axis_unit=_unit(g), net_rot_unit=_unit(net), a_lin_unit=_unit(a_lin),
                          clipped=clipped, feat=tuple(feat) if feat is not None else (0.0,) * 12, src=self._src)
