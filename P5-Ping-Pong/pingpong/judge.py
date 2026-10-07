"""HitJudge: a swing is a HIT only if every hard gate passes (J1-J6).

    J1 timing      contact in [t_c - E, t_c + L]   (earlier = ignored practice swing).  The contact is the gyro's
                   peak plus the stroke's lag: the forward stroke of a swing ENDS ~0.1 s after the peak of its rate, and
                   that is when the player means the paddle to meet the ball (the first live games: the peaks came a
                   median 0.25 s before the ball's nominal arrival, the stroke's ends right on it)
    J2 pose        the hand is level with the ball ACROSS the court (how high it is does not matter) over
                   [t_i-0.30, t_i+0.05] (the impact included: a stroke sweeps ~10 SW/s, so the hand is within reach
                   only at the moment of impact) AND at the impact (closes "touch, then swing elsewhere"; the newest
                   pose is no use here: detection comes up to 150 ms after the peak, when the hand is in the
                   follow-through).  Where the ball is comes from its flight, at the contact.
    J3 swing       peak, duration and oscillation limits
    J4 cross-sensor (logged only; made hard only after measuring false rejects)
    J5 refractory  one hit per ball, 0.35 s after a counted hit, <= 3 hits per second
    J6 shake lock  paddle locked after a shake

Every verdict carries per-gate results with a note, so the x-ray HUD can show WHY.
A late peak detected after the plane still counts: the miss is only declared at
t_c + L + D95 (D95 = measured p95 detection lag: 150 ms on the real hub, whose swings are slow lobes that
are only recognised once they have fallen a third from their peak; their duration is 0.5-1.4 s).
"""

import math
from collections import deque
from dataclasses import dataclass

from pingpong import latency as latency_mod
from pingpong import physics, stage
from pingpong.events import GateResult, Verdict

S = 1_000_000_000
MIN_PEAK_SPEED_SW_S = 0.3        # a hand that barely moves has no speed peak worth comparing


@dataclass(frozen=True)
class BallWindow:
    ball_id: int
    t_c_ns: int          # when the ball is over the sweet spot
    aim_ab: tuple        # arrival point in reach-box coordinates (a: across the court; b: how deep it bounced)
    level: object        # pingpong.levels.Level
    leg: object = None   # its flight (physics.Leg): where the ball is at any moment; without one it arrives at aim a


def pose_at(samples, t_ns, *, before_s=0.10, after_s=0.05, min_conf=0.0):
    """The pose reading nearest to t_ns, within [t_ns - before_s, t_ns + after_s]: where the hand was at that
    moment (None if there is none).  A swing is only recognised 30-150 ms after its peak, so the newest reading is
    the follow-through, not the impact."""
    near = [p for p in samples if -int(before_s * S) <= p.t_scene_ns - t_ns <= int(after_s * S) and p.conf >= min_conf]
    return min(near, key=lambda p: abs(p.t_scene_ns - t_ns), default=None)


def cross_sensor_offset_ms(samples, t_i_ns):
    """How far (ms, signed) the camera's hand-speed peak is from the IMU's swing peak t_i; None if unknowable.

    Looks at poses from 0.5 s before to 0.15 s after the IMU peak.  Needs five readings and a hand that
    actually moves; negative = the camera saw its peak first.
    """
    window = [p for p in samples if t_i_ns - int(0.5 * S) <= p.t_scene_ns <= t_i_ns + int(0.15 * S)]
    if len(window) < 5:
        return None
    times = [(a.t_scene_ns + b.t_scene_ns) / 2 for a, b in zip(window, window[1:])]
    speeds = [math.hypot(b.u - a.u, b.v - a.v) / max(1e-6, (b.t_scene_ns - a.t_scene_ns) / S)
              for a, b in zip(window, window[1:])]
    smooth = [sum(speeds[max(0, k - 1):k + 2]) / len(speeds[max(0, k - 1):k + 2]) for k in range(len(speeds))]
    k = max(range(len(smooth)), key=smooth.__getitem__)
    if smooth[k] < MIN_PEAK_SPEED_SW_S:
        return None
    return (times[k] - t_i_ns) / 1e6


class HitJudge:
    def __init__(self, box, t_pk=250.0, d95_s=0.15, min_dur_ms=60.0, max_dur_ms=2000.0, max_reversals=2,
                 min_conf=0.6, refractory_s=0.35, max_hits_per_s=3, contact_lag_s=None):
        self.box, self.t_pk, self.d95_s = box, t_pk, d95_s
        self.contact_lag_s = latency_mod.Latency.from_config().contact_lag_s if contact_lag_s is None else contact_lag_s
        self.min_dur_ms, self.max_dur_ms, self.max_reversals = min_dur_ms, max_dur_ms, max_reversals
        self.min_conf, self.refractory_s, self.max_hits_per_s = min_conf, refractory_s, max_hits_per_s
        self._counted = set()
        self._hit_times_ns = deque()
        self._locked_until_ns = 0

    def lock_paddle(self, until_ns):
        self._locked_until_ns = max(self._locked_until_ns, until_ns)

    def sweet_ns(self, ball, v):
        """When the ball is over the paddle that a hand at height v puts on the table (the nominal arrival without a flight)."""
        if ball.leg is None or v is None:
            return ball.t_c_ns
        return ball.leg.time_at_z(stage.rest_z(self.box, v))

    def miss_deadline_ns(self, ball):
        """Only when no paddle position could still reach the ball: the one furthest back is the last to meet it."""
        t_ns = ball.t_c_ns if ball.leg is None else ball.leg.time_at_z(stage.Z_REST_MIN)
        return t_ns + round((ball.level.late_s + self.d95_s) * 1e9)

    def judge(self, swing, ball, pose_samples, now_ns):
        level = ball.level
        intended_ns = swing.t_ns + round(self.contact_lag_s * 1e9)      # when the stroke means the paddle to meet the ball
        contact_ns = max(now_ns, intended_ns)                           # ... and the ball is where it is NOW: never the past
        impact = pose_at(pose_samples, swing.t_ns, min_conf=self.min_conf)     # where the hand put the paddle, and how far up
        e_s = (intended_ns - self.sweet_ns(ball, None if impact is None else impact.v)) / 1e9
        if e_s < -level.early_s:
            note = "early cue" if e_s >= -2 * level.early_s else "early: practice swing, ignored"
            return Verdict("IGNORED", 0.0, e_s, 0.0, (GateResult("J1", False, note),), contact_ns)

        gates = [GateResult("J1", e_s <= level.late_s, f"timing {e_s * 1000:+.0f} ms (window "
                            f"-{level.early_s * 1000:.0f}/+{level.late_s * 1000:.0f})")]
        j2, d_min = self._pose_gate(swing, ball, pose_samples, contact_ns)
        gates += [j2, self._swing_gate(swing), self._cross_gate(swing, pose_samples),
                  self._refractory_gate(swing, ball), self._lock_gate(swing)]
        q_pos = max(0.0, min(1.0, 1.0 - d_min / level.radius_sw)) if math.isfinite(d_min) else 0.0
        hard = [g for g in gates if g.name != "J4"]
        if all(g.passed for g in hard):
            self._counted.add(ball.ball_id)
            self._hit_times_ns.append(swing.t_ns)
            return Verdict("HIT", q_pos, e_s, d_min, tuple(gates), contact_ns)
        return Verdict("REJECTED", q_pos, e_s, d_min if math.isfinite(d_min) else 0.0, tuple(gates), contact_ns)

    # --- gates -------------------------------------------------------------------------
    def ball_u(self, ball, t_ns):
        """Where the ball is across the court at t_ns, in the hand's own units (shoulder widths)."""
        x = ball.leg.position(t_ns)[0] if ball.leg is not None else physics.x_of_a(ball.aim_ab[0])
        return self.box.to_uv(physics.a_of_x(x), 0.5)[0]

    def _pose_gate(self, swing, ball, samples, contact_ns):
        R = ball.level.radius_sw
        bu = self.ball_u(ball, contact_ns)
        lo, hi = swing.t_ns - int(0.30 * S), swing.t_ns + int(0.05 * S)
        visible = [p for p in samples if lo <= p.t_scene_ns <= hi and p.conf >= self.min_conf]
        dist = lambda p: abs(p.u - bu)                            # noqa: E731  (across the court: the height is free)
        d_min = min((dist(p) for p in visible), default=math.inf)
        if len(visible) < 2:
            return GateResult("J2", False, f"only {len(visible)} confident pose frame(s) in the approach window"), d_min
        if d_min > R:
            return GateResult("J2", False, f"hand {d_min:.2f} SW from the ball (limit {R:.2f})"), d_min
        at_impact = pose_at(visible, swing.t_ns)
        if at_impact is None or dist(at_impact) > 1.6 * R:
            return GateResult("J2", False, "hand moved away from the ball before the swing"), d_min
        return GateResult("J2", True, f"hand {d_min:.2f} SW from the ball (limit {R:.2f})"), d_min

    def _cross_gate(self, swing, samples):
        """J4 is LOGGED ONLY: it measures whether camera and IMU saw the swing at the same moment."""
        if swing.src == "pose":
            return GateResult("J4", True, "the camera is the swing sensor: nothing to compare (logged)")
        offset = cross_sensor_offset_ms(samples, swing.t_ns)
        if offset is None:
            return GateResult("J4", True, "not enough pose frames to compare (logged)")
        flag = ": disagree" if abs(offset) > 150 else ""
        return GateResult("J4", True, f"pose peak {offset:+.0f} ms vs IMU{flag} (logged)")

    def _swing_gate(self, swing):
        if swing.w_pk < self.t_pk:
            return GateResult("J3", False, f"swing too weak ({swing.w_pk:.0f} < {self.t_pk:.0f} dps)")
        if not self.min_dur_ms <= swing.dur_ms <= self.max_dur_ms:
            return GateResult("J3", False, f"swing duration {swing.dur_ms:.0f} ms out of range")
        if swing.n_reversals > self.max_reversals:
            return GateResult("J3", False, f"{swing.n_reversals} reversals: looks like waving")
        return GateResult("J3", True, f"{swing.w_pk:.0f} dps, {swing.dur_ms:.0f} ms")

    def _refractory_gate(self, swing, ball):
        if ball.ball_id in self._counted:
            return GateResult("J5", False, "this ball was already hit")
        if self._hit_times_ns and swing.t_ns - self._hit_times_ns[-1] < self.refractory_s * S:
            return GateResult("J5", False, "refractory: too soon after the last hit")
        recent = sum(1 for t in self._hit_times_ns if swing.t_ns - t <= S)
        if recent >= self.max_hits_per_s:
            return GateResult("J5", False, f"rate limit: {recent} hits in the last second")
        return GateResult("J5", True, "")

    def _lock_gate(self, swing):
        if swing.t_ns < self._locked_until_ns:
            return GateResult("J6", False, "paddle locked after shaking")
        return GateResult("J6", True, "")
