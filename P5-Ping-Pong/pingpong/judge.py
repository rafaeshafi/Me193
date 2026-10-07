"""HitJudge: a swing is a HIT only if every hard gate passes (J1-J6).

    J1 timing      t_i in [t_c - E, t_c + L]   (earlier = ignored practice swing)
    J2 pose        hand near the ball over the approach window [t_i-0.30, t_i-0.05]
                   AND still near it at detection (closes "touch, then swing elsewhere")
    J3 swing       peak, duration and oscillation limits
    J4 cross-sensor (logged only; made hard only after measuring false rejects)
    J5 refractory  one hit per ball, 0.35 s after a counted hit, <= 3 hits per second
    J6 shake lock  paddle locked after a shake

Every verdict carries per-gate results with a note, so the x-ray HUD can show WHY.
A late peak detected after the plane still counts: the miss is only declared at
t_c + L + D95 (D95 = measured p95 detection lag).
"""

import math
from collections import deque
from dataclasses import dataclass

from pingpong.events import GateResult, Verdict

S = 1_000_000_000


@dataclass(frozen=True)
class BallWindow:
    ball_id: int
    t_c_ns: int
    aim_ab: tuple        # arrival point in reach-box coordinates
    level: object        # pingpong.levels.Level


class HitJudge:
    def __init__(self, box, t_pk=250.0, d95_s=0.12, min_dur_ms=60.0, max_dur_ms=400.0, max_reversals=2,
                 min_conf=0.6, refractory_s=0.35, max_hits_per_s=3):
        self.box, self.t_pk, self.d95_s = box, t_pk, d95_s
        self.min_dur_ms, self.max_dur_ms, self.max_reversals = min_dur_ms, max_dur_ms, max_reversals
        self.min_conf, self.refractory_s, self.max_hits_per_s = min_conf, refractory_s, max_hits_per_s
        self._counted = set()
        self._hit_times_ns = deque()
        self._locked_until_ns = 0

    def lock_paddle(self, until_ns):
        self._locked_until_ns = max(self._locked_until_ns, until_ns)

    def miss_deadline_ns(self, ball):
        return ball.t_c_ns + round((ball.level.late_s + self.d95_s) * 1e9)

    def judge(self, swing, ball, pose_samples, now_ns):
        level = ball.level
        e_s = (swing.t_ns - ball.t_c_ns) / 1e9
        if e_s < -level.early_s:
            note = "early cue" if e_s >= -2 * level.early_s else "early: practice swing, ignored"
            return Verdict("IGNORED", 0.0, e_s, 0.0, (GateResult("J1", False, note),))

        gates = [GateResult("J1", e_s <= level.late_s, f"timing {e_s * 1000:+.0f} ms (window "
                            f"-{level.early_s * 1000:.0f}/+{level.late_s * 1000:.0f})")]
        j2, d_min = self._pose_gate(swing, ball, pose_samples)
        gates += [j2, self._swing_gate(swing), GateResult("J4", True, "logged-only"),
                  self._refractory_gate(swing, ball), self._lock_gate(swing)]
        q_pos = max(0.0, min(1.0, 1.0 - d_min / level.radius_sw)) if math.isfinite(d_min) else 0.0
        hard = [g for g in gates if g.name != "J4"]
        if all(g.passed for g in hard):
            self._counted.add(ball.ball_id)
            self._hit_times_ns.append(swing.t_ns)
            return Verdict("HIT", q_pos, e_s, d_min, tuple(gates))
        return Verdict("REJECTED", q_pos, e_s, d_min if math.isfinite(d_min) else 0.0, tuple(gates))

    # --- gates -------------------------------------------------------------------------
    def _pose_gate(self, swing, ball, samples):
        R = ball.level.radius_sw
        bu, bv = self.box.to_uv(*ball.aim_ab)
        lo, hi = swing.t_ns - int(0.30 * S), swing.t_ns - int(0.05 * S)
        visible = [p for p in samples if lo <= p.t_scene_ns <= hi and p.conf >= self.min_conf]
        dist = lambda p: math.hypot(p.u - bu, p.v - bv)           # noqa: E731
        d_min = min((dist(p) for p in visible), default=math.inf)
        latest = samples[-1] if samples else None
        if len(visible) < 2:
            return GateResult("J2", False, f"only {len(visible)} confident pose frame(s) in the approach window"), d_min
        if d_min > R:
            return GateResult("J2", False, f"hand {d_min:.2f} SW from the ball (limit {R:.2f})"), d_min
        if latest is None or latest.conf < self.min_conf or dist(latest) > 1.6 * R:
            return GateResult("J2", False, "hand moved away from the ball before the swing"), d_min
        return GateResult("J2", True, f"hand {d_min:.2f} SW from the ball (limit {R:.2f})"), d_min

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
