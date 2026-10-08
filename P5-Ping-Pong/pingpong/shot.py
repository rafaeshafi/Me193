"""One shot from one hit: speed, spin, aim, quality and the deterministic fault rule.

Everything here is a pure function so a fault is explainable ("you swung too hard
and too sloppily") and testable -- the seeded RNG is only used for the CPU.
"""

import math

from pingpong.events import ShotParams

PERFECT_Q = 0.9


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def swing_strength(w_pk, omega_lo, omega_hi):
    """0..1 from the peak gyro rate between the player's soft and full calibration swings."""
    if omega_hi <= omega_lo:
        return 1.0 if w_pk >= omega_hi else 0.0
    return _clip((w_pk - omega_lo) / (omega_hi - omega_lo), 0.0, 1.0)


def out_speed(s):
    """Player -> CPU ball speed in m/s: 3 m/s for a soft swing up to 14 m/s for a full one."""
    return 3.0 + 11.0 * s ** 0.8


def kmh(v_ms):
    return 3.6 * v_ms


def quality(d_min_sw, e_s, level):
    """-> (q_pos, q_time, q_total): position and timing blended equally."""
    q_pos = _clip(1.0 - d_min_sw / level.radius_sw, 0.0, 1.0)
    window = level.early_s if e_s < 0 else level.late_s
    q_time = _clip(1.0 - abs(e_s) / window, 0.0, 1.0)
    return q_pos, q_time, 0.5 * q_pos + 0.5 * q_time


def grade(q_total, e_s, level):
    if q_total >= PERFECT_Q:
        return "perfect"
    if e_s < -0.55 * level.early_s:
        return "early"
    if e_s > 0.55 * level.late_s:
        return "late"
    return "good"


def fault_for(s, q_total, level):
    """None, "net" or "out": a hard AND sloppy swing faults; a perfect hit never does."""
    if q_total >= PERFECT_Q:
        return None
    if s * (1.0 - q_total) > level.fault_th:
        return "out" if s > 0.7 else "net"
    return None


def spin_from_probs(probs, strength):
    """Class probabilities (flat/top/back[/left/right]) -> (topspin, sidespin, amplitude).

    Uncertain (max probability < 0.5) or no classifier means a flat ball.
    """
    if not probs or max(probs.values()) < 0.5:
        return 0.0, 0.0, 0.0
    amp = 0.4 + 0.6 * strength
    top = (probs.get("top", 0.0) - probs.get("back", 0.0)) * amp
    side = (probs.get("right", 0.0) - probs.get("left", 0.0)) * amp
    return top, side, amp


def aim_from_a(a):
    """Hand position in the reach box (0..1) -> lateral aim -1..1 (the edges are not reachable)."""
    return _clip((a - 0.5) * 2 * 0.8, -1.0, 1.0)


def make(*, w_pk, omega_lo, omega_hi, d_min_sw, e_s, level, paddle_a, spin_probs=None, stroke=None, strength=None):
    """stroke: a strokepath.ReturnShape (the path of the hand and the flick of the wrist): it moves the landing point and
    adds its spin to what a trained spin model hears (its loft is the physics', not part of the shot's parameters).
    strength: 0..1 given outright (a hand that met the ball); otherwise the swing's own from w_pk."""
    s = swing_strength(w_pk, omega_lo, omega_hi) if strength is None else strength
    _, _, q_total = quality(d_min_sw, e_s, level)
    top, side, amp = spin_from_probs(spin_probs, s)
    if stroke is not None:
        top, side = _clip(top + stroke.topspin, -1.0, 1.0), _clip(side + stroke.sidespin, -1.0, 1.0)
        amp = max(amp, min(1.0, math.hypot(top, side)))
        paddle_a = _clip(paddle_a + stroke.aim_shift, 0.0, 1.0)
    return ShotParams(v_out=out_speed(s), T=top, S=side, A=amp, aim_a=aim_from_a(paddle_a),
                      q_total=q_total, fault=fault_for(s, q_total, level),
                      label=grade(q_total, e_s, level))
