"""HitJudge: every gate fails independently; each failure explains itself."""

import pytest

from pingpong import levels
from pingpong.events import PaddlePose, SwingEvent
from pingpong.judge import BallWindow, HitJudge
from pingpong.paddle import ReachBox

S = 1_000_000_000
CLUB = levels.LEVELS[2]
BOX = ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5)
T_C = 10 * S


def ball(ball_id=1, aim=(0.5, 0.5), level=CLUB, t_c=T_C):
    return BallWindow(ball_id=ball_id, t_c_ns=t_c, aim_ab=aim, level=level)


def swing(t_ns=T_C, w_pk=600.0, dur_ms=150.0, rev=0):
    return SwingEvent(kind="IMPACT", t_ns=t_ns, w_pk=w_pk, dur_ms=dur_ms, n_reversals=rev,
                      axis_unit=(1, 0, 0), net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1),
                      clipped=False, feat=(0.0,) * 12)


def poses(t_i, u=0.0, v=0.0, conf=0.9, n=6, hand="right"):
    """Samples spread over the approach window [t_i-0.30, t_i-0.05] plus one at detection."""
    out = [PaddlePose(t_scene_ns=int(t_i - 0.30 * S + k * 0.05 * S), u=u, v=v, conf=conf, hand=hand)
           for k in range(n)]
    out.append(PaddlePose(t_scene_ns=int(t_i + 0.02 * S), u=u, v=v, conf=conf, hand=hand))
    return out


def judge(**kw):
    return HitJudge(BOX, t_pk=250.0, **kw)


def gate(verdict, name):
    return {g.name: g for g in verdict.gates}[name]


def test_a_well_timed_well_placed_strong_swing_is_a_hit():
    v = judge().judge(swing(), ball(), poses(T_C), now_ns=T_C + S // 10)
    assert v.kind == "HIT"
    assert all(g.passed for g in v.gates)
    assert v.e_s == pytest.approx(0.0) and v.d_min_sw == pytest.approx(0.0)
    assert v.q_pos == 1.0


def test_j1_early_swings_are_ignored_not_punished():
    t_i = T_C - int(1.0 * S)
    v = judge().judge(swing(t_ns=t_i), ball(), poses(t_i), now_ns=t_i + S // 10)
    assert v.kind == "IGNORED" and not gate(v, "J1").passed


def test_j1_a_swing_after_the_late_window_is_rejected():
    t_i = T_C + int((CLUB.late_s + 0.05) * S)
    v = judge().judge(swing(t_ns=t_i), ball(), poses(t_i), now_ns=t_i + S // 10)
    assert v.kind == "REJECTED" and not gate(v, "J1").passed


def test_j1_window_edges_are_inclusive():
    edge_early = T_C - int(CLUB.early_s * S) + 1_000
    edge_late = T_C + int(CLUB.late_s * S) - 1_000
    assert judge().judge(swing(t_ns=edge_early), ball(), poses(edge_early), edge_early).kind == "HIT"
    assert judge().judge(swing(t_ns=edge_late), ball(), poses(edge_late), edge_late).kind == "HIT"


def test_j2_hand_far_from_the_ball_is_rejected_and_explains_itself():
    far = poses(T_C, u=0.9, v=0.4)
    v = judge().judge(swing(), ball(), far, now_ns=T_C)
    assert v.kind == "REJECTED"
    assert not gate(v, "J2").passed and "from the ball" in gate(v, "J2").note
    assert gate(v, "J1").passed and gate(v, "J3").passed


def test_j2_touch_then_swing_elsewhere_is_rejected():
    # the hand WAS on the ball during the approach, but at detection it is far away
    on_ball = poses(T_C)[:-1]
    away = PaddlePose(t_scene_ns=T_C + int(0.02 * S), u=0.9, v=0.45, conf=0.9, hand="right")
    v = judge().judge(swing(), ball(), on_ball + [away], now_ns=T_C)
    assert v.kind == "REJECTED" and not gate(v, "J2").passed


def test_j2_needs_confident_landmarks_on_at_least_two_frames():
    low = poses(T_C, conf=0.3)
    assert judge().judge(swing(), ball(), low, now_ns=T_C).kind == "REJECTED"
    one = [poses(T_C)[0], poses(T_C)[-1]]
    only_one_in_window = [PaddlePose(t_scene_ns=one[0].t_scene_ns, u=0.0, v=0.0, conf=0.9, hand="right"),
                          PaddlePose(t_scene_ns=T_C + 20_000_000, u=0.0, v=0.0, conf=0.9, hand="right")]
    assert judge().judge(swing(), ball(), only_one_in_window, now_ns=T_C).kind == "REJECTED"


def test_j3_a_weak_swing_is_rejected():
    v = judge().judge(swing(w_pk=200.0), ball(), poses(T_C), now_ns=T_C)
    assert v.kind == "REJECTED" and not gate(v, "J3").passed


def test_j3_duration_and_oscillation_limits():
    assert not gate(judge().judge(swing(dur_ms=30.0), ball(), poses(T_C), T_C), "J3").passed
    assert not gate(judge().judge(swing(dur_ms=2600.0), ball(), poses(T_C), T_C), "J3").passed
    assert not gate(judge().judge(swing(rev=3), ball(), poses(T_C), T_C), "J3").passed


def test_j3_accepts_the_swing_durations_measured_on_the_real_hub():
    # real swings built up for 0.25-0.7 s before their peak, so 2 x (onset -> peak) was 0.5-1.4 s
    for dur in (300.0, 700.0, 1400.0):
        assert gate(judge().judge(swing(dur_ms=dur), ball(), poses(T_C), T_C), "J3").passed, dur


def test_j4_cross_sensor_is_logged_only_and_never_blocks():
    v = judge().judge(swing(), ball(), poses(T_C), T_C)
    assert gate(v, "J4").passed and "logged" in gate(v, "J4").note


def test_j5_one_hit_per_ball_and_a_refractory_after_a_counted_hit():
    j = judge()
    assert j.judge(swing(), ball(1), poses(T_C), T_C).kind == "HIT"
    again = j.judge(swing(t_ns=T_C + int(0.1 * S)), ball(1), poses(T_C + int(0.1 * S)), T_C)
    assert again.kind == "REJECTED" and not gate(again, "J5").passed
    t2 = T_C + int(0.2 * S)
    other = j.judge(swing(t_ns=t2), ball(2, t_c=t2), poses(t2), t2)
    assert not gate(other, "J5").passed            # inside the 0.35 s refractory
    t3 = T_C + int(0.5 * S)
    later = j.judge(swing(t_ns=t3), ball(3, t_c=t3), poses(t3), t3)
    assert later.kind == "HIT"


def test_j5_at_most_three_hits_per_second():
    j = judge()
    hits = 0
    for k in range(6):
        t = T_C + int(k * 0.36 * S)
        v = j.judge(swing(t_ns=t), ball(k + 1, t_c=t), poses(t), t)
        hits += v.kind == "HIT"
    assert hits == 6                                  # 0.36 s spacing is 2.8 hits/s: all count
    j2 = judge()
    results = []
    for k in range(5):
        t = T_C + int(k * 0.30 * S)                  # faster than the refractory: J5 limits it
        results.append(j2.judge(swing(t_ns=t), ball(k + 1, t_c=t), poses(t), t).kind)
    assert results.count("HIT") <= 3


def test_j6_shake_lock_blocks_everything_until_it_expires():
    j = judge()
    j.lock_paddle(until_ns=T_C + S)
    v = j.judge(swing(), ball(), poses(T_C), T_C)
    assert v.kind == "REJECTED" and not gate(v, "J6").passed
    t = T_C + 2 * S
    assert j.judge(swing(t_ns=t), ball(2, t_c=t), poses(t), t).kind == "HIT"


def test_miss_deadline_includes_the_measured_detection_lag():
    j = judge(d95_s=0.12)
    assert j.miss_deadline_ns(ball()) == T_C + int((CLUB.late_s + 0.12) * S)


def test_a_late_swing_detected_after_the_plane_still_counts():
    t_i = T_C + int((CLUB.late_s - 0.05) * S)       # peak just inside the window ...
    now = T_C + int((CLUB.late_s + 0.06) * S)       # ... but detected after the plane
    v = judge().judge(swing(t_ns=t_i), ball(), poses(t_i), now_ns=now)
    assert v.kind == "HIT"
    assert now < judge().miss_deadline_ns(ball())


def _moving_hand(t_peak_ns, t_i, hz=30.0, span=0.5, amp=4.0):
    """A hand sweeping along u whose speed peaks at t_peak_ns (a bell-shaped speed profile, peak `amp` SW/s)."""
    import math

    out, u = [], 0.0
    n = int(span * hz)
    t0 = t_i - int(0.4 * S)
    for k in range(n):
        t = t0 + int(k * S / hz)
        speed = amp * math.exp(-(((t - t_peak_ns) / S) / 0.06) ** 2)
        u += speed / hz
        out.append(PaddlePose(t_scene_ns=t, u=u, v=0.0, conf=0.9, hand="right"))
    return out


def test_j4_shows_the_measured_offset_in_its_note_and_never_blocks():
    import re

    hand = _moving_hand(T_C - int(0.04 * S), T_C, amp=0.5)         # a small wiggle: the hand stays near the ball
    v = judge().judge(swing(), ball(), hand, now_ns=T_C)
    note = gate(v, "J4").note
    assert v.kind == "HIT" and gate(v, "J4").passed is True and "logged" in note
    assert abs(float(re.search(r"([+-]\d+) ms", note).group(1)) + 40) <= 20


def test_j4_reports_the_signed_offset_between_the_pose_speed_peak_and_the_imu_peak():
    from pingpong.judge import cross_sensor_offset_ms

    t_i = T_C
    for true_offset_ms in (-80, -40, 0, 60):
        hand = _moving_hand(t_i + int(true_offset_ms * 1e6), t_i)
        assert cross_sensor_offset_ms(hand, t_i) == pytest.approx(true_offset_ms, abs=20)


def test_j4_says_so_when_there_are_too_few_pose_frames_to_find_a_peak():
    from pingpong.judge import cross_sensor_offset_ms

    assert cross_sensor_offset_ms([], T_C) is None
    assert cross_sensor_offset_ms(poses(T_C)[:3], T_C) is None
    v = judge().judge(swing(), ball(), poses(T_C), now_ns=T_C)
    assert gate(v, "J4").passed is True and "not enough" in gate(v, "J4").note


def test_j4_flags_a_large_disagreement_in_its_note_without_rejecting_the_hit():
    hand = _moving_hand(T_C - int(0.25 * S), T_C, amp=0.5)          # the camera saw the peak 250 ms before the IMU
    v = judge().judge(swing(), ball(), hand, now_ns=T_C)
    assert v.kind == "HIT" and gate(v, "J4").passed is True and "disagree" in gate(v, "J4").note
