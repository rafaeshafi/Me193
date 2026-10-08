"""HitJudge: every gate fails independently; each failure explains itself.

The moment a swing is judged is its CONTACT: the gyro's peak plus the stroke's lag (the forward stroke of a swing ends
about 0.1 s after the peak of its rate, which is when the player means the paddle to meet the ball).  A swing whose peak
is LAG before the ball's t_c therefore makes contact exactly on time.  Position is judged across the court only: how
high the hand is does not matter.
"""

import pytest

from pingpong import latency, levels, physics
from pingpong.events import PaddlePose, SwingEvent
from pingpong.judge import BallWindow, HitJudge
from pingpong.paddle import ReachBox

S = 1_000_000_000
CLUB = levels.LEVELS[2]
BOX = ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5)
T_C = 10 * S
LAG = latency.Latency.from_config().contact_lag_s       # the judge's default peak -> contact lag, seconds
T_I = T_C - round(LAG * S)                   # the peak of a perfectly timed swing


def ball(ball_id=1, aim=(0.5, 0.5), level=CLUB, t_c=T_C, leg=None):
    return BallWindow(ball_id=ball_id, t_c_ns=t_c, aim_ab=aim, level=level, leg=leg)


def swing(t_ns=T_I, w_pk=600.0, dur_ms=150.0, rev=0):
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
    v = judge().judge(swing(), ball(), poses(T_I), now_ns=T_C + S // 10)
    assert v.kind == "HIT"
    assert all(g.passed for g in v.gates)
    assert v.e_s == pytest.approx(0.0) and v.d_min_sw == pytest.approx(0.0)
    assert v.q_pos == 1.0


def test_the_contact_is_the_peak_plus_the_stroke_lag_and_never_before_the_swing_was_seen():
    j = judge()
    assert j.contact_lag_s == pytest.approx(LAG)
    on_time = j.judge(swing(), ball(1), poses(T_I), now_ns=T_I + 10_000_000)       # seen long before the contact
    assert on_time.e_s == pytest.approx(0.0) and on_time.contact_ns == T_C
    t_i = T_C + S
    seen_late = judge().judge(swing(t_ns=t_i), ball(2, t_c=t_i + round(LAG * S)), poses(t_i),
                              now_ns=t_i + round((LAG + 0.07) * S))
    assert seen_late.contact_ns == t_i + round((LAG + 0.07) * S)                     # the ball is where it is NOW, not in the past
    assert seen_late.e_s == pytest.approx(0.0, abs=1e-9)                              # ... but the timing is judged on the stroke


def test_j1_early_swings_are_ignored_not_punished():
    t_i = T_C - int(1.0 * S)
    v = judge().judge(swing(t_ns=t_i), ball(), poses(t_i), now_ns=t_i + S // 10)
    assert v.kind == "IGNORED" and not gate(v, "J1").passed


def test_j1_a_swing_after_the_late_window_is_rejected():
    t_i = T_C + int((CLUB.late_s + 0.05) * S)
    v = judge().judge(swing(t_ns=t_i), ball(), poses(t_i), now_ns=t_i + S // 10)
    assert v.kind == "REJECTED" and not gate(v, "J1").passed


def test_j1_window_edges_are_inclusive():
    edge_early = T_I - int(CLUB.early_s * S) + 1_000
    edge_late = T_I + int(CLUB.late_s * S) - 1_000
    assert judge().judge(swing(t_ns=edge_early), ball(), poses(edge_early), edge_early).kind == "HIT"
    assert judge().judge(swing(t_ns=edge_late), ball(), poses(edge_late), edge_late).kind == "HIT"
    just_early = T_I - int(CLUB.early_s * S) - 1_000
    assert judge().judge(swing(t_ns=just_early), ball(), poses(just_early), just_early).kind == "IGNORED"


def test_j2_hand_far_from_the_ball_is_rejected_and_explains_itself():
    far = poses(T_I, u=0.9, v=0.4)
    v = judge().judge(swing(), ball(), far, now_ns=T_C)
    assert v.kind == "REJECTED"
    assert not gate(v, "J2").passed and "from the ball" in gate(v, "J2").note
    assert gate(v, "J1").passed and gate(v, "J3").passed


def test_j2_touch_then_swing_elsewhere_is_rejected():
    # the hand WAS on the ball during the approach, but at the impact it is far away
    on_ball = poses(T_I)[:-1]
    away = PaddlePose(t_scene_ns=T_I + int(0.02 * S), u=1.6 * CLUB.radius_sw + 0.2, v=0.45, conf=0.9, hand="right")      # beyond 1.6 radii
    v = judge().judge(swing(), ball(), on_ball + [away], now_ns=T_C)
    assert v.kind == "REJECTED" and not gate(v, "J2").passed


def test_j2_needs_confident_landmarks_on_at_least_two_frames():
    low = poses(T_I, conf=0.3)
    assert judge().judge(swing(), ball(), low, now_ns=T_C).kind == "REJECTED"
    one = [poses(T_I)[0], poses(T_I)[-1]]
    only_one_in_window = [PaddlePose(t_scene_ns=one[0].t_scene_ns, u=0.0, v=0.0, conf=0.9, hand="right"),
                          PaddlePose(t_scene_ns=T_C + 80_000_000, u=0.0, v=0.0, conf=0.9, hand="right")]       # after the window's end
    assert judge().judge(swing(), ball(), only_one_in_window, now_ns=T_C).kind == "REJECTED"


def test_j2_looks_across_the_court_only_how_high_the_hand_is_does_not_matter():
    # a hand level with the ball sideways but a whole box-height above or below where the old plane put it
    for v_hand in (-0.5, 0.0, 0.5):
        v = judge().judge(swing(), ball(aim=(0.5, 0.85)), poses(T_I, u=0.0, v=v_hand), now_ns=T_C)
        assert v.kind == "HIT" and v.d_min_sw == pytest.approx(0.0), v_hand


def test_j2_measures_against_where_the_ball_is_at_the_contact_when_it_has_a_flight():
    # the ball is still moving sideways: from x = +0.75 m towards the middle, it is not yet at its aim when you swing early
    pro = levels.LEVELS[4]                                      # (a fast ball, as Pro was before the levels were eased: it moves sideways quickly)
    leg = physics.plan_leg(T_C - round(3.0 / pro.v_tier * S), pro.v_tier, 0.75, (0.5, 0.5))
    early = T_I - round(0.18 * S)
    x_there = leg.position(early + round(LAG * S))[0]
    assert abs(x_there) > 0.25
    u_there = BOX.to_uv(physics.a_of_x(x_there), 0.5)[0]
    assert u_there > pro.radius_sw + 0.05                       # so the hand at the aim (u = 0) is clearly not on the ball
    on_the_ball = judge().judge(swing(t_ns=early), ball(level=pro, leg=leg), poses(early, u=u_there), now_ns=early + S // 10)
    at_the_aim = judge().judge(swing(t_ns=early), ball(level=pro, leg=leg), poses(early, u=0.0), now_ns=early + S // 10)
    assert on_the_ball.kind == "HIT" and on_the_ball.d_min_sw == pytest.approx(0.0, abs=0.02)
    assert at_the_aim.kind == "REJECTED" and not gate(at_the_aim, "J2").passed


def test_j3_a_weak_swing_is_rejected():
    v = judge().judge(swing(w_pk=200.0), ball(), poses(T_I), now_ns=T_C)
    assert v.kind == "REJECTED" and not gate(v, "J3").passed


def test_j3_duration_and_oscillation_limits():
    assert not gate(judge().judge(swing(dur_ms=30.0), ball(), poses(T_I), T_C), "J3").passed
    assert not gate(judge().judge(swing(dur_ms=2600.0), ball(), poses(T_I), T_C), "J3").passed
    assert not gate(judge().judge(swing(rev=3), ball(), poses(T_I), T_C), "J3").passed


def test_j3_accepts_the_swing_durations_measured_on_the_real_hub():
    # real swings built up for 0.25-0.7 s before their peak, so 2 x (onset -> peak) was 0.5-1.4 s
    for dur in (300.0, 700.0, 1400.0):
        assert gate(judge().judge(swing(dur_ms=dur), ball(), poses(T_I), T_C), "J3").passed, dur


def test_j4_cross_sensor_is_logged_only_and_never_blocks():
    v = judge().judge(swing(), ball(), poses(T_I), T_C)
    assert gate(v, "J4").passed and "logged" in gate(v, "J4").note


def test_j5_one_hit_per_ball_and_a_refractory_after_a_counted_hit():
    j = judge()
    assert j.judge(swing(), ball(1), poses(T_I), T_C).kind == "HIT"
    again = j.judge(swing(t_ns=T_C + int(0.1 * S)), ball(1), poses(T_C + int(0.1 * S)), T_C)
    assert again.kind == "REJECTED" and not gate(again, "J5").passed
    t2 = T_C + int(0.05 * S)
    other = j.judge(swing(t_ns=t2), ball(2, t_c=t2), poses(t2), t2)
    assert not gate(other, "J5").passed            # inside the 0.35 s refractory (of the first swing, at T_I)
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
    v = j.judge(swing(), ball(), poses(T_I), T_C)
    assert v.kind == "REJECTED" and not gate(v, "J6").passed
    t = T_C + 2 * S
    assert j.judge(swing(t_ns=t), ball(2, t_c=t), poses(t), t).kind == "HIT"


def test_miss_deadline_includes_the_measured_detection_lag():
    j = judge(d95_s=0.12)
    assert j.miss_deadline_ns(ball()) == T_C + round((CLUB.late_s + 0.12) * S)


def test_a_late_swing_detected_after_the_plane_still_counts():
    t_i = T_I + int((CLUB.late_s - 0.05) * S)       # the stroke's end just inside the window ...
    now = T_C + int((CLUB.late_s + 0.06) * S)       # ... but detected after the plane
    v = judge().judge(swing(t_ns=t_i), ball(), poses(t_i), now_ns=now)
    assert v.kind == "HIT" and v.contact_ns == now
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

    hand = _moving_hand(T_I - int(0.04 * S), T_I, amp=0.5)         # a small wiggle: the hand stays near the ball
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
    v = judge().judge(swing(), ball(), poses(T_I), now_ns=T_C)
    assert gate(v, "J4").passed is True and "not enough" in gate(v, "J4").note


def test_j4_flags_a_large_disagreement_in_its_note_without_rejecting_the_hit():
    hand = _moving_hand(T_I - int(0.25 * S), T_I, amp=0.5)          # the camera saw the peak 250 ms before the IMU
    v = judge().judge(swing(), ball(), hand, now_ns=T_C)
    assert v.kind == "HIT" and gate(v, "J4").passed is True and "disagree" in gate(v, "J4").note


# --- the paddle's depth: the moment the ball is over YOUR paddle -------------------------------------------------------------
def test_the_ball_is_on_time_when_it_is_over_the_paddle_your_hand_height_puts_on_the_table():
    from pingpong import stage

    level = levels.LEVELS[1]
    leg = physics.plan_leg(T_C - round(3.0 / level.v_tier * S), level.v_tier, 0.0, (0.5, 0.5))
    t_high, t_low = leg.time_at_z(stage.rest_z(BOX, 0.5)), leg.time_at_z(stage.rest_z(BOX, -0.5))
    assert t_high < leg.arrival_ns < t_low                               # a paddle further up the table meets the ball sooner

    def judged(v_hand, t_contact):
        t_i = t_contact - round(LAG * S)
        return judge().judge(swing(t_ns=t_i), ball(level=level, leg=leg), poses(t_i, v=v_hand), now_ns=t_contact)

    on_high = judged(0.5, t_high)
    assert on_high.kind == "HIT" and abs(on_high.e_s) < 0.01             # timed to the ball reaching the HIGH paddle
    assert judged(-0.5, t_high).e_s < -0.05                              # the same swing is early for a paddle held back
    assert judged(-0.5, t_low).kind == "HIT" and abs(judged(-0.5, t_low).e_s) < 0.01


def test_the_miss_is_only_declared_when_no_paddle_position_could_still_reach_the_ball():
    from pingpong import stage

    level = levels.LEVELS[1]
    leg = physics.plan_leg(T_C - round(3.0 / level.v_tier * S), level.v_tier, 0.0, (0.5, 0.5))
    j = judge(d95_s=0.12)
    deadline = j.miss_deadline_ns(ball(level=level, leg=leg))
    assert deadline == leg.time_at_z(stage.Z_REST_MIN) + round((level.late_s + 0.12) * S)
    assert deadline > j.miss_deadline_ns(ball(level=level))              # later than the nominal one (no flight known)
