"""Per-shot speed, spin, aim, quality and the deterministic fault rule."""

import math

import pytest

from pingpong import levels, shot, strokepath

CLUB = levels.LEVELS[2]


def test_speed_map_is_clipped_monotone_and_hits_the_documented_endpoints():
    lo, hi = 300.0, 1200.0
    assert shot.swing_strength(250.0, lo, hi) == 0.0
    assert shot.swing_strength(1500.0, lo, hi) == 1.0
    vals = [shot.swing_strength(w, lo, hi) for w in range(300, 1201, 50)]
    assert vals == sorted(vals)
    assert shot.out_speed(0.0) == pytest.approx(3.0)
    assert shot.out_speed(1.0) == pytest.approx(14.0)
    assert shot.out_speed(0.5) < shot.out_speed(0.6)


def test_kmh_is_3_6_times_ms():
    assert shot.kmh(10.0) == pytest.approx(36.0)


def test_degenerate_calibration_does_not_divide_by_zero():
    assert shot.swing_strength(500.0, 400.0, 400.0) in (0.0, 1.0)


def test_quality_blends_position_and_timing_equally():
    q_pos, q_time, q_total = shot.quality(d_min_sw=0.0, e_s=0.0, level=CLUB)
    assert (q_pos, q_time, q_total) == (1.0, 1.0, 1.0)
    q_pos, q_time, q_total = shot.quality(d_min_sw=CLUB.radius_sw, e_s=0.0, level=CLUB)
    assert q_pos == 0.0 and q_time == 1.0 and q_total == pytest.approx(0.5)


def test_early_uses_the_early_window_and_late_uses_the_late_window():
    _, q_early, _ = shot.quality(0.0, -CLUB.early_s / 2, CLUB)
    _, q_late, _ = shot.quality(0.0, CLUB.late_s / 2, CLUB)
    assert q_early == pytest.approx(0.5) and q_late == pytest.approx(0.5)


def test_grades_perfect_good_early_late():
    assert shot.grade(0.95, 0.0, CLUB) == "perfect"
    assert shot.grade(0.7, 0.0, CLUB) == "good"
    assert shot.grade(0.7, -0.6 * CLUB.early_s, CLUB) == "early"
    assert shot.grade(0.7, 0.6 * CLUB.late_s, CLUB) == "late"


def test_a_perfect_hit_can_never_fault_at_any_level_or_strength():
    for level in levels.LEVELS.values():
        for s in [i / 20 for i in range(21)]:
            assert shot.fault_for(s, q_total=0.9, level=level) is None


def test_a_sloppy_smash_faults_out_and_a_sloppy_medium_hit_faults_into_the_net_at_pro():
    pro = levels.LEVELS[3]
    assert shot.fault_for(1.0, 0.4, pro) == "out"         # 1.0 * 0.6 = 0.6 > 0.55, s > 0.7
    assert shot.fault_for(0.65, 0.0, pro) == "net"        # 0.65 * 1.0 = 0.65 > 0.55, s <= 0.7
    assert shot.fault_for(0.6, 0.1, pro) is None          # 0.6 * 0.9 = 0.54: a mildly sloppy hit does not fault since the levels were eased
    assert shot.fault_for(0.6, 0.9, pro) is None


def test_a_beginner_level_never_faults_however_sloppy_and_hard_the_swing():
    for level in (levels.LEVELS[1], levels.LEVELS[2]):
        for s in (0.2, 0.6, 1.0):
            assert shot.fault_for(s, 0.0, level) is None, (level.name, s)


def test_fault_is_a_pure_function_of_its_inputs():
    assert shot.fault_for(0.8, 0.3, CLUB) == shot.fault_for(0.8, 0.3, CLUB)


def test_spin_from_class_probabilities_scales_with_strength():
    probs = {"flat": 0.1, "top": 0.8, "back": 0.1}
    t, s, a = shot.spin_from_probs(probs, strength=1.0)
    assert a == pytest.approx(1.0) and t == pytest.approx(0.7) and s == 0.0
    t2, _, a2 = shot.spin_from_probs(probs, strength=0.0)
    assert a2 == pytest.approx(0.4) and t2 == pytest.approx(0.7 * 0.4)


def test_uncertain_or_missing_classifier_means_flat():
    assert shot.spin_from_probs({"flat": 0.4, "top": 0.3, "back": 0.3}, 1.0) == (0.0, 0.0, 0.0)
    assert shot.spin_from_probs(None, 1.0) == (0.0, 0.0, 0.0)


def test_aim_is_the_hand_position_scaled_and_clipped():
    assert shot.aim_from_a(0.5) == 0.0
    assert shot.aim_from_a(0.0) == pytest.approx(-0.8)
    assert shot.aim_from_a(1.0) == pytest.approx(0.8)
    assert shot.aim_from_a(2.0) == 1.0 or shot.aim_from_a(2.0) <= 1.0


def test_make_assembles_a_shot_from_one_verdict():
    p = shot.make(w_pk=900.0, omega_lo=300.0, omega_hi=1200.0, d_min_sw=0.0, e_s=0.0, level=CLUB,
                  paddle_a=0.5, spin_probs=None)
    assert p.v_out == pytest.approx(shot.out_speed(shot.swing_strength(900.0, 300.0, 1200.0)))
    assert p.fault is None and p.label == "perfect"
    assert p.q_total == 1.0


# --- the path of the hand and the twist of the hub shape the return -----------------------------------------------------------
def made(stroke=None, paddle_a=0.5, probs=None):
    return shot.make(w_pk=600.0, omega_lo=300.0, omega_hi=1200.0, d_min_sw=0.1, e_s=0.0, level=CLUB, paddle_a=paddle_a,
                     spin_probs=probs, stroke=stroke)


def test_a_stroke_gives_the_ball_its_spin_and_the_amplitude_is_the_size_of_it():
    stroke = strokepath.ReturnShape(aim_shift=0.0, loft_m=0.1, topspin=0.3, sidespin=-0.2)
    sp = made(stroke)
    assert sp.T == pytest.approx(0.3) and sp.S == pytest.approx(-0.2) and sp.A == pytest.approx(math.hypot(0.3, 0.2))
    flat = made()
    assert (flat.T, flat.S, flat.A) == (0.0, 0.0, 0.0)


def test_a_stroke_adds_to_the_spin_a_trained_model_hears_and_the_sum_stays_within_one():
    stroke = strokepath.ReturnShape(aim_shift=0.0, loft_m=0.0, topspin=0.5, sidespin=0.0)
    trained = made(probs={"flat": 0.05, "top": 0.9, "back": 0.05})
    both = made(stroke, probs={"flat": 0.05, "top": 0.9, "back": 0.05})
    assert both.T == pytest.approx(min(1.0, trained.T + 0.5)) and both.T > trained.T and both.A >= trained.A


def test_a_stroke_moves_the_landing_point_across_the_table_and_it_stays_on_the_table():
    moved = made(strokepath.ReturnShape(aim_shift=0.1, loft_m=0.0, topspin=0.0, sidespin=0.0))
    assert moved.aim_a == pytest.approx(made().aim_a + 0.1 * 1.6)
    far = made(strokepath.ReturnShape(aim_shift=0.3, loft_m=0.0, topspin=0.0, sidespin=0.0), paddle_a=0.95)
    assert far.aim_a == pytest.approx(shot.aim_from_a(1.0))
    left = made(strokepath.ReturnShape(aim_shift=-0.3, loft_m=0.0, topspin=0.0, sidespin=0.0), paddle_a=0.05)
    assert left.aim_a == pytest.approx(shot.aim_from_a(0.0))

