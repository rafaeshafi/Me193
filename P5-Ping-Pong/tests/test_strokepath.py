"""How the ball leaves the paddle: the path of the hand decides where it goes, how high and (when the hand into the ball is
the hit) how hard.  The spin is the wrist flick's (tests/test_flick.py)."""

import math

import pytest

from pingpong import strokepath
from pingpong.events import PaddlePose

S = 1_000_000_000
PEAK = 10 * S


def track(vu, vv, *, start=-0.35, stop=0.15, hz=28.0, u0=0.2, v0=0.1, noise=0.0, conf=0.9):
    """A hand moving at (vu, vv) shoulder widths a second through the stroke's peak (at PEAK)."""
    out, k = [], 0
    while start + k / hz <= stop:
        t = start + k / hz
        wob = noise * math.sin(7.3 * k)
        out.append(PaddlePose(t_scene_ns=PEAK + round(t * S), u=u0 + vu * t + wob, v=v0 + vv * t - wob, conf=conf, hand="right"))
        k += 1
    return out


# --- the path of the hand ------------------------------------------------------------------------------------------------
def test_the_hands_velocity_over_the_stroke_is_the_slope_of_its_readings():
    path = strokepath.hand_path(track(2.0, -1.0), PEAK)
    assert path.vu == pytest.approx(2.0, rel=0.02) and path.vv == pytest.approx(-1.0, rel=0.02) and path.n >= 8


def test_a_little_jitter_in_the_readings_barely_moves_the_slope():
    path = strokepath.hand_path(track(1.5, 2.0, noise=0.02), PEAK)
    assert path.vu == pytest.approx(1.5, abs=0.35) and path.vv == pytest.approx(2.0, abs=0.35)


def test_only_readings_around_the_peak_that_the_camera_was_sure_of_count():
    before = track(-3.0, 3.0, start=-1.5, stop=-0.5)                       # an earlier movement
    after = track(-3.0, 3.0, start=0.5, stop=1.5)
    unsure = track(4.0, 4.0, conf=0.2)
    path = strokepath.hand_path(before + after + unsure + track(1.0, 1.0), PEAK)
    assert path.vu == pytest.approx(1.0, rel=0.05) and path.vv == pytest.approx(1.0, rel=0.05)


def test_too_few_readings_or_too_short_a_stretch_give_no_path_instead_of_a_wild_one():
    assert strokepath.hand_path(track(1.0, 1.0)[:3], PEAK) is None
    assert strokepath.hand_path(track(1.0, 1.0, start=-0.05, stop=0.02, hz=60.0), PEAK) is None
    assert strokepath.hand_path([], PEAK) is None


def test_the_window_can_end_at_the_moment_the_paddle_met_the_ball():
    moving_then_still = track(3.0, 0.0, start=-0.30, stop=0.0) + track(0.0, 0.0, start=0.001, stop=0.5, u0=0.2 + 3.0 * 0.0)
    path = strokepath.hand_path(moving_then_still, PEAK, window_s=(-0.25, 0.0))
    assert path.vu == pytest.approx(3.0, rel=0.1)


# --- how hard ------------------------------------------------------------------------------------------------------------------
M = strokepath.ShotModel()


def test_a_hand_that_barely_moves_is_a_block_and_a_fast_one_a_full_hit_in_between_it_is_proportional():
    assert strokepath.hand_strength(strokepath.HandPath(0.2, 0.1, 9), M) == 0.0
    assert strokepath.hand_strength(strokepath.HandPath(0.0, M.hand_hi_sw_s, 9), M) == 1.0
    assert strokepath.hand_strength(strokepath.HandPath(9.0, 9.0, 9), M) == 1.0
    mid = strokepath.hand_strength(strokepath.HandPath(3.0 * 0.6, 3.0 * 0.8, 9), M)                     # 3.0 shoulder widths a second
    assert mid == pytest.approx((3.0 - M.hand_lo_sw_s) / (M.hand_hi_sw_s - M.hand_lo_sw_s))
    assert strokepath.hand_strength(None, M) == 0.0                                                     # a hand nobody saw moving


# --- the shape of the return -----------------------------------------------------------------------------------------------------
def shape(vu, vv, top=0.0, side=0.0, model=M):
    return strokepath.shape_return(strokepath.HandPath(vu, vv, 10), top, side, model)


def test_a_hand_at_rest_or_unseen_changes_nothing():
    flat = strokepath.shape_return(None, 0.0, 0.0, M)
    assert (flat.aim_shift, flat.loft_m, flat.topspin, flat.sidespin) == (0.0, 0.0, 0.0, 0.0)
    assert shape(0.0, 0.0).loft_m == 0.0


def test_a_stroke_that_lifts_gives_a_higher_arc_and_one_that_chops_does_not_flatten_below_the_net_clearing_arc():
    up, down = shape(0.0, 2.5), shape(0.0, -2.5)
    assert up.loft_m == pytest.approx(M.k_loft_m) and down.loft_m == 0.0
    assert up.topspin == 0.0 and down.topspin == 0.0                       # the path no longer makes spin: the flick does


def test_a_stroke_that_goes_sideways_places_the_ball_that_way_mirror_image_and_makes_no_spin():
    right, left = shape(2.5, 0.0), shape(-2.5, 0.0)
    assert right.aim_shift == pytest.approx(M.k_aim) and left.aim_shift == pytest.approx(-M.k_aim)
    assert right.sidespin == 0.0 and left.sidespin == 0.0


def test_the_flicks_spin_passes_through_clipped_to_minus_one_to_one():
    s = shape(0.0, 0.0, top=0.4, side=-0.3)
    assert (s.topspin, s.sidespin) == (0.4, -0.3)
    wild = shape(0.0, 0.0, top=7.0, side=-7.0)
    assert (wild.topspin, wild.sidespin) == (1.0, -1.0)


def test_a_hand_speed_counts_only_up_to_a_cap():
    wild = shape(40.0, 40.0)
    assert wild.aim_shift == pytest.approx(M.k_aim * M.speed_cap) and wild.loft_m == pytest.approx(M.k_loft_m * M.speed_cap)


def test_the_amplitude_of_the_spin_is_its_size():
    s = shape(0.0, 0.0, top=0.6, side=0.3)
    assert s.amplitude == pytest.approx(math.hypot(0.6, 0.3))


def test_the_model_can_be_retuned_by_field():
    soft = strokepath.ShotModel(k_aim=0.0, k_loft_m=0.5)
    assert shape(2.5, 2.5, model=soft).aim_shift == 0.0 and shape(2.5, 2.5, model=soft).loft_m == pytest.approx(0.5)
