"""How the ball leaves the paddle: the path of the hand and the twist of the hub decide aim, loft and spin.

The hand's velocity over the stroke comes from the pose track; the hub's twist is the rotation it made about the doorknob
direction (the tilt calibration's axis, made perpendicular to the stroke axis) during the stroke, compared with what this
player's strokes usually do about it."""

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


# --- the twist of the hub ----------------------------------------------------------------------------------------------------
def test_the_wrist_axis_is_the_doorknob_axis_made_perpendicular_to_the_stroke():
    u_fwd, tilt = (1.0, 0.0, 0.0), (0.6, 0.8, 0.0)
    axis = strokepath.wrist_axis(u_fwd, tilt)
    assert axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert sum(a * b for a, b in zip(axis, u_fwd)) == pytest.approx(0.0, abs=1e-9)
    assert strokepath.wrist_axis(u_fwd, (-0.6, -0.8, 0.0)) == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)    # the sign (right = +) is kept


def test_without_a_tilt_calibration_or_with_the_doorknob_along_the_stroke_there_is_no_wrist_axis():
    assert strokepath.wrist_axis((1.0, 0.0, 0.0), None) is None
    assert strokepath.wrist_axis((1.0, 0.0, 0.0), (0.999, 0.01, 0.0)) is None


def test_the_twist_is_the_rotation_about_the_wrist_axis_in_degrees():
    assert strokepath.twist_deg((75.0, -20.0, 5.0), (0.0, 1.0, 0.0)) == pytest.approx(-20.0)
    assert strokepath.twist_deg((75.0, -20.0, 5.0), None) is None
    assert strokepath.twist_deg((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == 0.0


def test_a_twist_is_measured_against_what_the_players_strokes_usually_do_once_enough_hits_are_known():
    base = strokepath.TwistBaseline(sd_deg=30.0, warmup=4)
    assert [base.z(t) for t in (-20.0, -25.0, -18.0, -22.0)] == [0.0, 0.0, 0.0, 0.0]         # still learning what is usual
    assert base.z(-21.0) == pytest.approx(0.0, abs=0.1)                                        # the usual: no sidespin
    assert base.z(9.0) == pytest.approx(1.0, abs=0.1)                                          # 30 degrees more to the right
    assert base.z(-51.0) == pytest.approx(-1.0, abs=0.1)
    assert base.z(9.0) == pytest.approx(1.0, abs=0.1)                                          # the usual does not drift with it
    assert base.z(None) == 0.0


# --- the shape of the return -----------------------------------------------------------------------------------------------------
M = strokepath.ShotModel()


def shape(vu, vv, z=0.0, model=M):
    return strokepath.shape_return(strokepath.HandPath(vu, vv, 10), z, model)


def test_a_hand_at_rest_or_unseen_changes_nothing():
    flat = strokepath.shape_return(None, 0.0, M)
    assert (flat.aim_shift, flat.loft_m, flat.topspin, flat.sidespin) == (0.0, 0.0, 0.0, 0.0)
    assert shape(0.0, 0.0).topspin == 0.0


def test_a_stroke_that_lifts_gives_loft_and_topspin_and_a_stroke_that_chops_gives_backspin_and_no_loft():
    up, down = shape(0.0, 2.5), shape(0.0, -2.5)
    assert up.loft_m == pytest.approx(M.k_loft_m) and up.topspin == pytest.approx(M.k_top) and up.sidespin == 0.0
    assert down.loft_m == 0.0 and down.topspin == pytest.approx(-M.k_top)
    assert shape(0.0, 1.0).topspin < up.topspin


def test_a_stroke_that_goes_sideways_places_the_ball_that_way_and_puts_sidespin_on_it_mirror_image():
    right, left = shape(2.5, 0.0), shape(-2.5, 0.0)
    assert right.aim_shift == pytest.approx(M.k_aim) and right.sidespin == pytest.approx(M.k_side_path)
    assert left.aim_shift == pytest.approx(-right.aim_shift) and left.sidespin == pytest.approx(-right.sidespin)
    assert right.topspin == 0.0 and right.loft_m == 0.0


def test_a_twist_of_the_hub_adds_sidespin_on_its_own_and_with_the_path():
    twisted = shape(0.0, 0.0, z=1.0)
    assert twisted.sidespin == pytest.approx(M.k_side_twist) and twisted.aim_shift == 0.0
    assert shape(2.5, 0.0, z=1.0).sidespin == pytest.approx(M.k_side_path + M.k_side_twist)
    assert strokepath.shape_return(None, 1.0, M).sidespin == pytest.approx(M.k_side_twist)          # with no path, the twist still counts


def test_a_hand_speed_counts_only_up_to_a_cap_and_spin_never_leaves_minus_one_to_one():
    wild = shape(40.0, 40.0, z=9.0)
    assert wild.aim_shift == pytest.approx(M.k_aim * M.speed_cap) and wild.topspin == pytest.approx(M.k_top * M.speed_cap)
    assert -1.0 <= wild.sidespin <= 1.0
    assert abs(shape(0.0, 0.0, z=100.0).sidespin) <= 1.0


def test_the_amplitude_of_the_spin_is_its_size():
    s = shape(2.5, 2.5)
    assert s.amplitude == pytest.approx(math.hypot(s.topspin, s.sidespin))


def test_the_model_can_be_retuned_by_field():
    soft = strokepath.ShotModel(k_top=0.1, k_aim=0.0)
    assert shape(2.5, 2.5, model=soft).topspin == pytest.approx(0.1) and shape(2.5, 2.5, model=soft).aim_shift == 0.0
