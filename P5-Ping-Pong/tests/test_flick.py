"""The flick of the wrist: how the hub turns around the moment the paddle meets the ball decides the spin.

The hub's axes are read from the tilt calibration: "up" is what its accelerometer reads upright, "forward" the doorknob axis
(turning right is clockwise as the player sees it) made horizontal, "right" completes the frame.  Tipping the hub's front down is
topspin, up is backspin; turning or rolling it to the right is sidespin right.  A stroke always turns the hub somewhat, so what
counts is the turn beyond what this player's strokes usually do (the first hits of a session decide that)."""

import pytest

from pingpong import flick, strokepath

S = 1_000_000_000
M = strokepath.ShotModel()
# a hub held like a paddle: its z up, its x forward (the doorknob axis), so its -y is to the right
FRAME = flick.wrist_frame((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def test_the_frame_is_up_forward_right_and_orthonormal():
    assert FRAME.up == pytest.approx((0.0, 0.0, 1.0)) and FRAME.fwd == pytest.approx((1.0, 0.0, 0.0))
    assert FRAME.right == pytest.approx((0.0, -1.0, 0.0))                  # forward x up: to the player's right
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))                    # noqa: E731
    assert dot(FRAME.up, FRAME.fwd) == pytest.approx(0.0) and dot(FRAME.right, FRAME.fwd) == pytest.approx(0.0)


def test_a_tilted_upright_pose_still_gives_a_horizontal_forward():
    frame = flick.wrist_frame((0.0, 0.5, 0.866), (1.0, 0.0, 0.0))
    assert sum(a * b for a, b in zip(frame.fwd, frame.up)) == pytest.approx(0.0, abs=1e-9)


def test_without_a_tilt_calibration_or_with_the_doorknob_along_up_there_is_no_frame():
    assert flick.wrist_frame(None, None) is None
    assert flick.wrist_frame((0.0, 0.0, 1.0), None) is None
    assert flick.wrist_frame((0.0, 0.0, 1.0), (0.0, 0.01, 1.0)) is None


# --- the turn around the contact ----------------------------------------------------------------------------------------------
def samples(rate, t0=0, hz=64.0, seconds=0.4):
    """[(arrival ns, gyro dps)] with a constant rate vector."""
    return [(t0 + round(i * S / hz), tuple(rate)) for i in range(int(seconds * hz))]


def test_the_turn_is_the_mean_gyro_vector_over_the_window_and_none_without_enough_samples():
    window = samples((10.0, -20.0, 30.0), seconds=0.4)
    assert flick.mean_rate(window, 0, round(0.4 * S)) == pytest.approx((10.0, -20.0, 30.0))
    assert flick.mean_rate(window, 0, round(0.02 * S)) is None                # one sample is no turn
    assert flick.mean_rate([], 0, S) is None


def test_the_window_is_around_the_contact_and_shifted_by_the_hubs_delay():
    lo, hi = flick.window_ns(1_000_000_000, round(0.04 * S), M)
    assert lo == 1_000_000_000 + round(0.04 * S) - round(M.flick_before_s * S)
    assert hi == 1_000_000_000 + round(0.04 * S) + round(M.flick_after_s * S)


# --- what is usual -------------------------------------------------------------------------------------------------------------
def test_the_usual_turn_is_learnt_from_the_first_hits_and_then_fixed():
    base = flick.FlickBaseline(warmup=3)
    assert base.deviation((100.0, 0.0, 0.0)) == (0.0, 0.0, 0.0) and base.deviation((120.0, 10.0, 0.0)) == (0.0, 0.0, 0.0)
    assert base.deviation((110.0, -10.0, 5.0)) == (0.0, 0.0, 0.0)
    assert base.deviation((110.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)                       # the median (110, 0, 0)
    far = base.deviation((410.0, 0.0, 0.0))
    assert far[0] == pytest.approx(300.0) and base.deviation((410.0, 0.0, 0.0))[0] == pytest.approx(300.0)    # it does not drift with it
    assert base.deviation(None) == (0.0, 0.0, 0.0)


# --- from the turn to the spin --------------------------------------------------------------------------------------------------
def spin(dev, model=M, frame=FRAME):
    return flick.spin_from_flick(dev, frame, model)


def test_tipping_the_hubs_front_down_is_topspin_and_up_is_backspin():
    # positive rotation about FRAME.right (here -y) tips the forward direction up: backspin; negative tips it down: topspin
    top_down, side = spin((0.0, 350.0, 0.0))                               # +y is negative about right: the front goes down
    top_up, _ = spin((0.0, -350.0, 0.0))
    assert top_down == pytest.approx(1.0 * M.k_flick_top, abs=0.01) and top_up == pytest.approx(-1.0, abs=0.01) and side == 0.0


def test_turning_or_rolling_to_the_right_is_sidespin_right_and_to_the_left_sidespin_left():
    _, turn_right = spin((0.0, 0.0, -350.0))                               # clockwise seen from above is negative about up
    _, roll_right = spin((350.0, 0.0, 0.0))                                # clockwise as the player sees is positive about forward
    _, turn_left = spin((0.0, 0.0, 350.0))
    assert turn_right == pytest.approx(1.0) and roll_right == pytest.approx(1.0) and turn_left == pytest.approx(-1.0)


def test_a_turn_inside_the_dead_zone_is_no_spin_and_the_amount_grows_from_its_edge():
    assert spin((0.0, 0.0, -0.9 * M.flick_min_dps)) == (0.0, 0.0)
    _, small = spin((0.0, 0.0, -(M.flick_min_dps + 50.0)))
    _, big = spin((0.0, 0.0, -(M.flick_min_dps + 200.0)))
    assert 0.0 < small < big < 1.0 + 1e-9


def test_spin_never_leaves_minus_one_to_one_and_a_negative_gain_flips_the_direction():
    assert spin((0.0, 9000.0, -9000.0)) == (1.0, 1.0)
    flipped = strokepath.ShotModel(k_flick_top=-1.0, k_flick_side=-1.0)
    assert spin((0.0, 350.0, -350.0), flipped) == (-1.0, -1.0)


def test_no_frame_means_no_spin():
    assert spin((0.0, 500.0, -500.0), frame=None) == (0.0, 0.0)


def test_reading_a_flick_end_to_end_is_flat_for_the_usual_turn_and_spinning_for_a_bigger_one():
    base = flick.FlickBaseline(warmup=2)
    usual = samples((50.0, 0.0, 0.0), seconds=0.6)
    contact = round(0.3 * S)
    for _ in range(2):
        assert flick.read_flick(usual, contact, 0, M, FRAME, base) == (0.0, 0.0)
    assert flick.read_flick(usual, contact, 0, M, FRAME, base) == (0.0, 0.0)               # the usual again: no spin
    flicked = samples((50.0, 0.0, -450.0), seconds=0.6)
    top, side = flick.read_flick(flicked, contact, 0, M, FRAME, base)
    assert top == 0.0 and side == pytest.approx(1.0)
