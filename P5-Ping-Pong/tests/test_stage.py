"""Where your paddle stands in the scene, and how a stroke moves it (the Konami-style depth of the paddle)."""

import pytest

from pingpong import levels, physics, stage
from pingpong.paddle import ReachBox

S = 1_000_000_000
BOX = ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5)


def test_your_hand_height_is_how_far_up_the_table_the_paddle_stands():
    assert stage.rest_z(BOX, -0.5) == pytest.approx(stage.Z_REST_MIN)
    assert stage.rest_z(BOX, 0.5) == pytest.approx(stage.Z_REST_MAX)
    assert stage.rest_z(BOX, 0.0) == pytest.approx(physics.HIT_Z_M)           # a hand in the middle stands where the sweet spot is
    assert stage.rest_z(BOX, -3.0) == stage.Z_REST_MIN and stage.rest_z(BOX, 9.0) == stage.Z_REST_MAX


def test_your_hand_across_the_box_is_where_the_paddle_stands_across_the_table():
    left, mid, right = (stage.rest_position(BOX, u, 0.0) for u in (-1.0, 0.0, 1.0))
    assert mid[0] == pytest.approx(0.0) and left[0] == pytest.approx(-right[0]) and right[0] > 0.5
    assert mid[1] == pytest.approx(stage.PADDLE_Y_M) and mid[2] == pytest.approx(physics.HIT_Z_M)
    overhang = stage.rest_position(BOX, 3.0, 0.0)[0]
    assert right[0] < overhang <= right[0] + 0.13                               # it may hang a little over the edge, no more


# --- a stroke ---------------------------------------------------------------------------------------------------------------
REST = (0.0, 0.16, 0.3)
BALL = (0.2, 0.26, 0.9)


def test_a_stroke_that_meets_the_ball_reaches_it_on_time_holds_and_comes_back():
    st = stage.hit_stroke(t0_ns=S, contact_ns=S + 100_000_000, ball=BALL)
    assert st.pose(REST, S) == REST and st.pose(REST, S - 10_000_000) == REST
    assert st.pose(REST, S + 100_000_000) == pytest.approx(BALL, abs=1e-9)
    mid = st.pose(REST, S + 50_000_000)
    assert REST[2] < mid[2] < BALL[2] and 0.0 < mid[0] < BALL[0]
    assert st.pose(REST, S + 130_000_000) == pytest.approx(BALL)               # held a moment on the ball
    assert st.pose(REST, st.end_ns) == pytest.approx(REST)
    assert not st.done(st.end_ns - 1) and st.done(st.end_ns)


def test_the_reach_is_smooth_the_paddle_never_goes_backwards_on_the_way_out():
    st = stage.hit_stroke(t0_ns=0, contact_ns=100_000_000, ball=BALL)
    zs = [st.pose(REST, round(k * 1e6))[2] for k in range(0, 101, 5)]
    assert all(b >= a - 1e-12 for a, b in zip(zs, zs[1:]))


def test_a_contact_that_is_already_due_snaps_to_the_ball_quickly_instead_of_jumping():
    st = stage.hit_stroke(t0_ns=S, contact_ns=S - 50_000_000, ball=BALL)
    assert st.contact_ns > st.t0_ns
    assert st.pose(REST, st.t0_ns) == REST and st.pose(REST, st.contact_ns) == pytest.approx(BALL)


def test_a_swing_that_hits_nothing_still_reaches_forward_and_comes_back():
    st = stage.miss_stroke(REST, t0_ns=S)
    top = st.pose(REST, st.contact_ns)
    assert top[2] > REST[2] + 0.25 and top[0] == pytest.approx(REST[0])
    assert st.pose(REST, st.end_ns) == pytest.approx(REST)


def test_the_stroke_returns_to_where_the_hand_is_now_not_where_it_was():
    st = stage.hit_stroke(t0_ns=0, contact_ns=100_000_000, ball=BALL)
    moved = (0.4, 0.16, 0.2)
    assert st.pose(moved, st.end_ns) == pytest.approx(moved)


def test_a_ball_that_is_hit_high_is_met_at_a_sensible_height():
    st = stage.hit_stroke(t0_ns=0, contact_ns=100_000_000, ball=(0.0, 0.9, 0.5))
    assert st.pose(REST, 100_000_000)[1] <= stage.PADDLE_MAX_Y_M


# --- the zone in which the ball can be hit --------------------------------------------------------------------------------
def test_the_zone_is_where_the_ball_is_early_before_the_sweet_moment_and_late_after_it():
    level = levels.LEVELS[2]
    leg = physics.plan_leg(0, level.v_tier, 0.0, (0.5, 0.5))
    z_far, z_near = stage.zone(leg, leg.arrival_ns, level)
    assert z_far == pytest.approx(leg.position(leg.arrival_ns - round(level.early_s * S))[2])
    assert z_near == pytest.approx(leg.position(leg.arrival_ns + round(level.late_s * S))[2])
    assert z_far > physics.HIT_Z_M > z_near


def test_a_paddle_further_up_the_table_has_its_zone_further_up_too():
    level = levels.LEVELS[1]
    leg = physics.plan_leg(0, level.v_tier, 0.0, (0.5, 0.5))
    near = stage.zone(leg, leg.time_at_z(0.1), level)
    far = stage.zone(leg, leg.time_at_z(0.5), level)
    assert far[0] > near[0] and far[1] > near[1]
