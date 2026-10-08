"""A hand that moves into the ball hits it: where the paddle meets the ball, from the hand's readings and the ball's flight."""

import pytest

from pingpong import contact, physics, stage
from pingpong.paddle import ReachBox

S = 1_000_000_000
BOX = ReachBox(-1.0, 1.0, -0.5, 0.5)
LEG = physics.plan_leg(0, 5.0, 0.0, (0.5, 0.5))               # the ball arrives over x = 0 (the box's middle) at z = 0.30
R = 0.7


def reading(t_s, u=0.0, v=0.0):
    return (round(t_s * S), u, v)


def meet(u=0.0, v=0.0, before=0.10, after=0.10, v_after=None):
    t_arrive = LEG.time_at_z(stage.rest_z(BOX, v)) / S
    return contact.crossing(reading(t_arrive - before, u, v), reading(t_arrive + after, u, v if v_after is None else v_after),
                            LEG, BOX, R)


def test_a_hand_in_the_balls_way_when_it_arrives_hits_it_at_the_moment_their_depths_meet():
    c = meet()
    expected = LEG.time_at_z(stage.rest_z(BOX, 0.0))
    assert c.hit and c.d_sw == pytest.approx(0.0, abs=0.02) and abs(c.t_ns - expected) < 3_000_000
    assert c.ball[2] == pytest.approx(stage.rest_z(BOX, 0.0), abs=0.01)


def test_a_hand_too_far_across_lets_the_ball_go_by_and_says_how_far():
    c = meet(u=0.9)
    assert c is not None and not c.hit and c.d_sw == pytest.approx(0.9, abs=0.05)


def test_the_edge_of_the_radius_is_the_edge_of_the_hit():
    assert meet(u=R - 0.05).hit and not meet(u=R + 0.05).hit


def test_a_raised_hand_meets_the_ball_sooner_because_the_paddle_stands_further_up_the_table():
    low, high = meet(v=-0.4), meet(v=0.4)
    assert high.t_ns < low.t_ns - 50_000_000
    assert high.ball[2] > low.ball[2]


def test_a_hand_that_moves_up_the_table_into_the_ball_catches_it_earlier_than_one_that_stays():
    t_still = LEG.time_at_z(stage.rest_z(BOX, -0.3)) / S
    still = contact.crossing(reading(t_still - 0.15, 0.0, -0.3), reading(t_still + 0.15, 0.0, -0.3), LEG, BOX, R)
    moving = contact.crossing(reading(t_still - 0.15, 0.0, -0.3), reading(t_still + 0.15, 0.0, 0.5), LEG, BOX, R)
    assert moving.hit and moving.t_ns < still.t_ns - 20_000_000


def test_readings_that_are_both_before_the_ball_or_both_after_it_are_no_contact():
    assert contact.crossing(reading(0.05), reading(0.10), LEG, BOX, R) is None                      # the ball is still far up the table
    assert contact.crossing(reading(1.5), reading(1.55), LEG, BOX, R) is None                       # it has long gone by


def test_the_hand_between_the_readings_is_interpolated_to_the_moment_of_contact():
    t_arrive = LEG.time_at_z(stage.rest_z(BOX, 0.0)) / S
    # sweeping across: at the moment of contact (the middle of the interval) the hand is at u = 0.4 of the way
    c = contact.crossing(reading(t_arrive - 0.1, -0.6, 0.0), reading(t_arrive + 0.1, 0.6, 0.0), LEG, BOX, R)
    assert c.d_sw == pytest.approx(0.0, abs=0.05) and c.hit


def test_the_contact_says_where_the_hand_was_when_it_met_the_ball():
    t_arrive = LEG.time_at_z(stage.rest_z(BOX, 0.0)) / S
    c = contact.crossing(reading(t_arrive - 0.1, -0.2, 0.0), reading(t_arrive + 0.1, 0.2, 0.0), LEG, BOX, R)
    assert c.u == pytest.approx(0.0, abs=0.03) and c.v == pytest.approx(0.0)

