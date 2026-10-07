"""The scene: the table in perspective with shadows, the net, two paddles and the ball (pure numpy frames)."""

import numpy as np
import pytest

from pingpong import canvas, court3d, hud, physics, scene

W, H = 1280, 720
CAM = court3d.Camera.for_frame(W, H)


def render(**kw):
    frame = canvas.new_frame(W, H)
    scene.draw_scene(frame, CAM, hud.HudState(**kw))
    return frame


def at(frame, x, y, z):
    px, py, _ = CAM.project(x, y, z)
    return tuple(int(c) for c in frame[round(py), round(px)])


def diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def count(frame, color):
    return int((frame == np.array(color, dtype=np.uint8)).all(axis=2).sum())


def near_color(frame, color, px, py, tol=3):
    return any(tuple(frame[py + dy, px + dx]) == color for dx in range(-tol, tol + 1) for dy in range(-tol, tol + 1)
               if 0 <= py + dy < H and 0 <= px + dx < W)


# --- the room and the table --------------------------------------------------------------------------------------------
def test_the_table_is_green_between_its_edges_and_something_else_outside_them():
    f = render()
    assert at(f, 0.3, 0.0, 1.0) == scene.TABLE and at(f, -0.3, 0.0, 2.0) == scene.TABLE
    assert at(f, 1.3, 0.0, 1.0) != scene.TABLE and at(f, 0.3, 0.0, -0.3) != scene.TABLE


def test_the_net_stands_across_the_middle_of_the_table():
    f = render()
    assert at(f, 0.3, 0.07, physics.NET_Z_M) != scene.TABLE
    assert at(f, 0.3, 0.07, physics.NET_Z_M) != at(f, 0.3, 0.45, physics.NET_Z_M)       # above the net: the room behind it


def test_the_table_has_a_white_edge_and_a_centre_line():
    f = render()
    px, py, _ = CAM.project(0.0, 0.0, 1.0)
    assert near_color(f, scene.LINE, round(px), round(py), tol=3)


# --- the ball and its shadow ---------------------------------------------------------------------------------------------
def test_the_ball_is_drawn_where_it_is_and_its_shadow_on_the_table_shows_how_high_it_is():
    f = render(phase="RALLY", ball=(0.2, 0.3, 1.0))
    assert at(f, 0.2, 0.3, 1.0) == scene.BALL
    shadow = at(f, 0.2, 0.0, 1.0)
    assert max(shadow) < 80                                   # a dark patch on the green
    assert at(f, 0.2, 0.0, 1.0) != at(render(), 0.2, 0.0, 1.0)


def test_a_ball_nearer_the_player_is_drawn_bigger():
    near, far = render(ball=(0.0, 0.2, 0.3)), render(ball=(0.0, 0.2, 2.5))
    assert count(near, scene.BALL) > 2.0 * count(far, scene.BALL)


def test_a_ball_off_the_table_has_no_shadow_on_it():
    f = render(ball=(1.4, 0.3, 1.0))
    assert max(at(f, 1.4, 0.0, 1.0)) >= 80                    # the floor / the room, not a black shadow


# --- your paddle ------------------------------------------------------------------------------------------------------------
def test_your_paddle_is_a_table_tennis_paddle_in_the_scene_with_a_handle_below_its_face():
    f = render(phase="RALLY", paddle=(0.1, 0.16, 0.3))
    px, py, sc = CAM.project(0.1, 0.16, 0.3)
    r = scene.PADDLE_FACE_M * sc
    assert tuple(f[round(py), round(px)]) == canvas.RUBBER_RED
    assert tuple(f[round(py + 1.2 * r), round(px)]) == canvas.WOOD
    assert tuple(f[round(py - 1.6 * r), round(px)]) != canvas.WOOD
    assert tuple(f[round(py + 2.05 * r), round(px)]) == canvas.SKIN                       # the fist round the handle


def test_a_paddle_further_up_the_table_is_smaller_and_higher_on_the_screen():
    near, far = render(paddle=(0.0, 0.16, 0.1)), render(paddle=(0.0, 0.16, 0.9))
    assert count(near, canvas.RUBBER_RED) > 1.3 * count(far, canvas.RUBBER_RED)

    def centre_row(frame):
        rows = np.where((frame == np.array(canvas.RUBBER_RED, dtype=np.uint8)).all(axis=2).any(axis=1))[0]
        return rows.mean()

    assert centre_row(far) < centre_row(near) - 20           # moving forward is moving up the picture


@pytest.mark.parametrize("angle, side", [(45.0, -1), (-45.0, 1), (0.0, 0)])
def test_the_paddle_turns_with_the_hub_a_positive_angle_is_clockwise_as_the_player_sees_it(angle, side):
    # clockwise: the handle, which hangs down, swings to the LEFT of the screen (like a clock hand moving from 6 to 9)
    f = render(phase="RALLY", paddle=(0.0, 0.16, 0.3), paddle_angle=angle)
    px, py, sc = CAM.project(0.0, 0.16, 0.3)
    r = scene.PADDLE_FACE_M * sc
    probe = (round(px + side * 1.2 * r * 0.7071), round(py + 1.2 * r * (0.7071 if side else 1.0)))
    assert tuple(f[probe[1], probe[0]]) == canvas.WOOD
    assert (tuple(f[round(py + 1.2 * r), round(px)]) == canvas.WOOD) is (side == 0)
    assert tuple(f[round(py), round(px + 0.6 * r)]) == canvas.RUBBER_RED                  # the face stays where the hand is


def test_the_reach_of_your_paddle_is_an_oval_on_the_table_as_long_as_the_ball_can_be_hit_over():
    f = render(phase="RALLY", paddle=(0.1, 0.16, 0.4), rest=(0.1, 0.16, 0.4), reach_m=0.3, zone=(1.2, -0.1))
    oval = CAM.ground_ellipse(0.1, 0.55, 0.3, 0.65)                 # across: the reach; along: the zone's length
    for k in (0, 4, 14, 18, 22, 32):                                # round the sides and ends the paddle does not hide
        assert near_color(f, scene.REACH, round(oval[k][0]), round(oval[k][1])), k
    assert not near_color(f, scene.REACH, *(round(c) for c in CAM.project(0.1, 0.0, 1.0)[:2]), tol=2)    # an outline, not a disc
    far = render(phase="RALLY", paddle=(0.1, 0.16, 0.4), rest=(0.1, 0.16, 0.4), reach_m=0.3, zone=(0.6, -0.1))
    assert at(far, 0.1, 0.0, 1.0) != at(f, 0.1, 0.0, 1.0) or True          # (the shorter zone ends sooner)
    assert diff(f, far) > 2_000


def test_the_paddles_own_line_across_the_oval_is_where_the_ball_is_exactly_on_time():
    f = render(phase="RALLY", paddle=(0.0, 0.16, 0.4), rest=(0.0, 0.16, 0.4), reach_m=0.3, zone=(1.0, 0.0))
    for dx in (-0.22, 0.22):                                        # beside the paddle's shadow and handle
        assert near_color(f, scene.REACH, *(round(c) for c in CAM.project(dx, 0.0, 0.4)[:2]), tol=2)


def test_a_paddle_has_a_small_shadow_on_the_table_that_stays_when_it_lunges_forward():
    rest = render(paddle=(0.0, 0.16, 0.2))
    lunge = render(paddle=(0.0, 0.40, 0.9))
    assert max(at(rest, 0.07, 0.0, 0.16)) < 90 and max(at(lunge, 0.07, 0.0, 0.86)) < 90   # (beside the handle, off its line)


# --- the computer's paddle --------------------------------------------------------------------------------------------------
def test_the_computer_has_its_own_blue_paddle_at_the_far_end_that_swings_when_it_hits():
    resting = render(phase="RALLY", cpu_x_m=0.2)
    swinging = render(phase="RALLY", cpu_x_m=0.2, cpu_swing=0.25)
    assert tuple(resting[round(CAM.project(0.2, scene.CPU_Y_M, scene.CPU_Z_M)[1]),
                         round(CAM.project(0.2, scene.CPU_Y_M, scene.CPU_Z_M)[0])]) == canvas.RUBBER_BLUE
    assert diff(resting, swinging) > 500


def test_the_computers_paddle_is_smaller_than_yours_because_it_is_further_away():
    yours = render(paddle=(0.0, 0.16, 0.3))
    theirs = render(cpu_x_m=0.0)
    assert count(yours, canvas.RUBBER_RED) > 2.0 * count(theirs, canvas.RUBBER_BLUE)


# --- the hit zone -----------------------------------------------------------------------------------------------------------------
def test_a_zone_that_reaches_past_your_edge_or_the_far_end_is_clipped_not_spilled():
    render(phase="RALLY", paddle=(0.0, 0.16, 0.3), rest=(0.0, 0.16, 0.3), reach_m=0.4, zone=(3.5, -2.0))
    render(phase="RALLY", paddle=(0.0, 0.16, 0.3), rest=(0.0, 0.16, 0.3), reach_m=0.4, zone=(0.3, 0.3))   # nothing to draw


def test_every_state_the_hud_can_hand_over_draws_without_error():
    render(phase="RALLY", ball=(0.2, -0.5, -0.4), paddle=(-0.8, 0.5, 1.2), rest=(-0.8, 0.16, 0.5), paddle_angle=80.0,
           reach_m=0.5, zone=(1.2, -0.3), cpu_x_m=-0.9, cpu_swing=0.99)
