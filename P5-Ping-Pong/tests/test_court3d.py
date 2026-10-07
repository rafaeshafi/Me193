"""The table in perspective: a camera behind the player's end, above the table, looking down it (like an arcade table
tennis game).  World metres as in physics.py: x across (right positive), y up from the table top, z from the player's
edge (0) to the computer's (2.74)."""

import pytest

from pingpong import court3d, physics

W, H = 1280, 720


@pytest.fixture
def cam():
    return court3d.Camera.for_frame(W, H)


def test_the_far_end_is_higher_and_narrower_than_the_near_end(cam):
    nx, ny, ns = cam.project(0.5, 0.0, 0.0)
    fx, fy, fs = cam.project(0.5, 0.0, physics.TABLE_LEN_M)
    assert fy < ny and fs < ns
    assert abs(fx - W / 2) < abs(nx - W / 2)


def test_depth_scale_roughly_doubles_from_the_far_end_to_the_near_end(cam):
    # enough growth that a ball visibly comes at you, without the far end vanishing
    ratio = cam.project(0, 0, 0.0)[2] / cam.project(0, 0, physics.TABLE_LEN_M)[2]
    assert 1.7 < ratio < 2.6


def test_the_table_fits_the_screen_below_the_score_and_above_the_footer(cam):
    nl = cam.project(-physics.HALF_WIDTH_M, 0, 0)
    nr = cam.project(physics.HALF_WIDTH_M, 0, 0)
    far = cam.project(0, 0, physics.TABLE_LEN_M)
    assert 0.45 < (nr[0] - nl[0]) / W < 0.60                     # about half the screen across at your end
    assert 0.24 * H < far[1] < 0.34 * H                          # the far edge sits under the score
    assert 0.68 * H < nl[1] < 0.84 * H                           # your edge leaves room for the legs and the footer


def test_lateral_position_maps_left_and_right_symmetrically(cam):
    lx, ly, _ = cam.project(-0.4, 0.0, 0.8)
    rx, ry, _ = cam.project(0.4, 0.0, 0.8)
    assert (lx + rx) / 2 == pytest.approx(W / 2, abs=1e-6) and ly == pytest.approx(ry)
    assert rx > lx


def test_height_lifts_a_point_up_the_screen_and_a_shadow_stays_on_the_table(cam):
    _, y0, _ = cam.project(0.0, 0.0, 0.8)
    _, y1, _ = cam.project(0.0, 0.3, 0.8)
    assert y1 < y0 - 40                                           # 0.3 m up is plainly higher: the shadow shows the height


def test_a_ball_over_the_sweet_spot_is_drawn_about_as_big_as_a_paddle_is_wide_is_not_needed_just_bigger_than_the_far_one(cam):
    near = cam.project(0.0, physics.STRIKE_Y_M, physics.HIT_Z_M)[2]
    far = cam.project(0.0, physics.STRIKE_Y_M, physics.CPU_Z_M)[2]
    assert near > far * 1.7


def test_a_circle_on_the_table_is_wider_than_it_is_tall_on_screen(cam):
    pts = cam.ground_circle(0.0, 0.8, 0.3)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert len(pts) >= 20
    assert (max(xs) - min(xs)) > 1.5 * (max(ys) - min(ys))
    assert (max(xs) + min(xs)) / 2 == pytest.approx(W / 2, abs=1)


def test_the_camera_follows_the_frame_size():
    small, big = court3d.Camera.for_frame(640, 360), court3d.Camera.for_frame(1280, 720)
    sx, sy, ss = small.project(0.2, 0.1, 1.0)
    bx, by, bs = big.project(0.2, 0.1, 1.0)
    assert (bx, by, bs) == pytest.approx((2 * sx, 2 * sy, 2 * ss))


def test_everything_the_game_draws_is_in_front_of_the_camera(cam):
    # the lowest, nearest thing: a ball that flew past you and dropped
    px, py, scale = cam.project(0.0, physics.FLOOR_Y_M, -1.0)
    assert scale > 0 and 0 < py < 2 * H


def test_an_ellipse_on_the_table_is_as_long_along_the_table_as_asked_and_symmetric_across_it(cam):
    pts = cam.ground_ellipse(0.0, 1.0, 0.3, 0.8)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    near_y, far_y = cam.project(0.0, 0.0, 0.2)[1], cam.project(0.0, 0.0, 1.8)[1]
    assert max(ys) == pytest.approx(near_y, abs=1.0) and min(ys) == pytest.approx(far_y, abs=1.0)
    assert (max(xs) + min(xs)) / 2 == pytest.approx(W / 2, abs=1.0)
