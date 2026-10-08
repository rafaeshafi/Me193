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


# --- a camera that can turn and fly (the intro) ------------------------------------------------------------------------------
def test_a_camera_with_no_yaw_projects_exactly_as_before(cam):
    turned = court3d.Camera(cx=cam.cx, cy=cam.cy, focal=cam.focal, pos=cam.pos, pitch_deg=cam.pitch_deg, yaw_deg=0.0)
    assert turned.project(0.3, 0.2, 1.1) == cam.project(0.3, 0.2, 1.1)


def test_turning_the_camera_to_the_right_moves_what_is_ahead_to_the_left(cam):
    import dataclasses

    ahead = cam.project(0.0, 0.0, 5.0)[0]
    right = dataclasses.replace(cam, yaw_deg=20.0).project(0.0, 0.0, 5.0)[0]
    left = dataclasses.replace(cam, yaw_deg=-20.0).project(0.0, 0.0, 5.0)[0]
    assert right < ahead < left


def test_the_point_straight_along_the_view_direction_is_in_the_middle_of_the_picture():
    import dataclasses, math

    cam = dataclasses.replace(court3d.Camera.for_frame(W, H), pos=(1.0, 3.0, 2.0), pitch_deg=15.0, yaw_deg=40.0)
    yaw, pitch = math.radians(40.0), math.radians(15.0)
    ahead = (1.0 + 10 * math.cos(pitch) * math.sin(yaw), 3.0 - 10 * math.sin(pitch), 2.0 + 10 * math.cos(pitch) * math.cos(yaw))
    px, py, _ = cam.project(*ahead)
    assert px == pytest.approx(cam.cx, abs=0.5) and py == pytest.approx(cam.cy, abs=0.5)


def test_depth_is_how_far_in_front_of_the_camera_a_point_is(cam):
    assert cam.depth_of(0.0, 1.4, 10.0) > cam.depth_of(0.0, 1.4, 5.0) > 0
    assert cam.depth_of(0.0, 1.4, -10.0) < 0                         # behind the camera


def test_a_polygon_wholly_behind_the_camera_has_no_picture(cam):
    assert len(cam.project_poly([(-1, 0, -10), (1, 0, -10), (1, 0, -12), (-1, 0, -12)])) == 0


def test_a_polygon_wholly_in_front_projects_to_the_same_points_as_project(cam):
    quad = [(-0.5, 0, 1.0), (0.5, 0, 1.0), (0.5, 0, 2.0), (-0.5, 0, 2.0)]
    pts = cam.project_poly(quad)
    assert len(pts) == 4
    for (px, py), point in zip(pts, quad):
        assert (px, py) == pytest.approx(cam.project(*point)[:2])


def test_a_floor_running_out_behind_the_camera_is_cut_at_the_near_plane_not_thrown_across_the_picture(cam):
    floor = [(-50, -0.76, -60), (50, -0.76, -60), (50, -0.76, 60), (-50, -0.76, 60)]
    pts = cam.project_poly(floor)
    assert len(pts) >= 4
    assert all(abs(px) < 5e5 and abs(py) < 5e5 for px, py in pts)       # nothing flung off to infinity
    ys = [py for _, py in pts]
    assert max(ys) > H                                                  # the floor fills the bottom of the picture
    assert min(ys) < 0.4 * H                                            # and runs back up towards the horizon


def test_a_line_is_cut_at_the_near_plane_and_one_behind_the_camera_has_no_picture(cam):
    assert cam.project_segment((0, 0, -10), (1, 0, -12)) is None
    cut = cam.project_segment((0, -0.76, -5), (0, -0.76, 8))
    assert cut is not None and all(abs(v) < 5e5 for point in cut for v in point)


def test_many_points_at_once_give_the_same_view_as_one_at_a_time():
    import dataclasses

    import numpy as np

    cam = dataclasses.replace(court3d.Camera.for_frame(W, H), pos=(2.0, 5.0, -7.0), pitch_deg=12.0, yaw_deg=-33.0)
    pts = [(0.0, 0.0, 1.0), (3.5, -2.0, 20.0), (-4.0, 1.0, -30.0), (10.0, 30.0, 100.0)]
    right, up, depth = cam.view_many(*(np.array(c, dtype=float) for c in zip(*pts)))
    for k, point in enumerate(pts):
        assert (right[k], up[k], depth[k]) == pytest.approx(cam.view(*point))
