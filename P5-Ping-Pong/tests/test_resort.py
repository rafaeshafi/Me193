"""The resort: a sunny seaside terrace on a pier, with an island behind it.  One world drawn through any camera, so the game's
own view and the intro's flight over the island show the same place."""

import dataclasses
import math

import numpy as np
import pytest

from pingpong import court3d, resort, resort_sky

W, H = 640, 360


def render(cam, t=0.0, **kw):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    resort.draw_backdrop(frame, cam, t, **kw)
    return frame


def px(cam, x, y, z):
    sx, sy, _ = cam.project(x, y, z)
    return round(sx), round(sy)


def colour_near(frame, cam, point, targets, tol=36, reach=2):
    """Is any pixel within `reach` of the projected point within `tol` (summed over channels) of one of the target colours?"""
    x, y = px(cam, *point)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if 0 <= y + dy < H and 0 <= x + dx < W:
                got = frame[y + dy, x + dx].astype(int)
                if any(np.abs(got - np.array(t)).sum() <= tol for t in targets):
                    return True
    return False


GAME = court3d.Camera.for_frame(W, H)                        # the game's own view, at half size
LOOKING_OUT = dataclasses.replace(GAME, pos=(0.0, 3.0, -1.8), pitch_deg=6.0, yaw_deg=90.0)    # across the side rail, level enough to see the horizon
HIGH = dataclasses.replace(GAME, pos=(-30.0, 60.0, -190.0), pitch_deg=20.0, yaw_deg=10.0)    # the start of the fly-in


# --- the terrace as the game sees it -----------------------------------------------------------------------------------------------
def test_the_deck_is_planks_under_and_beside_the_table():
    f = render(GAME)
    wood = (resort.DECK, resort.DECK_ALT, resort.DECK_SEAM)
    for point in ((2.2, resort.DECK_Y, 1.0), (-1.8, resort.DECK_Y, 3.0), (1.4, resort.DECK_Y, 0.8), (0.0, resort.DECK_Y, 4.5)):
        assert colour_near(f, GAME, point, wood, tol=40), point


def test_the_far_railing_is_white_posts_and_rails_with_the_sea_above_it():
    f = render(GAME)
    assert colour_near(f, GAME, (0.45, resort.RAIL_TOP_Y, resort.RAIL_Z), (resort.RAIL,), tol=30)
    top = f[2:6, W // 4:3 * W // 4].reshape(-1, 3).mean(axis=0)
    assert top[0] > top[2] + 15                                       # BGR: blue over red, the sea behind the railing


def test_nothing_is_left_undrawn_every_pixel_is_painted():
    f = render(GAME)
    assert (f.sum(axis=2) > 0).all()


def test_bunting_hangs_along_the_far_railing_in_more_than_one_colour():
    f = render(GAME)
    x0, y0 = px(GAME, -1.5, resort.RAIL_TOP_Y + 0.3, resort.RAIL_Z)
    x1, y1 = px(GAME, 1.5, resort.RAIL_TOP_Y + 0.1, resort.RAIL_Z)
    strip = f[max(0, y0 - 14):y1 + 14, x0:x1].reshape(-1, 3)
    flags = [c for c in resort.BUNTING if (np.abs(strip.astype(int) - np.array(c)).sum(axis=1) <= 30).sum() > 6]
    assert len(flags) >= 3


def test_the_scene_is_the_same_every_time_it_is_drawn():
    assert np.array_equal(render(GAME, 1.0), render(GAME, 1.0))


def test_a_shadow_of_the_table_darkens_the_deck_beneath_it():
    f = render(GAME)
    under = px(GAME, 0.0, resort.DECK_Y, 1.4)
    beside = px(GAME, 1.6, resort.DECK_Y, 1.4)
    assert f[under[1], under[0]].astype(int).sum() < f[beside[1], beside[0]].astype(int).sum() - 25


# --- the sky, the sun, the sea ---------------------------------------------------------------------------------------------------
def test_the_sky_is_bluer_overhead_and_paler_at_the_horizon():
    f = render(LOOKING_OUT)
    horizon = round(LOOKING_OUT.cy - LOOKING_OUT.focal * math.tan(math.radians(LOOKING_OUT.pitch_deg)))
    assert 20 < horizon < H - 40
    high, low = f[3, 20].astype(int), f[horizon - 4, 20].astype(int)
    assert low[2] > high[2] + 25                                      # more red (paler) towards the horizon


def test_the_sea_is_turquoise_below_the_horizon_and_hazier_the_further_away():
    f = render(LOOKING_OUT)
    near, far = f[H - 4, W - 8].astype(int), f[round(LOOKING_OUT.cy - LOOKING_OUT.focal * math.tan(math.radians(6))) + 6, W - 8].astype(int)
    assert near[0] > near[2] + 40 and near[1] > near[2] + 30          # BGR: blue and green over red
    assert far[2] > near[2] + 25                                      # the far sea has washed out towards the haze


def test_the_sun_is_a_bright_disc_where_the_camera_looks_at_it_and_not_in_the_picture_when_it_looks_away():
    seen = resort_sky.sun_px(LOOKING_OUT)
    assert seen is None or not (0 < seen[0] < W and 0 < seen[1] < H)
    facing = dataclasses.replace(LOOKING_OUT, yaw_deg=math.degrees(math.atan2(resort_sky.SUN_DIR[0], resort_sky.SUN_DIR[2])),
                                 pitch_deg=-resort_sky.sun_elevation_deg() + 4.0)
    sx, sy = resort_sky.sun_px(facing)
    assert 0 < sx < W and 0 < sy < H
    f = render(facing)
    assert f[round(sy), round(sx)].astype(int).min() > 215            # white-hot in the middle
    assert resort_sky.sun_px(dataclasses.replace(facing, yaw_deg=facing.yaw_deg + 180.0)) is None      # behind the camera


def test_clouds_drift_across_the_sky_over_time():
    sky_cam = dataclasses.replace(GAME, pos=(0.0, 90.0, -120.0), pitch_deg=8.0)
    a, b = render(sky_cam, 0.0), render(sky_cam, 20.0)
    assert np.abs(a.astype(int) - b.astype(int)).sum() > 20_000


def test_clouds_are_white_and_there_are_some_to_see_from_up_high():
    sky_cam = dataclasses.replace(GAME, pos=(0.0, 90.0, -120.0), pitch_deg=8.0)
    f = render(sky_cam, 0.0)
    assert (f.astype(int).min(axis=2) > 235).sum() > 400


def test_the_island_is_there_from_the_start_of_the_fly_in_and_is_not_when_looking_the_other_way():
    f = render(HIGH)
    sand, grass = np.array(resort.SAND), np.array(resort.GRASS)
    near_sand = (np.abs(f.astype(int) - sand).sum(axis=2) <= 40).sum()
    near_grass = (np.abs(f.astype(int) - grass).sum(axis=2) <= 60).sum()
    assert near_sand > 150 and near_grass > 400
    away = render(dataclasses.replace(HIGH, yaw_deg=HIGH.yaw_deg + 180.0))
    assert (np.abs(away.astype(int) - grass).sum(axis=2) <= 60).sum() < near_grass / 10


def test_the_terrace_can_be_seen_from_up_high_and_the_island_can_be_left_out():
    f = render(HIGH)
    quiet = render(HIGH, island=False)
    assert np.abs(f.astype(int) - quiet.astype(int)).sum() > 50_000
    deck = np.array(resort.DECK)
    assert (np.abs(f.astype(int) - deck).sum(axis=2) <= 40).sum() > 20


# --- the camera can be anywhere ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pos, pitch, yaw", [((0.0, 1.4, -1.8), 28.0, 0.0), ((-30.0, 60.0, -190.0), 20.0, 10.0),
                                             ((5.0, 3.0, -20.0), 14.0, -25.0), ((0.0, 0.5, -3.0), 5.0, 70.0),
                                             ((0.0, 200.0, 0.0), 80.0, 0.0), ((0.0, 3.0, 4.0), 10.0, 180.0)])
def test_the_world_draws_from_anywhere_without_error_and_leaves_no_holes(pos, pitch, yaw):
    f = render(dataclasses.replace(GAME, pos=pos, pitch_deg=pitch, yaw_deg=yaw), 3.0)
    assert f.shape == (H, W, 3) and (f.sum(axis=2) > 0).all()


def test_drawing_into_a_smaller_picture_looks_the_same_only_smaller_the_cameras_scale_does_it():
    small = np.zeros((180, 320, 3), dtype=np.uint8)
    resort.draw_backdrop(small, court3d.Camera.for_frame(320, 180), 0.0)
    assert small.shape == (180, 320, 3) and (small.sum(axis=2) > 0).all()
