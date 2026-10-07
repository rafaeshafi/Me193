"""Court projection and HUD rendering (pure numpy frames, no window)."""

import numpy as np
import pytest

from pingpong import canvas, hud
from pingpong.events import GateResult

W, H = 1280, 720


def diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def state(**kw):
    return hud.HudState(**kw)


def test_far_end_is_higher_and_narrower_than_the_near_end():
    fx, fy, fs = canvas.project(0.5, 0.0, 0.0, W, H)
    nx, ny, ns = canvas.project(0.5, 1.0, 0.0, W, H)
    assert fy < ny and fs < ns
    assert abs(fx - W / 2) < abs(nx - W / 2)


def test_lateral_position_maps_left_and_right_symmetrically():
    lx, _, _ = canvas.project(-0.4, 0.6, 0.0, W, H)
    rx, _, _ = canvas.project(0.4, 0.6, 0.0, W, H)
    assert (lx + rx) / 2 == pytest.approx(W / 2, abs=1)
    assert rx > lx


def test_height_lifts_the_ball_up_the_screen():
    _, y0, _ = canvas.project(0.0, 0.5, 0.0, W, H)
    _, y1, _ = canvas.project(0.0, 0.5, 0.3, W, H)
    assert y1 < y0


def test_render_returns_a_bgr_frame_of_the_requested_size():
    frame = hud.render(state(), size=(W, H))
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8


def test_a_ball_in_flight_changes_the_picture():
    empty = hud.render(state(phase="RALLY"), size=(W, H))
    ball = hud.render(state(phase="RALLY", ball=(0.2, 0.5, 0.1)), size=(W, H))
    assert diff(empty, ball) > 5_000


def test_the_streak_number_changes_the_picture():
    a = hud.render(state(phase="RALLY", streak=3), size=(W, H))
    b = hud.render(state(phase="RALLY", streak=17), size=(W, H))
    assert diff(a, b) > 5_000


def test_the_countdown_digit_is_drawn_only_in_the_countdown_phase():
    lobby = hud.render(state(phase="LOBBY"), size=(W, H))
    count = hud.render(state(phase="COUNTDOWN", countdown=3), size=(W, H))
    assert diff(lobby, count) > 20_000


def test_the_xray_panel_lists_gate_results_when_enabled():
    gates = (GateResult("J1", True, "timing +12 ms"), GateResult("J2", False, "hand 0.9 SW from the ball"))
    off = hud.render(state(phase="RALLY", gates=gates, show_xray=False), size=(W, H))
    on = hud.render(state(phase="RALLY", gates=gates, show_xray=True), size=(W, H))
    assert diff(off, on) > 10_000


def test_a_camera_background_shows_through():
    cam = np.full((360, 640, 3), (0, 200, 0), dtype=np.uint8)
    plain = hud.render(state(), size=(W, H))
    with_cam = hud.render(state(), size=(W, H), background=cam)
    assert diff(plain, with_cam) > 100_000


@pytest.mark.parametrize("phase", ["LOBBY", "COUNTDOWN", "RALLY", "POINT_OVER", "MATCH_OVER"])
@pytest.mark.parametrize("mode", ["survival", "match"])
def test_every_phase_renders_in_both_modes_without_error(phase, mode):
    frame = hud.render(state(phase=phase, mode=mode, countdown=2, streak=4, record=9, player_points=3,
                             cpu_points=5, last_kmh=88.4, last_label="perfect", spin_text="TOPSPIN",
                             paddle_ab=(0.4, 0.5), arrival_ab=(0.85, 0.5), mqtt_status="ok", flash=((0, 255, 0), 0.3),
                             message="NEW RECORD"), size=(W, H))
    assert frame.shape == (H, W, 3)


def test_the_flash_overlay_tints_the_frame():
    plain = hud.render(state(phase="RALLY"), size=(W, H))
    flash = hud.render(state(phase="RALLY", flash=((0, 255, 0), 0.5)), size=(W, H))
    assert flash[:, :, 1].mean() > plain[:, :, 1].mean() + 20
