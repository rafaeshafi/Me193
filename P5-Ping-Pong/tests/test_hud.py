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


def test_a_swing_trace_is_drawn_with_its_threshold_and_an_empty_one_changes_nothing():
    import math

    base = hud.render(state(), size=(W, H))
    trace = tuple(600.0 * math.sin(math.pi * k / 10) for k in range(11)) + (0.0,) * 20
    drawn = hud.render(state(swing_trace=trace, swing_scale=1200.0, swing_threshold=180.0), size=(W, H))
    assert diff(base, drawn) > 3_000
    assert diff(base, hud.render(state(swing_trace=(), swing_scale=1200.0, swing_threshold=180.0), size=(W, H))) == 0
    hud.render(state(swing_trace=(5.0,), swing_scale=1200.0), size=(W, H))        # a single point must not crash


def test_a_stronger_swing_is_drawn_taller():
    weak = hud.render(state(swing_trace=(0.0, 200.0, 0.0), swing_scale=1200.0), size=(W, H))
    strong = hud.render(state(swing_trace=(0.0, 1000.0, 0.0), swing_scale=1200.0), size=(W, H))
    assert diff(weak, strong) > 500


def test_the_longest_xray_notes_stay_inside_the_frame():
    long_notes = (GateResult("J1", True, "timing -220 ms (window -220/+140)"),
                  GateResult("J2", True, "hand 0.00 SW from the ball (limit 0.45)"),
                  GateResult("J3", False, "swing too weak (123 < 180 dps) and then some"),
                  GateResult("J4", True, "logged-only"), GateResult("J5", False, "refractory: too soon after the last hit"),
                  GateResult("J6", False, "paddle locked after shaking"))
    base = hud.render(state(show_xray=False), size=(W, H))
    shown = hud.render(state(show_xray=True, gates=long_notes), size=(W, H))
    assert diff(base[:, -6:], shown[:, -6:]) == 0           # no text pixel touches the right edge


BOARD = (("maya", 21), ("rafae", 17), ("omar", 12), ("zed", 9), ("guest", 4))


def test_the_leaderboard_is_a_panel_on_the_left_that_keeps_clear_of_the_prompts():
    base = hud.render(state(phase="MATCH_OVER", streak=12, record=17), size=(W, H))
    board = hud.render(state(phase="MATCH_OVER", streak=12, record=17, leaderboard=BOARD), size=(W, H))
    assert diff(base[90:420, 0:420], board[90:420, 0:420]) > 20_000           # the panel is there ...
    assert diff(base[:, 430:], board[:, 430:]) == 0                            # ... and nothing else moved
    assert diff(base[H - 150:H - 90], board[H - 150:H - 90]) == 0              # "SPACE to play again" is untouched


def test_the_players_own_row_is_highlighted():
    plain = hud.render(state(phase="MATCH_OVER", leaderboard=BOARD, player_name="nobody"), size=(W, H))
    mine = hud.render(state(phase="MATCH_OVER", leaderboard=BOARD, player_name="Rafae"), size=(W, H))
    assert diff(plain[170:260, 0:420], mine[170:260, 0:420]) > 500            # rafae is row 2: recoloured
    assert diff(plain[290:420, 0:420], mine[290:420, 0:420]) == 0             # the other rows are not


def test_the_board_is_only_drawn_on_the_end_screen():
    rally = hud.render(state(phase="RALLY", leaderboard=BOARD), size=(W, H))
    assert diff(rally, hud.render(state(phase="RALLY"), size=(W, H))) == 0


# --- a notice longer than the screen is wide ---------------------------------------------------------------------------
LONG = "UNCALIBRATED: run ./pp calibrate_swing --player rafae --swing-source pose --no-hub"


def test_a_short_message_stays_one_big_line_and_a_long_one_shrinks_and_wraps_to_fit_the_screen():
    import cv2

    short, scale = canvas.fit_text("MISSED", W - 100, max_scale=1.7, min_scale=0.7, thickness=4)
    assert short == ["MISSED"] and scale == 1.7
    lines, scale = canvas.fit_text(LONG, W - 100, max_scale=1.7, min_scale=0.7, thickness=4)
    assert 0.7 <= scale < 1.7                                          # shrunk (and wrapped if even that is too wide)
    assert " ".join(lines) == LONG
    assert all(cv2.getTextSize(line, canvas.FONT, scale, 4)[0][0] <= W - 100 for line in lines)


def test_text_that_cannot_fit_even_at_the_smallest_size_is_cut_with_dots_not_drawn_off_screen():
    import cv2

    lines, scale = canvas.fit_text("word " * 100, 600, max_scale=1.7, min_scale=0.7, thickness=4, max_lines=2)
    assert len(lines) == 2 and scale == 0.7 and lines[-1].endswith("...")
    assert all(cv2.getTextSize(line, canvas.FONT, scale, 4)[0][0] <= 600 for line in lines)
    lines, scale = canvas.fit_text("x" * 400, 600, max_scale=1.7, min_scale=0.7, thickness=4)       # one endless word
    assert len(lines) == 1 and lines[0].endswith("...")
    assert cv2.getTextSize(lines[0], canvas.FONT, scale, 4)[0][0] <= 600


def test_a_long_lobby_notice_is_drawn_inside_the_frame_and_clear_of_the_start_prompt():
    plain = hud.render(state(phase="LOBBY"), size=(W, H))
    noted = hud.render(state(phase="LOBBY", message=LONG), size=(W, H))
    changed = np.abs(noted.astype(np.int32) - plain.astype(np.int32)).sum(axis=2) > 0
    rows, cols = np.where(changed)
    assert rows.size > 1000
    assert cols.min() >= 20 and cols.max() <= W - 20                  # nothing clipped at the left or right edge
    assert rows.min() > 410                                           # below "SHOW THE START CARD / or press SPACE"
