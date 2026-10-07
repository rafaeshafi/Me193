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


# --- what the x-ray panel and the leaderboard panel must not do ---------------------------------------------------------
def drawn_text(monkeypatch):
    seen = []
    real = canvas.draw_text

    def spy(frame, text, org, *a, **kw):
        seen.append(text)
        return real(frame, text, org, *a, **kw)

    monkeypatch.setattr(canvas, "draw_text", spy)
    return seen


def test_the_xray_never_cuts_a_gate_note_in_the_middle_of_its_numbers(monkeypatch):
    seen = drawn_text(monkeypatch)
    note = "hand 0.62 SW from the ball (limit 0.45)"
    hud.render(state(phase="RALLY", show_xray=True, gates=(GateResult("J2", False, note),
                                                          GateResult("J6", False, "paddle locked after shaking"))),
               size=(W, H))
    assert note in " ".join(seen) or any(note in line for line in seen)
    assert "paddle locked after shaking" in " ".join(seen)


def test_the_leaderboard_panel_stays_clear_of_the_score_line_at_the_top(monkeypatch):
    boxes = []
    real = canvas.dim_rect
    monkeypatch.setattr(canvas, "dim_rect", lambda frame, x, y, w, h, *a, **kw: (boxes.append((x, y, w, h)),
                                                                                  real(frame, x, y, w, h, *a, **kw)))
    hud.render(state(phase="MATCH_OVER", mode="match", leaderboard=(("rafae", 3), ("guest", 1))), size=(W, H))
    assert boxes and all(y >= 140 for _, y, _, _ in boxes)
    assert all(x + w <= 364 for x, _, w, _ in boxes)                   # and left of the centred GAME OVER / MATCH OVER


def test_the_xray_panel_stays_clear_of_the_centred_best_line(monkeypatch):
    boxes = []
    real = canvas.dim_rect
    monkeypatch.setattr(canvas, "dim_rect", lambda frame, x, y, w, h, *a, **kw: (boxes.append((x, y, w, h)),
                                                                                  real(frame, x, y, w, h, *a, **kw)))
    hud.render(state(phase="RALLY", show_xray=True, gates=(GateResult("J1", True, "timing +10 ms"),)), size=(W, H))
    assert boxes and all(x >= 715 for x, _, _, _ in boxes) and all(x + w <= W - 10 for x, _, w, _ in boxes)


# --- the hub's battery ----------------------------------------------------------------------------------------------------
def test_the_top_bar_shows_the_hubs_battery_and_goes_amber_when_it_is_low(monkeypatch):
    seen = drawn_text(monkeypatch)
    hud.render(state(hub_status="ok", hub_battery=83), size=(W, H))
    assert any("HUB OK 83%" in text for text in seen)
    seen.clear()
    hud.render(state(hub_status="ok", hub_battery=None), size=(W, H))
    assert any(text.endswith("HUB OK") for text in seen)                  # unknown (not yet reported): no number
    low = hud.render(state(hub_status="ok", hub_battery=12, mqtt_status="ok"), size=(W, H))
    fine = hud.render(state(hub_status="ok", hub_battery=80, mqtt_status="ok"), size=(W, H))
    top = (slice(20, 60), slice(W - 700, W - 20))
    assert diff(low[top], fine[top]) > 1000                                # the colour differs: amber vs green


# --- the hand plane: where the paddle and the target are drawn, in the judge's own units ------------------------------------
BOX_SW = (3.0, 2.0)                                   # the reach box in shoulder widths
S_PX = hud.PLANE_SW_PX * H                            # pixels per shoulder width on the hand plane


def on_ring(frame, color, cx, cy, r, tol=3):
    """Is `color` drawn within `tol` px of radius r, to the right, left, above and below the centre?"""
    def hit(x, y):
        return any(tuple(frame[y + dy, x + dx]) == color for dx in range(-tol, tol + 1) for dy in range(-tol, tol + 1)
                   if 0 <= y + dy < H and 0 <= x + dx < W)
    return all(hit(cx + dx, cy + dy) for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)))


def test_a_shoulder_width_is_the_same_number_of_pixels_across_and_up_on_the_hand_plane():
    # the court's own mapping squashed the height 2.5x against the width: rings that looked like they touched were a
    # shoulder width apart and the judge said miss
    cx, cy = hud.plane_xy((0.5, 0.5), BOX_SW, W, H)
    ax, ay = hud.plane_xy((0.5 + 1.0 / BOX_SW[0], 0.5), BOX_SW, W, H)          # one shoulder width to the player's right
    bx, by = hud.plane_xy((0.5, 0.5 + 1.0 / BOX_SW[1]), BOX_SW, W, H)          # one shoulder width up
    assert (ax - cx, ay - cy) == (round(S_PX), 0) and (bx - cx, by - cy) == (0, -round(S_PX))
    assert hud.plane_xy((0.0, 1.0), BOX_SW, W, H)[1] > 0.25 * H                 # the whole box stays below the score


def test_the_target_ring_is_the_judges_hit_zone_a_circle_of_the_levels_radius():
    frame = hud.render(state(phase="RALLY", arrival_ab=(0.5, 0.5), box_sw=BOX_SW, radius_sw=0.55), size=(W, H))
    cx, cy = hud.plane_xy((0.5, 0.5), BOX_SW, W, H)
    assert on_ring(frame, hud.AMBER, cx, cy, round(0.55 * S_PX))


def test_the_paddle_is_a_dot_at_the_hand_on_the_same_plane():
    frame = hud.render(state(phase="RALLY", paddle_ab=(0.8, 0.3), box_sw=BOX_SW), size=(W, H))
    px, py = hud.plane_xy((0.8, 0.3), BOX_SW, W, H)
    assert tuple(frame[py, px]) == hud.GREEN


def test_the_incoming_ball_arrives_on_the_target_ring_not_at_the_edge_of_the_table():
    ab = (0.8, 0.3)
    tx, ty = hud.plane_xy(ab, BOX_SW, W, H)
    arrived = hud.render(state(phase="RALLY", arrival_ab=ab, box_sw=BOX_SW, ball=(canvas_x(ab), 1.0, 0.0)), size=(W, H))
    far = hud.render(state(phase="RALLY", arrival_ab=ab, box_sw=BOX_SW, ball=(canvas_x(ab), 0.3, 0.1)), size=(W, H))
    assert tuple(arrived[ty, tx]) == (40, 160, 255)                           # the ball's own colour, on the target
    assert tuple(far[ty, tx]) != (40, 160, 255)                               # still out on the table


def canvas_x(ab):
    from pingpong import physics
    return physics.x_of_a(ab[0])


def test_the_arrival_window_is_outlined_so_the_player_sees_where_balls_can_come():
    frame = hud.render(state(phase="RALLY", box_sw=BOX_SW, reach=0.6), size=(W, H))
    cx, cy = hud.plane_xy((0.5, 0.5), BOX_SW, W, H)
    left = cx - round(0.5 * 0.6 * BOX_SW[0] * S_PX)
    assert tuple(frame[cy, left]) == hud.GREY
    assert tuple(frame[cy, cx - round(0.5 * BOX_SW[0] * S_PX)]) != hud.GREY      # the full box is not what is outlined

