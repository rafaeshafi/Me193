"""HUD rendering: the scene plus the score panels, prompts, x-ray, camera corner (pure numpy frames, no window)."""

import numpy as np
import pytest

from pingpong import canvas, fonts, holdstart, hud, ui
from pingpong.events import GateResult

W, H = 1280, 720


def diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def state(**kw):
    return hud.HudState(**kw)


def test_render_returns_a_bgr_frame_of_the_requested_size():
    frame = hud.render(state(), size=(W, H))
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8


def test_a_ball_in_flight_changes_the_picture():
    empty = hud.render(state(phase="RALLY"), size=(W, H))
    ball = hud.render(state(phase="RALLY", ball=(0.2, 0.3, 1.0)), size=(W, H))
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


def test_the_camera_is_a_small_picture_in_the_corner_and_nothing_else_moves():
    cam = np.full((360, 640, 3), (0, 200, 0), dtype=np.uint8)
    plain = hud.render(state(), size=(W, H))
    with_cam = hud.render(state(), size=(W, H), background=cam)
    pw, ph = hud.PIP_SIZE
    x0, y0 = W - pw - 24, H - ph - 60
    assert diff(plain, with_cam) > 100_000
    assert tuple(with_cam[y0 + ph // 2, x0 + pw // 2]) == (0, 200, 0)
    mask = np.ones((H, W), dtype=bool)
    mask[y0 - 30:y0 + ph + 36, x0 - 30:x0 + pw + 30] = False                # (its white frame and soft shadow are round it)
    assert diff(plain[mask], with_cam[mask]) == 0                         # outside the corner: the same picture
    assert tuple(with_cam[y0 + 1, x0 + 1]) != (0, 200, 0)                  # the window has rounded corners


def test_a_ring_marks_your_hand_in_the_camera_picture():
    cam = np.full((360, 640, 3), (0, 200, 0), dtype=np.uint8)
    without = hud.render(state(), size=(W, H), background=cam)
    marked = hud.render(state(hand_img=(0.25, 0.5)), size=(W, H), background=cam)
    assert diff(without, marked) > 300
    unmarked = hud.render(state(hand_img=(0.25, 0.5)), size=(W, H))
    assert diff(unmarked, hud.render(state(), size=(W, H))) == 0           # no picture, no ring


def test_a_card_in_each_top_corner_names_what_it_shows(monkeypatch):
    seen = drawn_text(monkeypatch)
    hud.render(state(mode="survival", streak=7, record=12, player_name="rafae"), size=(W, H))
    assert {"STREAK", "BEST", "7", "12", "RAFAE"} <= set(seen)
    seen.clear()
    hud.render(state(mode="match", player_points=3, cpu_points=5, player_name="rafae", level_name="Club"), size=(W, H))
    assert {"POINTS", "3", "5", "RAFAE", "COCO"} <= set(seen)                    # you and the opponent of that level, by name


@pytest.mark.parametrize("phase", ["LOBBY", "COUNTDOWN", "RALLY", "POINT_OVER", "MATCH_OVER"])
@pytest.mark.parametrize("mode", ["survival", "match"])
def test_every_phase_renders_in_both_modes_without_error(phase, mode):
    frame = hud.render(state(phase=phase, mode=mode, countdown=2, streak=4, record=9, player_points=3,
                             cpu_points=5, last_kmh=88.4, last_label="perfect", spin_text="TOPSPIN",
                             ball=(0.2, 0.3, 1.0), paddle=(0.1, 0.16, 0.3), rest=(0.1, 0.16, 0.3), zone=(1.1, -0.1),
                             mqtt_status="ok", flash=((0, 255, 0), 0.3), message="NEW RECORD"), size=(W, H))
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
    real = fonts.draw

    def spy(frame, text, *a, **kw):
        seen.append(text)
        return real(frame, text, *a, **kw)

    monkeypatch.setattr(fonts, "draw", spy)
    return seen


def panels(monkeypatch):
    boxes = []
    real = ui.panel
    monkeypatch.setattr(ui, "panel", lambda frame, x, y, w, h, *a, **kw: (boxes.append((x, y, w, h)), real(frame, x, y, w, h, *a, **kw))[1])
    return boxes


def test_the_xray_never_cuts_a_gate_note_in_the_middle_of_its_numbers(monkeypatch):
    seen = drawn_text(monkeypatch)
    note = "hand 0.62 SW from the ball (limit 0.45)"
    hud.render(state(phase="RALLY", show_xray=True, gates=(GateResult("J2", False, note),
                                                          GateResult("J6", False, "paddle locked after shaking"))),
               size=(W, H))
    assert any(note in line for line in seen)
    assert any("paddle locked after shaking" in line for line in seen)


def test_the_leaderboard_panel_stays_clear_of_the_score_cards_at_the_top(monkeypatch):
    boxes = panels(monkeypatch)
    hud.render(state(phase="MATCH_OVER", mode="match", leaderboard=(("rafae", 3), ("guest", 1))), size=(W, H))
    board = [b for b in boxes if b[2] == 340]
    assert board and all(y >= 140 for _, y, _, _ in board)
    assert all(x + w <= 364 for x, _, w, _ in board)                   # and left of the centred GAME OVER / MATCH OVER


def test_the_xray_panel_stays_clear_of_the_cards_and_inside_the_frame(monkeypatch):
    boxes = panels(monkeypatch)
    hud.render(state(phase="RALLY"), size=(W, H))
    without = len(boxes)
    hud.render(state(phase="RALLY", show_xray=True, gates=(GateResult("J1", True, "timing +10 ms"),)), size=(W, H))
    xray = boxes[without:][-1]
    assert xray[0] >= 715 and xray[0] + xray[2] <= W - 10 and xray[1] >= 120          # under the right card, not over it


# --- the hub's battery ----------------------------------------------------------------------------------------------------
def test_the_status_chip_shows_the_hubs_battery_and_goes_amber_when_it_is_low(monkeypatch):
    seen = drawn_text(monkeypatch)
    hud.render(state(hub_status="ok", hub_battery=83), size=(W, H))
    assert any("HUB OK 83%" in text for text in seen)
    seen.clear()
    hud.render(state(hub_status="ok", hub_battery=None), size=(W, H))
    assert any(text.endswith("HUB OK") for text in seen)                  # unknown (not yet reported): no number
    low = hud.render(state(hub_status="ok", hub_battery=12, mqtt_status="ok"), size=(W, H))
    fine = hud.render(state(hub_status="ok", hub_battery=80, mqtt_status="ok"), size=(W, H))
    corner = (slice(H - 56, H - 14), slice(20, 360))
    assert diff(low[corner], fine[corner]) > 1000                          # the colour differs: amber vs green


def test_the_survival_mode_is_called_rally_on_screen(monkeypatch):
    seen = drawn_text(monkeypatch)
    hud.render(state(mode="survival", level_name="Rookie"), size=(W, H))
    assert any(text.startswith("RALLY") for text in seen) and not any("SURVIVAL" in text for text in seen)
    seen.clear()
    hud.render(state(mode="match", level_name="Rookie"), size=(W, H))
    assert any(text.startswith("MATCH") for text in seen)


# --- the START button (hold the hub on it) ----------------------------------------------------------------------------------
FILL = hud.rgb(255, 214, 70)                      # what the hold fills the button with


def button_rect(w=W, h=H):
    return tuple(round(f * v) for f, v in zip(holdstart.BUTTON, (w, h, w, h)))


def filled(frame, rect):
    x, y, bw, bh = rect
    patch = frame[y:y + bh, x:x + bw].reshape(-1, 3)
    return int((patch == FILL).all(axis=1).sum())


def test_the_start_button_sits_in_the_top_right_and_fills_while_the_hub_is_held_on_it():
    off = hud.render(state(phase="LOBBY"), size=(W, H))
    idle = hud.render(state(phase="LOBBY", start_button=(0.0, False)), size=(W, H))
    half = hud.render(state(phase="LOBBY", start_button=(0.5, True)), size=(W, H))
    full = hud.render(state(phase="LOBBY", start_button=(1.0, True)), size=(W, H))
    rect = button_rect()
    area = (slice(rect[1], rect[1] + rect[3]), slice(rect[0], rect[0] + rect[2]))
    assert diff(off[area], idle[area]) > 5_000
    assert filled(idle, rect) == 0 < filled(half, rect) < filled(full, rect)


def test_the_pointer_follows_the_hand_over_the_whole_screen():
    base = hud.render(state(phase="LOBBY", start_button=(0.0, False)), size=(W, H))
    left = hud.render(state(phase="LOBBY", start_button=(0.0, False), cursor=(0.2, 0.3)), size=(W, H))
    cx, cy = round(0.2 * (W - 1)), round(0.7 * (H - 1))
    spot = (slice(cy - 72, cy + 73), slice(cx - 72, cx + 73))
    assert diff(base[spot], left[spot]) > 2_000
    mask = np.ones((H, W), dtype=bool)
    mask[spot] = False
    assert diff(base[mask], left[mask]) == 0                                   # and nothing else moved
    held = hud.render(state(phase="LOBBY", start_button=(0.6, True), cursor=(0.2, 0.3)), size=(W, H))
    assert diff(left[spot], held[spot]) > 300                                   # the ring round it fills as the hold goes on


def test_the_pointer_is_clamped_to_the_screen():
    plain = hud.render(state(phase="LOBBY", start_button=(0.0, True)), size=(W, H))
    far = hud.render(state(phase="LOBBY", start_button=(0.0, True), cursor=(1.6, 1.4)), size=(W, H))
    assert far.shape == (H, W, 3) and diff(plain[:80, W - 80:], far[:80, W - 80:]) > 500


def test_the_prompts_say_to_hold_the_hub_on_start_when_the_button_is_there():
    lobby_off = hud.render(state(phase="LOBBY"), size=(W, H))
    lobby_on = hud.render(state(phase="LOBBY", start_button=(0.0, False)), size=(W, H))
    centre = (slice(H // 2 - 80, H // 2 + 70), slice(160, W - 160))
    assert diff(lobby_off[centre], lobby_on[centre]) > 15_000
    over_off = hud.render(state(phase="MATCH_OVER"), size=(W, H))
    over_on = hud.render(state(phase="MATCH_OVER", start_button=(0.0, False)), size=(W, H))
    prompt = (slice(H // 2 + 40, H // 2 + 100), slice(200, W - 200))
    assert diff(over_off[prompt], over_on[prompt]) > 8_000


def test_the_button_stays_inside_the_frame_below_the_cards_whatever_the_size():
    for size in ((1280, 720), (960, 540), (1920, 1080)):
        w, h = size
        x, y, bw, bh = button_rect(w, h)
        assert x + bw <= w - 10 and y >= round(0.116 * h) and y + bh < h // 2
        frame = hud.render(state(phase="LOBBY", start_button=(1.0, True), cursor=(0.9, 0.8)), size=size)
        assert filled(frame, (x, y, bw, bh)) > 100


# --- the paddle's grip (where a hand holds it) ----------------------------------------------------------------------------------------
def test_the_grip_is_below_the_face_for_you_and_above_it_for_the_far_player_and_turns_with_the_paddle():
    below = canvas.paddle_grip(100, 100, 20, 0.0, handle_up=False)
    above = canvas.paddle_grip(100, 100, 20, 0.0, handle_up=True)
    assert below[0] == pytest.approx(100) and below[1] > 100 + 2.0 * 20
    assert above[0] == pytest.approx(100) and above[1] < 100 - 2.0 * 20
    turned = canvas.paddle_grip(100, 100, 20, 90.0, handle_up=False)
    assert turned[0] < 100 - 2.0 * 20 and turned[1] == pytest.approx(100, abs=1e-6)       # clockwise: the handle swings to the left


def test_a_fist_round_the_handle_can_be_any_skin():
    import numpy as np

    frame = np.full((200, 200, 3), 50, dtype=np.uint8)
    canvas.draw_paddle(frame, 100, 100, 20, hand=True, skin=(10, 200, 30))
    gx, gy = canvas.paddle_grip(100, 100, 20, 0.0, handle_up=False)
    assert tuple(frame[round(gy), round(gx)]) == (10, 200, 30)


def test_go_is_a_big_word_in_the_middle_of_the_screen_not_a_banner(monkeypatch):
    sizes = {}
    real = fonts.draw
    monkeypatch.setattr(fonts, "draw", lambda frame, text, x, y, size, *a, **kw: (sizes.setdefault(text, (x, y, size)), real(frame, text, x, y, size, *a, **kw))[1])
    hud.render(state(phase="RALLY", message="GO!"), size=(W, H))
    x, y, size = sizes["GO!"]
    assert size >= 120 and abs(x - W / 2) < 5 and H * 0.4 < y < H * 0.7


# --- a friend online ------------------------------------------------------------------------------------------------------------------------------
def test_a_friend_is_named_on_the_right_card_and_stands_behind_the_table(monkeypatch):
    seen = []
    real = fonts.draw
    monkeypatch.setattr(fonts, "draw", lambda frame, text, *a, **kw: (seen.append(text), real(frame, text, *a, **kw))[1])
    computer = hud.render(state(phase="RALLY", mode="match", level_name="Club"), size=(W, H))
    assert "COCO" in seen
    seen.clear()
    friend = hud.render(state(phase="RALLY", mode="match", level_name="Club", opponent_name="MAYA"), size=(W, H))
    assert "MAYA" in seen and "COCO" not in seen
    assert diff(computer, friend) > 20_000
    assert any("ONLINE" in text for text in seen)                                      # the tag under your card


def test_an_online_game_shows_the_ping_in_place_of_the_score_topics_state_and_worries_when_it_is_slow(monkeypatch):
    seen = []
    real = fonts.draw
    monkeypatch.setattr(fonts, "draw", lambda frame, text, *a, **kw: (seen.append(text), real(frame, text, *a, **kw))[1])
    hud.render(state(phase="RALLY", mode="match", opponent_name="MAYA", ping_ms=48.4, mqtt_status="off"), size=(W, H))
    assert any("PING 48 ms" in text for text in seen) and not any("MQTT" in text for text in seen)
    seen.clear()
    hud.render(state(phase="RALLY", mode="match", opponent_name="MAYA", ping_ms=None), size=(W, H))
    assert any("PING ..." in text for text in seen)
    fast = hud.render(state(phase="RALLY", mode="match", opponent_name="MAYA", ping_ms=40.0), size=(W, H))
    slow = hud.render(state(phase="RALLY", mode="match", opponent_name="MAYA", ping_ms=900.0), size=(W, H))
    chip = (slice(H - 56, H - 14), slice(16, 420))
    assert diff(fast[chip], slow[chip]) > 3_000
