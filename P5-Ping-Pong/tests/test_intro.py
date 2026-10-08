"""The intro: the camera flies in from the clouds over the island and the pier and lands on the game's own view of the court."""

import numpy as np
import pytest

from pingpong import court3d, flow, fonts, hud, intro
from pingpong.uistate import UiState

W, H = 640, 360


def state(t_s, **kw):
    return hud.HudState(screen="INTRO", ui=UiState(screen="INTRO", t_s=t_s, **kw), player_name="rafae", level_name="Club", anim_t=t_s)


def render(t_s, size=(W, H), **kw):
    return hud.render(state(t_s, **kw), size=size)


def diff(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


GAME = court3d.Camera.for_frame(W, H)


# --- the flight -----------------------------------------------------------------------------------------------------------------------------
def test_the_flight_starts_high_and_far_and_ends_on_the_games_own_camera():
    start, end = intro.camera_at(0.0, W, H), intro.camera_at(intro.ARRIVE_S, W, H)
    assert start.pos[1] > 50 and start.pos[2] < -150
    assert end.pos == pytest.approx(GAME.pos) and end.pitch_deg == pytest.approx(GAME.pitch_deg) and end.yaw_deg == pytest.approx(0.0)
    assert end.cy == pytest.approx(GAME.cy) and end.focal == pytest.approx(GAME.focal) and end.cx == pytest.approx(GAME.cx)
    assert intro.camera_at(intro.ARRIVE_S + 5, W, H) == end                           # and it stays there


def test_the_camera_only_ever_comes_down_and_forward_never_dips_below_the_players_eye():
    cams = [intro.camera_at(k / 30, W, H) for k in range(int(intro.ARRIVE_S * 30) + 1)]
    heights, depths = [c.pos[1] for c in cams], [c.pos[2] for c in cams]
    assert all(b <= a + 1e-9 for a, b in zip(heights, heights[1:])) and all(b >= a - 1e-9 for a, b in zip(depths, depths[1:]))
    assert min(heights) >= GAME.pos[1] - 1e-9


def test_the_flight_is_smooth_with_no_jump_between_two_frames_and_it_eases_to_a_stop():
    cams = [intro.camera_at(k / 60, W, H) for k in range(int(intro.ARRIVE_S * 60) + 1)]
    steps = [np.linalg.norm(np.subtract(a.pos, b.pos)) for a, b in zip(cams, cams[1:])]
    assert max(steps) < 1.2                                                       # metres in a sixtieth of a second: no cut
    assert steps[-1] < 0.05 and steps[0] < 0.2                                    # starts and ends gently
    turns = [abs(a.pitch_deg - b.pitch_deg) + abs(a.yaw_deg - b.yaw_deg) for a, b in zip(cams, cams[1:])]
    assert max(turns) < 0.35


def test_the_flight_passes_over_the_island_and_the_pier_on_its_way_in():
    from pingpong import resort

    cx, cz, r = resort.ISLAND
    over = [c for c in (intro.camera_at(k / 30, W, H) for k in range(int(intro.ARRIVE_S * 30))) if abs(c.pos[0] - cx) < r and cz - r < c.pos[2] < cz + r]
    assert over and max(c.pos[1] for c in over) < 40                              # low enough over the island to see the palms


def test_the_intro_is_as_long_as_the_flow_plays_it():
    assert intro.LENGTH_S == flow.INTRO_S and intro.ARRIVE_S < intro.LENGTH_S


# --- the pictures ------------------------------------------------------------------------------------------------------------------------------
def test_every_moment_of_the_intro_is_a_full_picture_and_it_is_the_same_every_time():
    for t in (0.0, 1.0, 3.5, 6.0, 8.0, 9.5, 10.0):
        a, b = render(t), render(t)
        assert a.shape == (H, W, 3) and np.array_equal(a, b) and a.std() > 4, t


def test_the_picture_changes_as_the_camera_flies():
    frames = [render(t) for t in (1.0, 3.0, 5.0, 7.0)]
    assert all(diff(a, b) > 8 for a, b in zip(frames, frames[1:]))


def test_it_comes_up_out_of_a_white_flash_of_sunlight():
    assert render(0.0).mean() > 215 > render(1.2).mean()


def test_the_name_of_the_game_is_over_the_first_shots_and_gone_when_the_camera_comes_down(monkeypatch):
    seen = []
    real = fonts.draw
    monkeypatch.setattr(fonts, "draw", lambda frame, text, *a, **kw: (seen.append(text), real(frame, text, *a, **kw))[1])
    render(1.8)
    assert "PING-PONG" in seen
    seen.clear()
    render(7.0)
    assert "PING-PONG" not in seen


def test_a_hint_says_how_to_skip_after_a_moment(monkeypatch):
    seen = []
    real = fonts.draw
    monkeypatch.setattr(fonts, "draw", lambda frame, text, *a, **kw: (seen.append(text), real(frame, text, *a, **kw))[1])
    render(0.6)
    assert not any("skip" in text.lower() for text in seen)
    render(4.0)
    assert any("skip" in text.lower() for text in seen)


def test_black_bars_frame_the_flight_and_slide_away_before_the_title():
    mid, end = render(4.0), render(intro.LENGTH_S - 0.05)
    assert mid[:8].mean() < 4 and mid[-8:].mean() < 4
    assert end[:8].mean() > 40 and end[-8:].mean() > 40


def test_the_last_picture_of_the_intro_is_the_court_the_title_starts_from():
    last = render(intro.LENGTH_S)
    title = hud.render(hud.HudState(screen="TITLE", ui=UiState(screen="TITLE", t_s=0.0), player_name="rafae", level_name="Club"), size=(W, H))
    court_rows = (slice(120, 260), slice(130, 510))                                # the table and the opponent, clear of the logo
    assert diff(last[court_rows], title[court_rows]) < 14


def test_a_skip_progress_fills_the_hint():
    plain, held = render(4.0), render(4.0, hover="skip", progress=0.6)
    assert diff(plain, held) > 0.3


def test_drawing_at_full_size_works_too():
    f = render(5.0, size=(1280, 720))
    assert f.shape == (720, 1280, 3)


# --- the video ---------------------------------------------------------------------------------------------------------------------------------
def test_the_intro_can_be_written_out_as_a_video_to_show_somebody(tmp_path):
    import cv2

    path = tmp_path / "intro.mp4"
    n = intro.export_video(path, size=(160, 90), fps=10, seconds=1.0, start_s=3.0)
    assert n == 10 and path.exists() and path.stat().st_size > 1000
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
    cap.release()
