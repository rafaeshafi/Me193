"""The bench tools' live camera window: a mirrored picture that tells you whether you are in position.

macOS draws an OpenCV window only while the main thread keeps calling waitKey, so every wait of a bench (the Enter
prompts, getting into position, the measurement itself) has to keep the window drawing.
"""

import os
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pingpong import preview
from pingpong.realenv import RealEnv

W, H = 640, 360


def body(overrides=None):
    """33 landmarks, all clearly in view; overrides = {index: (x, y, visibility)}."""
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.95) for _ in range(33)]
    for i, (x, y, vis) in (overrides or {}).items():
        lm[i] = SimpleNamespace(x=x, y=y, visibility=vis)
    return lm


class Vision:
    """What the preview needs of a VisionWorker."""

    def __init__(self, frame=None, landmarks=None):
        self.frame, self.last_landmarks = frame, landmarks

    def latest_frame(self):
        return self.frame


BLACK = np.zeros((H, W, 3), np.uint8)


# --- is the player in position? ----------------------------------------------------------------------------------
def test_head_shoulders_and_hips_in_view_is_in_position():
    assert preview.missing_parts(body()) == []


def test_nobody_in_view_is_reported_as_no_person_not_as_missing_parts():
    assert preview.missing_parts(None) == ["person"]


def test_a_part_that_is_hidden_or_outside_the_picture_is_named():
    assert preview.missing_parts(body({23: (0.45, 0.95, 0.2)})) == ["hips"]                  # cut off at the bottom
    assert preview.missing_parts(body({0: (0.5, -0.05, 0.9)})) == ["head"]                   # above the top edge
    assert preview.missing_parts(body({12: (1.2, 0.4, 0.9)})) == ["shoulders"]               # right shoulder outside
    assert preview.missing_parts(body({0: (0.5, 0.1, 0.1), 24: (0.5, 0.9, 0.1)})) == ["head", "hips"]


# --- the picture --------------------------------------------------------------------------------------------------
def test_the_picture_is_mirrored_like_a_mirror_and_resized_to_the_window_width():
    frame = BLACK.copy()
    frame[:, :80] = 255                                                           # the camera's left edge
    image = preview.compose(frame, "caption", body(), width=480)
    assert image.shape == (270, 480, 3)
    assert image[135, -5].min() > 200 and image[135, 5].max() < 60                # ... shows on the RIGHT in the mirror


def test_the_joints_are_drawn_where_the_mirror_shows_them_green_when_seen_and_red_when_not():
    lm = body({16: (0.2, 0.5, 0.9), 15: (0.8, 0.5, 0.1)})
    image = preview.compose(BLACK, "c", lm, width=480)
    assert tuple(image[135, round((1 - 0.2) * 480)]) == preview.GREEN             # camera-left is mirror-right
    assert tuple(image[135, round((1 - 0.8) * 480)]) == preview.RED


def test_the_status_line_is_green_only_when_you_are_in_position():
    ok = preview.compose(BLACK, "c", body(), width=480)
    hidden_hips = preview.compose(BLACK, "c", body({23: (0.5, 0.99, 0.1)}), width=480)
    nobody = preview.compose(BLACK, "c", None, width=480)
    green = np.array(preview.GREEN)
    assert (ok[-50:] == green).all(axis=2).any()
    assert not (hidden_hips[-50:] == green).all(axis=2).any()
    assert not (nobody[-50:] == green).all(axis=2).any() and (nobody[-50:] > 200).any()      # but the text is there


def test_the_caption_is_drawn_at_the_top():
    assert (preview.compose(BLACK, "Stand at the play position", None, width=480)[:60] > 200).any()
    assert not (preview.compose(BLACK, "", None, width=480)[:60] > 200).any()


# --- the window ----------------------------------------------------------------------------------------------------
def make_preview(vision=None, keys=(), fps=25.0):
    shown, clock, pressed = [], {"t": 0.0}, iter(keys)
    window = preview.Preview(shown.append, wait_key=lambda ms: next(pressed, -1), now=lambda: clock["t"], fps=fps)
    if vision is not None:
        window.attach(vision)
    return window, shown, clock


def test_a_preview_with_no_camera_attached_does_nothing():
    window, shown, _ = make_preview()
    assert window.active is False and window.update() == preview.NO_KEY and shown == []


def test_update_draws_the_newest_frame_at_most_at_the_frame_rate():
    window, shown, clock = make_preview(Vision(np.full((H, W, 3), 50, np.uint8), body()), fps=25.0)
    window.caption = "hello"
    window.update()
    window.update()                                                               # at the same instant: one redraw
    clock["t"] = 0.01
    window.update()                                                               # inside the 40 ms period
    assert len(shown) == 1 and shown[0].shape[1] == preview.WIDTH
    clock["t"] = 0.05
    window.update()
    assert len(shown) == 2


def test_the_window_handles_events_on_every_update_even_when_nothing_is_redrawn():
    calls = []
    window = preview.Preview(lambda image: None, wait_key=lambda ms: calls.append(ms) or -1, now=lambda: 0.0)
    window.attach(Vision(BLACK))
    for _ in range(5):
        window.update()
    assert calls == [1] * 5                                                       # macOS draws only while waitKey runs


def test_a_camera_that_has_not_delivered_a_frame_yet_draws_nothing_and_does_not_fail():
    window, shown, _ = make_preview(Vision(None))
    assert window.update() == preview.NO_KEY and shown == []


def test_update_returns_the_key_pressed_in_the_window():
    window, _, _ = make_preview(Vision(BLACK), keys=[ord("y")])
    assert window.update() == ord("y")


def test_a_still_picture_stays_up_the_live_drawing_stops():
    window, shown, clock = make_preview(Vision(BLACK, body()))
    window.show_still("the HUD")
    clock["t"] = 5.0
    window.update()
    assert shown == ["the HUD"]


def test_a_window_that_cannot_draw_is_dropped_and_the_bench_goes_on(capsys):
    def broken(image):
        raise cv2.error("no display")

    window = preview.Preview(broken, wait_key=lambda ms: -1, now=lambda: 0.0)
    window.attach(Vision(BLACK, body()))
    assert window.update() == preview.NO_KEY and window.active is False
    assert "preview" in capsys.readouterr().out.lower()


def test_the_real_environment_keeps_calling_the_idle_hook_while_it_sleeps():
    env, calls = RealEnv(), []
    env.on_idle = lambda: calls.append(1)
    t0 = time.monotonic()
    env.sleep(0.2)
    assert time.monotonic() - t0 >= 0.2 and len(calls) >= 5


# --- the prompts -----------------------------------------------------------------------------------------------------
class Terminal:
    """A stand-in for the terminal's stdin: type() puts a line in front of the program."""

    def __init__(self):
        r, self._w = os.pipe()
        self.stdin = os.fdopen(r)

    def type(self, text):
        os.write(self._w, text.encode())

    def close(self):
        self.stdin.close()
        os.close(self._w)


@pytest.fixture
def terminal():
    t = Terminal()
    yield t
    t.close()


def prompts(terminal, keys=(), camera=True):
    window, shown, _ = make_preview(Vision(BLACK, body()) if camera else None, keys=keys)
    prompt, ask = preview.make_prompts(window, stdin=terminal.stdin, poll_s=0.001, out=lambda *a, **k: None)
    return window, prompt, ask


def test_a_prompt_keeps_the_window_alive_while_it_waits_and_ends_on_a_typed_enter(terminal):
    window, prompt, _ = prompts(terminal)
    updates, real = [], window.update
    window.update = lambda: updates.append(1) or real()
    threading.Timer(0.1, terminal.type, ("\n",)).start()
    prompt("Stand at the play position. Press Enter.")
    assert len(updates) >= 3


@pytest.mark.parametrize("key", [13, 10, 32])
def test_a_prompt_also_ends_on_enter_or_space_pressed_in_the_window(terminal, key):
    window, prompt, _ = prompts(terminal, keys=[-1, -1, key])
    prompt("Stand at the play position. Press Enter.")                            # nothing typed in the terminal


def test_the_caption_is_the_instruction_with_a_hint_while_waiting_and_just_the_instruction_after(terminal):
    window, prompt, _ = prompts(terminal)
    terminal.type("\n")
    seen, real = [], window.update
    window.update = lambda: seen.append(window.caption) or real()
    prompt("Hold card 1 up for 5 s. Press Enter.")
    assert seen[0] == "Hold card 1 up for 5 s." + preview.WAIT_HINT
    assert window.caption == "Hold card 1 up for 5 s."


def test_ask_takes_y_or_n_from_the_window_and_ignores_every_other_key(terminal):
    assert prompts(terminal, keys=[ord("x"), 13, ord("y")])[2]("Readable? (y/n)") is True
    assert prompts(terminal, keys=[ord("n")])[2]("Readable? (y/n)") is False


@pytest.mark.parametrize("typed, expected", [("y\n", True), (" Y \n", True), ("\n", False), ("no\n", False)])
def test_ask_takes_a_typed_answer_from_the_terminal(terminal, typed, expected):
    terminal.type(typed)
    assert prompts(terminal)[2]("Readable? (y/n)") is expected


def test_without_a_camera_the_prompts_are_plain_terminal_questions(terminal, monkeypatch):
    _, prompt, ask = prompts(terminal, camera=False)
    answers = iter(["", " Y "])
    monkeypatch.setattr("builtins.input", lambda text="": next(answers))
    prompt("anything")
    assert ask("really?") is True
