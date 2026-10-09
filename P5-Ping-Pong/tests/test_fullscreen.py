"""The game's window: full screen on launch, and a picture that is never stretched or left with a band in the window's own colour
(fullscreen.py).  OpenCV's macOS window keeps a picture's shape, scales it to the window's width and sits it at the bottom, so the
picture is padded with black to the shape of the window before it is shown."""

import cv2
import numpy as np
import pytest

from pingpong import fullscreen

TITLE = "P5 test"
W, H = 1280, 720
PICTURE = (np.arange(H * W * 3) % 251 + 1).astype(np.uint8).reshape(H, W, 3)         # never black: the bars can be told from it


class FakeCv2:
    """The few things of OpenCV's window API the game uses, written down."""
    WINDOW_NORMAL, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN = cv2.WINDOW_NORMAL, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN

    def __init__(self):
        self.calls, self.shown = [], []

    def namedWindow(self, title, *flags):
        self.calls.append(("namedWindow", title) + flags)

    def setWindowProperty(self, title, prop, value):
        self.calls.append(("setWindowProperty", title, prop, value))

    def imshow(self, title, picture):
        self.calls.append(("imshow", title))
        self.shown.append(picture)

    def waitKey(self, ms):
        self.calls.append(("waitKey", ms))
        return -1

    def setMouseCallback(self, title, callback):
        self.mouse_callback = callback


def make(fullscreen_on=True, view=(1512.0, 949.0), clock=None):
    asked = []

    def view_size(title):
        asked.append(title)
        return view

    t = clock if clock is not None else [0.0]
    window = fullscreen.Window(TITLE, size=(W, H), fullscreen=fullscreen_on, cv2=FakeCv2(), view_size=view_size, now=lambda: t[0])
    return window, window._cv2, asked


# --- padding the picture to the shape of the window ---------------------------------------------------------------------------------------
def test_a_window_taller_than_the_picture_gets_black_bars_above_and_below_and_the_picture_is_untouched():
    padded, (left, top) = fullscreen.pad_to(PICTURE, 1512 / 949)
    assert padded.shape == (803, W, 3) and (left, top) == (0, 41)
    assert np.array_equal(padded[top:top + H], PICTURE)
    assert not padded[:top].any() and not padded[top + H:].any()
    assert abs(padded.shape[1] / padded.shape[0] - 1512 / 949) < 0.002


def test_a_window_wider_than_the_picture_gets_black_bars_left_and_right():
    padded, (left, top) = fullscreen.pad_to(PICTURE, 2.0)
    assert padded.shape == (H, 1440, 3) and (left, top) == (80, 0)
    assert np.array_equal(padded[:, left:left + W], PICTURE)
    assert not padded[:, :left].any() and not padded[:, left + W:].any()


def test_a_window_of_the_pictures_own_shape_gets_the_picture_itself():
    for aspect in (W / H, W / H + 0.001, 0):
        padded, offset = fullscreen.pad_to(PICTURE, aspect)
        assert padded is PICTURE and offset == (0, 0)


# --- the window ---------------------------------------------------------------------------------------------------------------------------
def test_the_window_opens_full_screen_and_shows_black_at_once_so_a_slow_start_is_not_a_blank_white_screen():
    window, cv, _ = make()
    window.open()
    assert cv.calls[0] == ("namedWindow", TITLE, cv.WINDOW_NORMAL)
    assert cv.calls[1] == ("setWindowProperty", TITLE, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    assert [c[0] for c in cv.calls[2:]] == ["imshow", "waitKey"] and cv.shown[0].shape == (H, W, 3) and not cv.shown[0].any()


def test_windowed_is_the_window_the_game_always_had_the_size_of_the_picture_and_nothing_else():
    window, cv, asked = make(fullscreen_on=False)
    window.open()
    assert cv.calls == [("namedWindow", TITLE)]
    window.show(PICTURE)
    assert cv.shown == [PICTURE] and cv.shown[0] is PICTURE and asked == []


def test_a_full_screen_picture_is_padded_to_the_shape_of_the_window():
    window, cv, _ = make(view=(1512.0, 949.0))
    window.show(PICTURE)
    shown = cv.shown[0]
    assert shown.shape == (803, W, 3) and abs(shown.shape[1] / shown.shape[0] - 1512 / 949) < 0.002
    assert np.array_equal(shown[41:41 + H], PICTURE)


def test_a_window_whose_size_cannot_be_asked_gets_the_picture_as_it_is():
    window, cv, _ = make(view=None)
    window.show(PICTURE)
    assert cv.shown[0] is PICTURE


def test_the_size_of_the_window_is_asked_about_once_a_second_not_every_picture():
    clock = [0.0]
    window, cv, asked = make(clock=clock)
    for _ in range(90):                                              # 1.5 s of pictures
        window.show(PICTURE)
        clock[0] += 1 / 60
    assert 2 <= len(asked) <= 3


def test_the_picture_follows_a_window_that_changes_shape():
    clock = [0.0]
    view = {"size": (1512.0, 949.0)}
    window = fullscreen.Window(TITLE, size=(W, H), cv2=FakeCv2(), view_size=lambda title: view["size"], now=lambda: clock[0])
    window.show(PICTURE)
    view["size"] = (1280.0, 720.0)                                   # the player left full screen
    clock[0] += 1.5
    window.show(PICTURE)
    assert window._cv2.shown[0].shape[0] == 803 and window._cv2.shown[1] is PICTURE


def test_a_mouse_position_over_the_padded_picture_is_a_position_in_the_picture():
    window, _, _ = make(view=(1512.0, 949.0))
    assert window.picture_xy(300, 120) == (300, 120)                 # before anything was shown: no bars
    window.show(PICTURE)
    assert window.picture_xy(300, 41 + 100) == (300, 100)
    wide, _, _ = make(view=(2560.0, 1280.0))
    wide.show(PICTURE)
    assert wide.picture_xy(80 + 640, 360) == (640, 360)


def test_the_size_of_a_window_that_is_not_there_is_not_known_and_asking_changes_nothing():
    assert fullscreen.view_size("no window has this title") is None
