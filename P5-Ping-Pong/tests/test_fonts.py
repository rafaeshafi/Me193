"""Text in a rounded typeface, drawn as cached sprites; the old OpenCV font stands in when the typeface is missing."""

import numpy as np
import pytest

from pingpong import fonts

W, H = 640, 360
GREEN = (60, 220, 90)


def blank(value=0):
    return np.full((H, W, 3), value, dtype=np.uint8)


def changed(a, b):
    return (a != b).any(axis=2)


def box(mask):
    rows, cols = np.where(mask)
    return cols.min(), rows.min(), cols.max(), rows.max()


@pytest.fixture(params=["typeface", "fallback"])
def both(request, monkeypatch):
    """Every behaviour holds with the rounded typeface and with the plain OpenCV one."""
    if request.param == "fallback":
        monkeypatch.setattr(fonts, "_load", lambda size: None)
    elif not fonts.available():
        pytest.skip("no rounded typeface on this machine")
    return request.param


def test_a_typeface_is_found_on_this_machine():
    assert fonts.available()


def test_bigger_text_is_wider_and_taller(both):
    small, big = fonts.measure("Rally", 30), fonts.measure("Rally", 60)
    assert big[0] > 1.7 * small[0] and big[1] > 1.7 * small[1]
    assert fonts.measure("Rally Rally", 40)[0] > 1.8 * fonts.measure("Rally", 40)[0]


def test_text_is_drawn_in_the_colour_asked_for(both):
    frame = blank()
    fonts.draw(frame, "START", W // 2, H // 2, 80, GREEN)
    assert (frame == np.array(GREEN, dtype=np.uint8)).all(axis=2).sum() > 300


@pytest.mark.parametrize("anchor, edge", [("lm", "left"), ("mm", "centre"), ("rm", "right")])
def test_the_anchor_says_which_part_of_the_text_stands_at_x(both, anchor, edge):
    frame = blank()
    fonts.draw(frame, "Hello", 320, 180, 70, GREEN, anchor=anchor)
    x0, _, x1, _ = box(changed(frame, blank()))
    if edge == "left":
        assert abs(x0 - 320) <= 12
    elif edge == "centre":
        assert abs((x0 + x1) / 2 - 320) <= 14
    else:
        assert abs(x1 - 320) <= 12


def test_the_vertical_anchor_puts_the_middle_top_or_bottom_of_the_line_at_y(both):
    mid, top, bottom = blank(), blank(), blank()
    fonts.draw(mid, "HELLO", 320, 180, 70, GREEN, anchor="mm")
    fonts.draw(top, "HELLO", 320, 180, 70, GREEN, anchor="mt")
    fonts.draw(bottom, "HELLO", 320, 180, 70, GREEN, anchor="mb")
    ym = box(changed(mid, blank()))
    yt = box(changed(top, blank()))
    yb = box(changed(bottom, blank()))
    assert yt[1] > ym[1] > yb[1]                           # the same text, drawn higher when anchored at its bottom
    assert yb[3] <= 180 + 12 and yt[1] >= 180 - 12


def test_an_outline_is_drawn_round_the_letters_in_its_own_colour(both):
    frame = blank()
    fonts.draw(frame, "OK", 320, 180, 90, (255, 255, 255), outline=(200, 0, 0), outline_px=5)
    assert (frame == np.array((200, 0, 0), dtype=np.uint8)).all(axis=2).sum() > 200


def test_a_shadow_sits_below_and_to_the_right_of_the_text(both):
    plain, shadowed = blank(), blank()
    fonts.draw(plain, "Hi", 320, 180, 90, GREEN)
    fonts.draw(shadowed, "Hi", 320, 180, 90, GREEN, shadow=(6, 6, (0, 0, 80), 0.6))
    extra = changed(plain, shadowed)
    assert extra.sum() > 100
    px0, py0, px1, py1 = box(changed(plain, blank()))
    sx0, sy0, sx1, sy1 = box(changed(shadowed, blank()))
    assert sx1 > px1 and sy1 > py1 and sx0 >= px0 - 1


def test_text_partly_off_the_picture_is_cut_not_an_error(both):
    frame = blank()
    fonts.draw(frame, "OFF THE EDGE", -30, 10, 80, GREEN, anchor="lm")
    fonts.draw(frame, "OFF THE EDGE", W - 10, H + 5, 80, GREEN, anchor="lm")
    assert frame.shape == (H, W, 3)


def test_translucent_text_blends_with_what_is_behind_it(both):
    frame = blank(100)
    fonts.draw(frame, "SOFT", 320, 180, 90, (255, 255, 255), opacity=0.5)
    inside = frame[(frame != 100).any(axis=2)]
    assert inside.max() < 240 and inside.max() > 150


def test_the_same_text_is_only_rendered_once():
    if not fonts.available():
        pytest.skip("no rounded typeface on this machine")
    fonts.draw(blank(), "cache me", 100, 100, 41, GREEN)
    before = fonts._sprite.cache_info().hits
    fonts.draw(blank(), "cache me", 300, 200, 41, GREEN)
    assert fonts._sprite.cache_info().hits == before + 1


def test_text_that_fits_shrinks_until_it_is_no_wider_than_the_space(both):
    size = fonts.fit_size("A rather long notice for a narrow space", 300, max_size=60, min_size=14)
    assert 14 <= size < 60 and fonts.measure("A rather long notice for a narrow space", size)[0] <= 300
    assert fonts.fit_size("OK", 300, max_size=60, min_size=14) == 60
    assert fonts.fit_size("x" * 200, 100, max_size=60, min_size=14) == 14          # as small as it goes
