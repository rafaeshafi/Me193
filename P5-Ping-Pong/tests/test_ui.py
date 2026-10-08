"""The shapes the menus and the HUD are built from: soft panels, pill buttons, rings, stars, glow, confetti (pure numpy frames)."""

import numpy as np
import pytest

from pingpong import ui

W, H = 640, 360
WHITE, NAVY, ORANGE = (255, 255, 255), (90, 50, 20), (30, 160, 255)


def blank(value=60):
    return np.full((H, W, 3), value, dtype=np.uint8)


def changed(a, b):
    return (a != b).any(axis=2)


def px(frame, x, y):
    return tuple(int(c) for c in frame[y, x])


# --- masks and gradients -------------------------------------------------------------------------------------------------
def test_a_rounded_mask_is_solid_inside_empty_at_the_corners_and_soft_on_the_edge():
    mask = ui.rounded_mask(100, 60, 20)
    assert mask.shape == (60, 100) and mask.dtype == np.uint8
    assert mask[30, 50] == 255 and mask[0, 0] == 0 and mask[59, 99] == 0
    assert mask[30, 0] == 255 and mask[0, 50] == 255                              # the straight edges are full
    assert len({int(v) for v in mask[0:20, 0:20].ravel()}) > 4                     # the corner is anti-aliased


def test_a_radius_of_nothing_is_a_plain_rectangle_and_one_too_big_is_a_pill():
    assert (ui.rounded_mask(50, 30, 0) == 255).all()
    pill = ui.rounded_mask(100, 40, 999)
    assert pill[20, 0] > 100 and pill[0, 0] == 0 and pill[20, 50] == 255


def test_a_vertical_gradient_runs_from_the_top_colour_to_the_bottom_one():
    g = ui.gradient(40, 10, (255, 0, 0), (0, 0, 255))
    assert g.shape == (40, 10, 3) and tuple(g[0, 0]) == (255, 0, 0) and tuple(g[-1, 0]) == (0, 0, 255)
    assert 100 < g[20, 0, 0] < 160


# --- panels ----------------------------------------------------------------------------------------------------------------
def test_a_panel_is_filled_inside_and_leaves_the_corners_alone():
    frame = blank()
    ui.panel(frame, 100, 80, 200, 120, radius=24, fill=WHITE, opacity=1.0, shadow=False)
    assert px(frame, 200, 140) == WHITE
    assert px(frame, 101, 81) == (60, 60, 60)                                      # the rounded corner


def test_a_panel_casts_a_soft_shadow_below_it_and_not_above_it():
    plain, shadowed = blank(200), blank(200)
    ui.panel(plain, 200, 100, 200, 100, fill=WHITE, opacity=1.0, shadow=False)
    ui.panel(shadowed, 200, 100, 200, 100, fill=WHITE, opacity=1.0, shadow=True)
    assert (shadowed[200:215, 250:350].mean() < plain[200:215, 250:350].mean() - 5)    # under the panel
    assert np.array_equal(shadowed[40:80, 250:350], plain[40:80, 250:350])             # nothing above it


def test_a_translucent_panel_shows_what_is_behind_it():
    frame = blank(0)
    ui.panel(frame, 100, 80, 200, 120, fill=WHITE, opacity=0.5, shadow=False)
    assert 110 <= frame[140, 200, 0] <= 145


def test_a_panels_border_is_drawn_in_its_own_colour_round_the_edge():
    frame = blank()
    ui.panel(frame, 100, 80, 200, 120, radius=20, fill=WHITE, opacity=1.0, border=NAVY, border_px=5, shadow=False)
    assert px(frame, 200, 82) == NAVY and px(frame, 200, 140) == WHITE


def test_a_panel_can_have_a_gradient_fill():
    frame = blank()
    ui.panel(frame, 100, 80, 200, 120, fill=((255, 255, 255), (255, 200, 120)), opacity=1.0, shadow=False)
    assert frame[90, 200, 0] == 255 and frame[190, 200, 2] < 200 < frame[190, 200, 0] + 200
    assert tuple(frame[90, 200]) != tuple(frame[190, 200])


def test_a_panel_partly_off_the_picture_is_cut_not_an_error():
    frame = blank()
    ui.panel(frame, -50, -30, 120, 90)
    ui.panel(frame, W - 40, H - 20, 200, 100)
    assert frame.shape == (H, W, 3)


# --- pill buttons ---------------------------------------------------------------------------------------------------------
def test_a_pill_button_is_centred_where_asked_and_says_what_it_does():
    frame = blank()
    ui.pill(frame, 320, 180, 240, 76, "PLAY", fill=ORANGE)
    assert px(frame, 320 - 100, 180 + 22) != (60, 60, 60) and px(frame, 320 - 118, 180 - 36) == (60, 60, 60)
    rows, cols = np.where(changed(frame, blank()))
    assert abs((cols.min() + cols.max()) / 2 - 320) < 8 and abs((rows.min() + rows.max()) / 2 - 180) < 14


def test_a_bigger_scale_makes_a_bigger_button_for_the_hover_pop():
    small, big = blank(), blank()
    ui.pill(small, 320, 180, 200, 70, "GO", fill=ORANGE, scale=1.0)
    ui.pill(big, 320, 180, 200, 70, "GO", fill=ORANGE, scale=1.25)
    assert changed(big, blank()).sum() > 1.3 * changed(small, blank()).sum()


def test_the_dwell_progress_fills_the_button_from_the_left():
    frame = blank()
    ui.pill(frame, 320, 180, 240, 76, "", fill=(255, 255, 255), fill_progress=(40, 200, 60), progress=0.5)
    assert px(frame, 320 - 90, 180 + 20) == (40, 200, 60) and px(frame, 320 + 90, 180 + 20) == (255, 255, 255)
    none = blank()
    ui.pill(none, 320, 180, 240, 76, "", fill=(255, 255, 255), fill_progress=(40, 200, 60), progress=0.0)
    assert px(none, 320 - 90, 180 + 20) == (255, 255, 255)


# --- rings, stars, bursts, glow ----------------------------------------------------------------------------------------------
def test_a_ring_fills_clockwise_from_the_top_as_its_progress_grows():
    frame = blank()
    ui.ring(frame, 320, 180, 40, 0.3, ORANGE, 8)
    assert px(frame, 320 + 40, 180) == ORANGE                                      # a bit over a quarter: the top to the right
    assert px(frame, 320 - 40, 180) == (60, 60, 60) and px(frame, 320, 180 + 40) == (60, 60, 60)
    full = blank()
    ui.ring(full, 320, 180, 40, 1.0, ORANGE, 8)
    assert all(px(full, 320 + dx, 180 + dy) == ORANGE for dx, dy in ((40, 0), (-40, 0), (0, 40), (0, -40)))
    assert np.array_equal(blank(), (lambda f: (ui.ring(f, 320, 180, 40, 0.0, ORANGE, 8), f)[1])(blank()))     # nothing at 0


def test_a_ring_can_have_a_track_behind_its_progress():
    frame = blank()
    ui.ring(frame, 320, 180, 40, 0.3, ORANGE, 8, track=(200, 200, 200))
    assert px(frame, 320 - 40, 180) == (200, 200, 200) and px(frame, 320 + 40, 180) == ORANGE


def test_a_star_is_solid_at_its_centre_and_stays_inside_its_radius():
    frame = blank()
    ui.star(frame, 320, 180, 40, ORANGE)
    assert px(frame, 320, 180) == ORANGE
    rows, cols = np.where(changed(frame, blank()))
    assert cols.min() >= 320 - 42 and cols.max() <= 320 + 42 and rows.max() <= 180 + 42


def test_a_burst_has_spikes_that_reach_further_than_its_body():
    frame = blank()
    ui.burst(frame, 320, 180, 80, 50, 12, ORANGE)
    assert px(frame, 320, 180) == ORANGE
    rows, cols = np.where(changed(frame, blank()))
    assert cols.max() - cols.min() >= 150 and cols.max() - cols.min() <= 168


def test_a_glow_is_brightest_in_the_middle_and_fades_to_nothing():
    frame = blank(0)
    ui.glow(frame, 320, 180, 60, (255, 255, 255), 0.8)
    assert frame[180, 320, 0] > frame[180, 320 + 30, 0] > frame[180, 320 + 55, 0] >= 0
    assert frame[180, 320 + 70, 0] == 0


def test_a_cursor_is_a_paddle_pointer_with_a_ring_that_fills_as_you_hold():
    idle, held = blank(), blank()
    ui.cursor(idle, 320, 180, progress=0.0)
    ui.cursor(held, 320, 180, progress=0.8)
    assert changed(idle, blank()).sum() > 400
    assert changed(held, idle).sum() > 300                                         # the ring is the difference


# --- confetti -----------------------------------------------------------------------------------------------------------------
def test_confetti_is_the_same_for_the_same_seed_and_moment_and_moves_over_time():
    a, b, later = blank(), blank(), blank()
    ui.confetti(a, 0.8, seed=3)
    ui.confetti(b, 0.8, seed=3)
    ui.confetti(later, 1.6, seed=3)
    assert np.array_equal(a, b) and not np.array_equal(a, later)
    assert changed(a, blank()).sum() > 300


def test_confetti_before_it_starts_is_nothing():
    frame = blank()
    ui.confetti(frame, 0.0, seed=1)
    assert np.array_equal(frame, blank())


def test_a_trophy_is_a_cup_on_a_foot_in_its_colour():
    frame = blank()
    ui.trophy(frame, 320, 180, 40, ORANGE)
    assert px(frame, 320, 160) == ORANGE and px(frame, 320, 180 + 25) == ORANGE         # the cup, the stem or foot
    assert px(frame, 320 + 80, 180) == (60, 60, 60)
    assert changed(frame, blank()).sum() > 1500


def test_things_still_growing_from_nothing_draw_nothing_and_do_not_fail():
    frame = blank()
    for scale in (0.0, 0.004, 0.02):
        ui.pill(frame, 320, 180, 240, 76, "GO", fill=ORANGE, scale=scale, progress=0.5, fill_progress=(40, 200, 60))
    ui.panel(frame, 100, 100, 0, 50)
    ui.panel(frame, 100, 100, 1, 1)
    assert np.array_equal(frame, blank())
    assert ui.rounded_mask(0, 10, 3).shape == (10, 0) and ui.rounded_mask(10, 0, 3).shape == (0, 10)
