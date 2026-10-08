"""The choice of opponent: a card for each of the three, with a portrait, a name, a level, three rows of stars and a line it likes to
say.  The one the hand (or the keys) is on pops up and grins; holding on it for a moment chooses it."""

import cv2
import numpy as np

from pingpong import cast, characters, fonts, levels, menu_layout, ui
from pingpong.screens_common import (CORAL, CORAL_DARK, GOLD, GREEN, GREEN_DARK, NAVY, ORANGE_DARK, PALE, SKY, SKY_DARK, SLATE, WHITE, active,
                                     arrival, back_button, bubbles, button_scale, card, frosted, hint, hold_bar, pointer, progress_of, ribbon,
                                     size_of)
from pingpong.cast import rgb

PORTRAIT_BG = {1: (rgb(186, 236, 255), rgb(232, 249, 255)), 2: (rgb(255, 208, 226), rgb(255, 238, 245)),
               3: (rgb(200, 212, 255), rgb(234, 238, 255))}
LEVEL_COLORS = {1: (GREEN, GREEN_DARK), 2: (SKY, SKY_DARK), 3: (CORAL, CORAL_DARK)}
STAT_ROWS = (("SPEED", "speed"), ("DEFENSE", "reach"), ("SPIN", "spin"))
EMPTY_STAR = rgb(208, 218, 234)


def _stars(frame, x, y, lit, r):
    for k in range(5):
        if k < lit:
            ui.star(frame, x + k * 2.5 * r, y, r, GOLD, outline=ORANGE_DARK, outline_px=max(1, round(r / 5)))
        else:
            ui.star(frame, x + k * 2.5 * r, y, r, EMPTY_STAR)


def _bubble(frame, x, y, scale, text):
    """A speech bubble at the top right of a card, with a tail pointing down to the face."""
    lines = fonts.wrap_lines(text, 150 * scale, round(18 * scale), 2)
    bw, bh = 168 * scale, (20 + 23 * len(lines)) * scale
    cv2.fillPoly(frame, [np.array([(x + 24 * scale, y + bh - 4), (x + 56 * scale, y + bh - 4), (x + 22 * scale, y + bh + 20 * scale)],
                                  dtype=np.int32)], PALE, cv2.LINE_AA)
    ui.panel(frame, x, y, bw, bh, radius=int(20 * scale), fill=WHITE, opacity=1.0, border=PALE, border_px=3, shadow=False)
    for i, line in enumerate(lines):
        fonts.draw(frame, line, x + bw / 2, y + (22 + 23 * i) * scale, round(18 * scale), NAVY)


def _opponent_card(frame, state, tag, rect, focus_target, entrance):
    u = state.ui
    target = f"opp{tag}"
    look, level = cast.OPPONENTS[tag], levels.LEVELS[tag]
    on = active(u, target, focus_target)
    scale = button_scale(u, target, focus_target, entrance)
    cx, cy, cw, ch = card(frame, rect, scale, on)
    top, left = cy - ch / 2, cx - cw / 2
    characters.portrait(frame, cx, top + 118 * scale, 170 * scale, look, mood="grin" if on else "happy", t=state.anim_t + 0.7 * tag,
                        bg=PORTRAIT_BG[tag])
    _bubble(frame, cx + cw / 2 - 182 * scale, top + 12 * scale, scale, look.tagline)
    fonts.draw(frame, look.name, cx, top + 236 * scale, round(52 * scale), NAVY)
    ui.pill(frame, cx, top + 282 * scale, 150 * scale, 36 * scale, level.name.upper(), fill=LEVEL_COLORS[tag], size=round(21 * scale),
            gloss=False, shadow=False)
    for row, (label, key) in enumerate(STAT_ROWS):
        y = top + (326 + 28 * row) * scale
        fonts.draw(frame, label, left + 34 * scale, y, round(19 * scale), SLATE, anchor="lm")
        _stars(frame, left + cw - 34 * scale - 4 * 2.5 * 9 * scale - 9 * scale, y, cast.stars(level)[key], 9 * scale)
    hold_bar(frame, cx, top + ch - 42 * scale, cw * 0.62, progress_of(u, target))


def opponent(frame, state):
    u, (w, h) = state.ui, size_of(frame)
    frosted(frame, state)
    bubbles(frame, u.t_s)
    ribbon(frame, w / 2, 96, "CHOOSE YOUR OPPONENT", (CORAL, CORAL_DARK), u.t_s)
    game = "MATCH   •   FIRST TO 7" if u.mode == "match" else "RALLY"
    tw, _ = fonts.measure(game, 22)
    ui.panel(frame, 24, 24, tw + 50, 42, radius=21, fill=WHITE, opacity=0.95, shadow=False)
    fonts.draw(frame, game, 49, 46, 22, NAVY, anchor="lm")
    focus = f"opp{u.focus + 1}"
    for k, tag in enumerate((1, 2, 3)):
        _opponent_card(frame, state, tag, menu_layout.OPPONENT_CARDS[tag], focus, arrival(u, 0.2 + 0.12 * k, 0.5))
    back_button(frame, u, menu_layout.BACK_BUTTON)
    hint(frame, "hold the hub on an opponent   •   or  <  >  ENTER   •   a LEVEL card picks   •   DELETE  back", cx=w / 2 + 130)
    pointer(frame, state)
