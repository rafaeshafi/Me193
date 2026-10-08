"""The screens round the game: the intro, the title, the choice of game and of opponent, the face-off and the results.

render() draws the one named by the state's `screen` from what the Flow told it (`state.ui`: how long it has been up, where the
hand is, what it is holding on, how far the hold has come).  The title shows the court itself; the others float over it, frosted.
Every button is where menu_layout.py says, because that is where flow.py presses it.
"""

import dataclasses
import math

import cv2
import numpy as np

from pingpong import anim, canvas, cast, characters, court3d, fonts, holdstart, intro, menu_layout, scene, screens_end, screens_pick, ui
from pingpong.screens_common import (GOLD, NAVY, ORANGE, ORANGE_DARK, SKY, SKY_DARK, SLATE, WHITE, active, arrival, back_button, bubbles,
                                     button_scale, card, frosted, hint, hold_bar, pointer, progress_of, ribbon, size_of, start_button)


def render(frame, state, background=None):
    screen = state.screen
    if screen == "INTRO":
        intro.render(frame, state)
    elif screen == "TITLE":
        title(frame, state)
    elif screen == "MODE":
        mode(frame, state)
    elif screen == "OPPONENT":
        screens_pick.opponent(frame, state)
    elif screen == "VS":
        screens_end.vs(frame, state)
    elif screen == "RESULTS":
        screens_end.results(frame, state)
    else:
        raise ValueError(f"no screen called {screen!r}")


# --- the title ---------------------------------------------------------------------------------------------------------------------------
def _court(frame, state):
    """The live court behind the title (the opponent waiting, breathing), without a ball or a paddle."""
    w, h = size_of(frame)
    empty = dataclasses.replace(state, phase="LOBBY", ball=None, paddle=None, rest=None, zone=None, cpu_swing=None, cpu_mood="happy",
                                screen="GAME", ui=None, results=None)
    scene.draw_scene(frame, court3d.Camera.for_frame(w, h), empty)


def _chip(frame, x, y, text, good):
    tw, _ = fonts.measure(text, 19)
    ui.panel(frame, x, y, tw + 50, 34, radius=17, fill=WHITE, opacity=0.92, shadow=False)
    cv2.circle(frame, (x + 22, y + 17), 7, cast.rgb(70, 200, 110) if good else cast.rgb(255, 160, 40), -1, cv2.LINE_AA)
    fonts.draw(frame, text, x + 38, y + 18, 19, NAVY if good else cast.rgb(190, 100, 0), anchor="lm")


def _logo(frame, u, w):
    drop = (1.0 - anim.ease_out_back(arrival(u, 0.1, 0.9))) * -300 + anim.bob(u.t_s, period_s=3.0, amplitude=5.0)
    fonts.draw(frame, "PING-PONG", w / 2, 128 + drop, 110, WHITE, outline=NAVY, outline_px=12, shadow=(0, 9, NAVY, 0.30))
    ui.pill(frame, w / 2, 232 + drop, 320, 68, "ISLAND", fill=(ORANGE, ORANGE_DARK), size=44, outline=WHITE)
    height = abs(math.sin(u.t_s * 3.2))                                  # a ball bouncing beside the name, with its shadow
    bx, floor = round(w / 2 - 392), 250 + drop
    cv2.ellipse(frame, (bx, round(floor)), (round(26 * (1 - 0.5 * height)), 7), 0, 0, 360, (150, 120, 90), -1, cv2.LINE_AA)
    cv2.circle(frame, (bx, round(floor - 28 - 70 * height)), 25, scene.BALL_RIM, -1, cv2.LINE_AA)
    cv2.circle(frame, (bx, round(floor - 28 - 70 * height)), 22, scene.BALL, -1, cv2.LINE_AA)


def title(frame, state):
    """The court the intro landed on, then (over its first second) the name, the START button, who is playing and the state of the
    hub and the broker come in: nothing is there at the first instant, so the cut from the intro is invisible."""
    u, (w, h) = state.ui, size_of(frame)
    _court(frame, state)
    scrim = (np.linspace(0.62, 0.0, 300, dtype=np.float32) * 255 * arrival(u, 0.0, 0.6)).astype(np.uint8)
    fonts.blit(frame, WHITE, np.repeat(scrim[:, None], w, axis=1), 0, 0)
    _logo(frame, u, w)
    start_button(frame, u, "START", holdstart.BUTTON, entrance=arrival(u, 0.7, 0.5))
    rise = (1.0 - anim.ease_out_cubic(arrival(u, 0.3, 0.6))) * 150                  # the badge and the chips slide up into place
    me = cast.player_look(state.player_name or "player")
    ui.panel(frame, 24, h - 112 + rise, 330, 84, radius=42, fill=WHITE, opacity=0.95)
    characters.portrait(frame, 72, h - 70 + rise, 64, me, mood="happy", t=state.anim_t)
    fonts.draw(frame, "PLAYER", 120, h - 85 + rise, 18, SLATE, anchor="lm")
    fonts.draw(frame, me.name, 120, h - 56 + rise, 34, NAVY, anchor="lm")
    _chip(frame, w - 190, round(h - 64 + rise), f"HUB {state.hub_status.upper()}", state.hub_status == "ok")
    _chip(frame, w - 190, round(h - 104 + rise), f"MQTT {state.mqtt_status.upper()}", state.mqtt_status == "ok")
    if state.message and arrival(u, 0.5, 0.4) > 0:
        lines = fonts.wrap_lines(state.message, w - 780, 26)
        ui.panel(frame, 380, h - 140, w - 760, 30 + 34 * len(lines), radius=28, fill=WHITE, opacity=0.95, border=ORANGE, border_px=4)
        for i, line in enumerate(lines):
            fonts.draw(frame, line, w / 2, h - 113 + 34 * i, 26, cast.rgb(150, 90, 0), anchor="mm")
    hint(frame, "ENTER  menus     \u2022     SPACE  quick start     \u2022     Q  quit", opacity=0.92 * arrival(u, 0.5, 0.5))
    pointer(frame, state)


# --- the choice of game ------------------------------------------------------------------------------------------------------------------
def _rally_icon(frame, cx, cy):
    """Two paddles and a ball in a high arc between them."""
    canvas.draw_paddle(frame, round(cx - 92), round(cy + 14), 26, angle_deg=-22.0)
    canvas.draw_paddle(frame, round(cx + 92), round(cy + 14), 26, rubber=canvas.RUBBER_BLACK, angle_deg=22.0)
    for k in range(1, 8):
        t = k / 8
        if k != 4:
            cv2.circle(frame, (round(cx - 72 + 144 * t), round(cy + 4 - 80 * 4 * t * (1 - t))), 3, ORANGE_DARK, -1, cv2.LINE_AA)
    cv2.circle(frame, (round(cx), round(cy - 76)), 16, scene.BALL_RIM, -1, cv2.LINE_AA)
    cv2.circle(frame, (round(cx), round(cy - 76)), 13, scene.BALL, -1, cv2.LINE_AA)


def _match_icon(frame, cx, cy):
    ui.trophy(frame, cx, cy - 2, 56, GOLD)
    for dx, dy, r in ((-96, -46, 13), (98, -56, 10), (82, 36, 8)):
        ui.star(frame, cx + dx, cy + dy, r, GOLD, outline=ORANGE_DARK, outline_px=2)
    fonts.draw(frame, "7", cx, cy - 14, 40, ORANGE_DARK)


def _game_card(frame, u, name, rect, focus_target, entrance):
    on = active(u, name, focus_target)
    scale = button_scale(u, name, focus_target, entrance)
    cx, cy, cw, ch = card(frame, rect, scale, on)
    top = cy - ch / 2
    rally = name == "rally"
    (_rally_icon if rally else _match_icon)(frame, cx, top + 112 * scale)
    fonts.draw(frame, "RALLY" if rally else "MATCH", cx, top + 240 * scale, round(66 * scale), NAVY)
    lines = (("Keep it going!", "The computer never misses.", "How long can you rally?") if rally
             else ("First to 7 points wins.", "Hit it hard, wide and spinny", "and the computer will miss."))
    for i, line in enumerate(lines):
        fonts.draw(frame, line, cx, top + (298 + 31 * i) * scale, round(24 * scale), NAVY if i == 0 else SLATE)
    hold_bar(frame, cx, top + ch - 40 * scale, cw * 0.6, progress_of(u, name))


def mode(frame, state):
    u, (w, h) = state.ui, size_of(frame)
    frosted(frame, state)
    bubbles(frame, u.t_s)
    ribbon(frame, w / 2, 104, "CHOOSE YOUR GAME", (SKY, SKY_DARK), u.t_s)
    focus = "rally" if u.focus == 0 else "match"
    for k, (name, rect) in enumerate(menu_layout.MODE_CARDS.items()):
        _game_card(frame, u, name, rect, focus, arrival(u, 0.25 + 0.15 * k, 0.5))
    back_button(frame, u, menu_layout.BACK_BUTTON)
    hint(frame, "hold the hub on a card   •   or  <  >  ENTER   •   DELETE  back", cx=w / 2 + 120)
    pointer(frame, state)
