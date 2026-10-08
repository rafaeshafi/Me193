"""Playing a friend: the list (a card to host a game with a chip for each speed, and a row for every game somebody has opened) and the
wait while a game is hosted or joined (its code on five tiles).  Every button is where menu_layout.py says, because that is where
flow.py presses it."""

import cv2

from pingpong import anim, cast, characters, fonts, levels, menu_layout, ui
from pingpong.screens_common import (CORAL, CORAL_DARK, GREEN, GREEN_DARK, HOVER_SCALE, NAVY, ORANGE, ORANGE_DARK, PALE, PURPLE, PURPLE_DARK,
                                     SKY, SKY_DARK, SLATE, WHITE, active, arrival, back_button, bubbles, button_scale, card, frosted, hint,
                                     hold_bar, pointer, progress_of, ribbon, size_of)
from pingpong.uistate import OnlineView

PACE_COLORS = {1: (GREEN, GREEN_DARK), 2: (SKY, SKY_DARK), 3: (CORAL, CORAL_DARK)}
NO_VIEW = OnlineView()


def _px(rect, size):
    w, h = size
    x0, y0, x1, y1 = rect
    return x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h


def _dots(frame, cx, cy, t_s, color=SLATE):
    """Three dots bobbing one after another: something is going on."""
    for i in range(3):
        cv2.circle(frame, (round(cx + (i - 1) * 34), round(cy + anim.bob(t_s + 0.18 * i, period_s=0.9, amplitude=9.0))), 8, color, -1, cv2.LINE_AA)


def _notice(frame, cx, cy, text):
    tw, _ = fonts.measure(text, 26)
    ui.pill(frame, cx, cy, tw + 70, 46, text, fill=(CORAL, CORAL_DARK), size=26, outline=WHITE, gloss=False)


# --- the list ---------------------------------------------------------------------------------------------------------------------------------
def online(frame, state):
    u, size = state.ui, size_of(frame)
    w, h = size
    view = u.online or NO_VIEW
    frosted(frame, state)
    bubbles(frame, u.t_s)
    ribbon(frame, w / 2, 96, "PLAY A FRIEND", (PURPLE, PURPLE_DARK), u.t_s)
    _host_card(frame, state, view)
    _games(frame, state, view)
    if view.message and view.status != "offline":
        _notice(frame, w / 2 + 120, 655, view.message)
    back_button(frame, u, menu_layout.BACK_BUTTON)
    hint(frame, "hold the hub on a speed to host   •   or on a game to join   •   ENTER  <  >  1-3   •   DELETE  back", cx=w / 2 + 120)
    pointer(frame, state)


def _chip(frame, u, size, tag, entrance):
    """One speed to host at: a tall chip with its name and as many balls as it is fast."""
    target = f"host{tag}"
    chosen = u.hover is None and u.focus == 0 and u.level_tag == tag
    on = u.hover == target or chosen
    scale = anim.pop(entrance) * (HOVER_SCALE if on else 1.0)
    x, y, cw, ch = _px(menu_layout.HOST_CHIPS[tag], size)
    cx, cy = x + cw / 2, y + ch / 2
    ui.panel(frame, cx - cw * scale / 2, cy - ch * scale / 2, cw * scale, ch * scale, radius=int(30 * scale), fill=PACE_COLORS[tag],
             border=ORANGE if on else WHITE, border_px=6 if on else 4)
    name = levels.LEVELS[tag].name.upper()
    fonts.draw(frame, name, cx, cy - 22 * scale, fonts.fit_size(name, cw * scale * 0.84, max_size=round(23 * scale), min_size=11), WHITE,
               outline=PACE_COLORS[tag][1], outline_px=2)
    for k in range(tag):                                                        # balls: the faster, the more
        bx = cx + (k - (tag - 1) / 2) * 24 * scale
        cv2.circle(frame, (round(bx), round(cy + 14 * scale)), round(8 * scale), WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (round(bx), round(cy + 14 * scale)), round(5 * scale), ORANGE, -1, cv2.LINE_AA)
    hold_bar(frame, cx, cy + ch * scale / 2 - 26 * scale, cw * 0.74 * scale, progress_of(u, target))


def _host_card(frame, state, view):
    u, size = state.ui, size_of(frame)
    entrance = arrival(u, 0.2, 0.5)
    on = (u.hover or "").startswith("host") or (u.hover is None and u.focus == 0)
    cx, cy, cw, ch = card(frame, menu_layout.HOST_CARD, anim.pop(entrance), on)
    top = cy - ch / 2
    me = cast.player_look(state.player_name or "player")
    characters.portrait(frame, cx, top + 100, 132, me, mood="grin" if on else "happy", t=state.anim_t)
    fonts.draw(frame, "HOST A GAME", cx, top + 196, fonts.fit_size("HOST A GAME", cw * 0.86, max_size=46, min_size=20), NAVY)
    fonts.draw(frame, "pick a speed, then wait for a friend", cx, top + 236, 20, SLATE)
    fonts.draw(frame, "MATCH SPEED", cx, menu_layout.HOST_CHIPS[1][1] * size[1] - 22, 16, SLATE)
    for k, tag in enumerate((1, 2, 3)):
        _chip(frame, u, size, tag, arrival(u, 0.4 + 0.1 * k, 0.4))


def _games(frame, state, view):
    u, size = state.ui, size_of(frame)
    w, h = size
    x0 = menu_layout.ROOM_ROWS[0][0] * w
    fonts.draw(frame, "OPEN GAMES", x0 + 8, 0.213 * h, 24, WHITE, anchor="lm", outline=NAVY, outline_px=3)
    rooms = view.rooms[:menu_layout.MAX_ROOMS]
    if not rooms:
        _empty(frame, u, view, size)
        return
    focus_target = f"join{max(0, u.focus - 1)}" if u.focus > 0 else None
    for k, room in enumerate(rooms):
        _row(frame, state, k, room, focus_target, arrival(u, 0.35 + 0.12 * k, 0.45))


def _empty(frame, u, view, size):
    w, h = size
    x, y, pw, ph = _px((menu_layout.ROOM_ROWS[0][0], menu_layout.ROOM_ROWS[0][1], menu_layout.ROOM_ROWS[0][2], menu_layout.ROOM_ROWS[2][3]), size)
    ui.panel(frame, x, y, pw, ph, radius=44, fill=WHITE, opacity=0.82)
    cx, cy = x + pw / 2, y + ph / 2
    if view.status == "offline":
        for i, line in enumerate(fonts.wrap_lines(view.message, pw - 90, 30, 3)):
            fonts.draw(frame, line, cx, cy - 20 + 40 * i, 30, ORANGE_DARK)
        return
    title = "CONNECTING..." if view.status == "connecting" else "NO OPEN GAMES YET"
    fonts.draw(frame, title, cx, cy - 56, 42, NAVY)
    if view.status != "connecting":
        fonts.draw(frame, "Host one, or wait here:", cx, cy, 24, SLATE)
        fonts.draw(frame, "games other players open show up by themselves.", cx, cy + 32, 24, SLATE)
    _dots(frame, cx, cy + 92, u.t_s)


def _row(frame, state, k, room, focus_target, entrance):
    u = state.ui
    target = f"join{k}"
    scale = button_scale(u, target, focus_target, entrance)
    cx, cy, cw, ch = card(frame, menu_layout.ROOM_ROWS[k], scale, active(u, target, focus_target))
    left, top = cx - cw / 2, cy - ch / 2
    characters.portrait(frame, left + 84 * scale, cy - 6 * scale, 96 * scale, cast.player_look(room["host"]), mood="happy", t=state.anim_t + k)
    fonts.draw(frame, room["host"], left + 152 * scale, cy - 26 * scale, fonts.fit_size(room["host"], 230 * scale, max_size=round(38 * scale),
                                                                                          min_size=12), NAVY, anchor="lm")
    fonts.draw(frame, f"MATCH   •   FIRST TO {room['target']}", left + 152 * scale, cy + 12 * scale, round(20 * scale), SLATE, anchor="lm")
    level = levels.LEVELS[room["pace"]]
    ui.pill(frame, left + cw - 112 * scale, cy - 12 * scale, 156 * scale, 42 * scale, level.name.upper(), fill=PACE_COLORS[room["pace"]],
            size=round(23 * scale), gloss=False, shadow=False)
    hold_bar(frame, cx, top + ch - 30 * scale, cw * 0.82, progress_of(u, target))


# --- the wait ----------------------------------------------------------------------------------------------------------------------------------
def _tiles(frame, u, code, cx, y):
    tw, th, gap = 112, 140, 14
    x0 = cx - (len(code) * tw + (len(code) - 1) * gap) / 2
    for i, letter in enumerate(code):
        pop = anim.pop(arrival(u, 0.25 + 0.1 * i, 0.4))
        tx = x0 + i * (tw + gap) + tw / 2
        ui.panel(frame, tx - tw * pop / 2, y + th / 2 - th * pop / 2, tw * pop, th * pop, radius=int(30 * pop), fill=(WHITE, PALE), border=PURPLE,
                 border_px=5)
        fonts.draw(frame, letter, tx, y + th / 2 + 4, round(104 * pop), NAVY)


def wait(frame, state):
    u, size = state.ui, size_of(frame)
    w, h = size
    view = u.online or NO_VIEW
    hosting = view.status == "hosting"
    frosted(frame, state)
    bubbles(frame, u.t_s)
    ribbon(frame, w / 2, 96, "WAITING FOR A FRIEND" if hosting else "JOINING...", (PURPLE, PURPLE_DARK), u.t_s)
    ui.panel(frame, w * 0.18, h * 0.25, w * 0.64, h * 0.55, radius=48, fill=WHITE, opacity=0.95)
    fonts.draw(frame, "GAME CODE" if hosting else "KNOCKING ON", w / 2, h * 0.25 + 46, 28, SLATE)
    _tiles(frame, u, view.code, w / 2, h * 0.25 + 84)
    pace = levels.LEVELS[min(3, max(1, view.pace))].name.upper()
    if hosting:
        name = cast.player_look(state.player_name or "player").name
        fonts.draw(frame, f"{name}   •   {pace} PACE   •   FIRST TO {view.target}", w / 2, h * 0.25 + 280, 30, NAVY)
        line = "Your game is on the list: your friend finds it under PLAY A FRIEND." if view.live else "Setting up your game..."
    else:
        host = next((r["host"] for r in view.rooms if r["code"] == view.code), "")
        fonts.draw(frame, f"{host}'s game" if host else "a game", w / 2, h * 0.25 + 280, 30, NAVY)
        line = "Waiting for their laptop to answer..."
    fonts.draw(frame, line, w / 2, h * 0.25 + 324, 22, SLATE)
    _dots(frame, w / 2, h * 0.25 + 376, u.t_s)
    back_button(frame, u, menu_layout.BACK_BUTTON, label="CANCEL", target="cancel")
    hint(frame, "hold the hub on CANCEL   •   or  DELETE  to go back", cx=w / 2 + 120)
    pointer(frame, state)
