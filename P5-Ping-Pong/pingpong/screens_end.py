"""The face-off before a game (you on one side, your opponent on the other, a big VS between you) and the results after it (who won,
the numbers, paper falling for a win or a record, and two buttons: play again, change opponent)."""

import math

import cv2
import numpy as np

from pingpong import anim, cast, characters, flow, fonts, holdstart, levels, ui
from pingpong.cast import rgb
from pingpong.uistate import Results
from pingpong.screens_common import (CORAL, CORAL_DARK, GOLD, GREEN, GREEN_DARK, NAVY, ORANGE, ORANGE_DARK, PURPLE, PURPLE_DARK, SKY, SKY_DARK,
                                     SLATE, WHITE, arrival, bubbles, frosted, pointer, ribbon, size_of, start_button)

LEVEL_PILL = {1: (GREEN, GREEN_DARK), 2: (SKY, SKY_DARK), 3: (CORAL, CORAL_DARK)}
CHANGE_BUTTON = (1.0 - holdstart.BUTTON[0] - holdstart.BUTTON[2], holdstart.BUTTON[1], holdstart.BUTTON[2], holdstart.BUTTON[3])


# --- the face-off -------------------------------------------------------------------------------------------------------------------------
def _split(frame):
    """Sky blue on the left, coral on the right, a white slash between them."""
    w, h = size_of(frame)
    frame[:] = ui.gradient(h, w, rgb(160, 220, 255), rgb(70, 150, 240))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([(0.58 * w, 0), (w, 0), (w, h), (0.42 * w, h)], dtype=np.int32)], 255, cv2.LINE_AA)
    right = ui.gradient(h, w, rgb(255, 196, 170), rgb(240, 96, 110))
    frame[mask > 0] = right[mask > 0]
    cv2.line(frame, (round(0.58 * w) + 9, 0), (round(0.42 * w) + 9, h), (0, 0, 0), 22, cv2.LINE_AA)
    cv2.line(frame, (round(0.58 * w), 0), (round(0.42 * w), h), WHITE, 18, cv2.LINE_AA)


def vs(frame, state):
    u, (w, h) = state.ui, size_of(frame)
    me = cast.player_look(state.player_name or "player")
    opp = cast.player_look(u.opponent) if u.opponent else cast.OPPONENTS[u.level_tag]        # a friend online, else the level's opponent
    level = levels.LEVELS[u.level_tag]
    _split(frame)
    slide = (1.0 - anim.ease_out_back(anim.progress(u.t_s, start=0.0, length=0.7))) * 720
    characters.draw_bust(frame, 0.26 * w - slide, 400, 470, me, mood="grin", t=state.anim_t)
    characters.draw_bust(frame, 0.74 * w + slide, 400, 470, opp, mood="smug", t=state.anim_t + 1.1)
    ui.pill(frame, 0.26 * w - slide, 664, 330, 70, me.name, fill=(SKY, SKY_DARK), size=44, outline=WHITE)
    ui.pill(frame, 0.74 * w + slide, 664, 330, 70, opp.name, fill=(CORAL, CORAL_DARK), size=44, outline=WHITE)
    ui.pill(frame, 0.74 * w + slide, 600, 170, 38, f"{level.name.upper()} PACE" if u.opponent else level.name.upper(), fill=LEVEL_PILL[u.level_tag],
            size=22, gloss=False, shadow=False)
    pop = anim.pop(anim.progress(u.t_s, start=0.45, length=0.45))
    shake = 7 * math.sin(u.t_s * 90) * max(0.0, 1.0 - (u.t_s - 0.45) / 0.5) if u.t_s > 0.45 else 0.0
    if pop > 0:
        ui.burst(frame, w / 2 + shake, 340, 150 * pop, 104 * pop, 14, GOLD, angle_deg=u.t_s * 18, outline=ORANGE_DARK, outline_px=5)
        fonts.draw(frame, "VS", w / 2 + shake, 346, round(124 * pop), WHITE, outline=NAVY, outline_px=11, shadow=(0, 8, NAVY, 0.3))
    game = (f"ONLINE MATCH   •   FIRST TO {state.target}" if u.opponent else
            f"MATCH   •   FIRST TO {state.target}" if u.mode == "match" else "RALLY   •   HOW LONG CAN YOU KEEP IT GOING?")
    tw, _ = fonts.measure(game, 30)
    ui.panel(frame, w / 2 - tw / 2 - 32, 38, tw + 64, 58, radius=29, fill=WHITE, opacity=0.95)
    fonts.draw(frame, game, w / 2, 68, 30, NAVY)
    flash = anim.clamp01((u.t_s - (flow.VS_S - 0.25)) / 0.25)
    if flash > 0:
        cv2.addWeighted(np.full_like(frame, 255), 0.95 * flash, frame, 1.0 - 0.95 * flash, 0, frame)


# --- the results -----------------------------------------------------------------------------------------------------------------------------
def _stats_card(frame, stats, x, y, width):
    ui.panel(frame, x, y, width, 150, radius=36, fill=WHITE, opacity=1.0)
    for i, (label, value) in enumerate(stats[:4]):
        cx = x + width * (i + 0.5) / max(1, min(4, len(stats)))
        fonts.draw(frame, label, cx, y + 44, 21, SLATE)
        fonts.draw(frame, value, cx, y + 98, round(fonts.fit_size(value, width / 4 - 20, max_size=50, min_size=24)), NAVY)


def _board(frame, state, x, y):
    rows = state.leaderboard[:5]
    if not rows:
        return
    ui.panel(frame, x, y, 400, 66 + 36 * len(rows), radius=30, fill=WHITE, opacity=1.0)
    fonts.draw(frame, "BEST STREAKS" if state.mode == "survival" else "MATCH WINS", x + 24, y + 34, 24, ORANGE_DARK, anchor="lm")
    for i, (name, score) in enumerate(rows):
        mine = bool(state.player_name) and name.lower() == state.player_name.lower()
        color = GREEN_DARK if mine else NAVY if i == 0 else SLATE
        yy = y + 76 + 36 * i
        if mine:
            ui.panel(frame, x + 12, yy - 17, 376, 34, radius=17, fill=rgb(222, 247, 232), opacity=1.0, shadow=False)
        fonts.draw(frame, f"{i + 1}. {name}", x + 26, yy, 25, color, anchor="lm")
        fonts.draw(frame, str(score), x + 374, yy, 25, color, anchor="rm")


def _result_colors(results):
    if results.new_record:
        return PURPLE, PURPLE_DARK
    if results.won is True:
        return GREEN, GREEN_DARK
    if results.won is False:
        return rgb(120, 140, 192), rgb(78, 98, 150)
    return SKY, SKY_DARK


def results(frame, state):
    u, r, (w, h) = state.ui, state.results or Results(), size_of(frame)
    me, opp = cast.player_look(state.player_name or "player"), cast.opponent_look(state.level_name, state.opponent_name)
    friend = bool(u.opponent)
    frosted(frame, state)
    bubbles(frame, u.t_s)
    celebrate = bool(r.won) or r.new_record
    if celebrate:
        ui.burst(frame, w / 2, 112, 340 * anim.pop(arrival(u, 0.0, 0.7)), 250 * anim.pop(arrival(u, 0.0, 0.7)), 18, rgb(255, 232, 150),
                 angle_deg=u.t_s * 8)
    if celebrate:
        ui.confetti(frame, u.t_s % 5.0, seed=7)                                    # behind the cards, over the court
    ribbon(frame, w / 2, 112, r.title, _result_colors(r), u.t_s, size=70)
    mood = "cheer" if celebrate else "sad" if r.won is False else "happy"
    slide = (1.0 - anim.ease_out_back(arrival(u, 0.2, 0.7)))
    characters.portrait(frame, 0.24 * w - 500 * slide, 330, 230, me, mood=mood, t=state.anim_t, bg=(rgb(186, 236, 255), rgb(232, 249, 255)))
    if state.mode == "match":
        characters.portrait(frame, 0.76 * w + 500 * slide, 330, 230, opp, mood="sad" if r.won else "cheer" if r.won is False else "happy",
                            t=state.anim_t + 1.3, bg=(rgb(255, 208, 226), rgb(255, 238, 245)))
        fonts.draw(frame, f"{state.player_points}  -  {state.cpu_points}", w / 2, 330, round(130 * anim.pop(arrival(u, 0.5, 0.5))), WHITE,
                   outline=NAVY, outline_px=11, shadow=(0, 9, NAVY, 0.3))
        fonts.draw(frame, f"{me.name}  vs  {opp.name}", w / 2, 424, 28, NAVY)
    else:
        ui.trophy(frame, 0.76 * w + 500 * slide, 330, 96, GOLD)
        fonts.draw(frame, str(state.streak), w / 2, 318, round(150 * anim.pop(arrival(u, 0.5, 0.5))), WHITE, outline=NAVY, outline_px=11,
                   shadow=(0, 9, NAVY, 0.3))
        fonts.draw(frame, f"STREAK      BEST {max(state.record, state.streak)}", w / 2, 424, 30, NAVY)
    _stats_card(frame, r.stats, 255 if friend else 40, 530, 770)
    if friend:
        _friend_buttons(frame, u)
    else:
        _board(frame, state, 840, 480)
        start_button(frame, u, "PLAY AGAIN", holdstart.BUTTON, target="again", entrance=arrival(u, 0.6, 0.5))
        start_button(frame, u, "CHANGE", CHANGE_BUTTON, target="change", entrance=arrival(u, 0.7, 0.5), colors=(SKY, SKY_DARK),
                     second="OPPONENT")
    pointer(frame, state)


def _friend_buttons(frame, u):
    """After a game with a friend: a rematch (which waits for them to say yes, and is not there if they have gone) and leaving."""
    if u.opponent_gone:
        ui.pill(frame, holdstart.BUTTON[0] * 1280 + holdstart.BUTTON[2] * 640, 120, 300, 54, f"{u.opponent} LEFT", fill=(CORAL, CORAL_DARK), size=26,
                gloss=False, outline=WHITE, scale=anim.pop(arrival(u, 0.6, 0.5)))
    elif u.rematch_pending:
        start_button(frame, u, "WAITING...", holdstart.BUTTON, target="again", entrance=arrival(u, 0.6, 0.5),
                     colors=(rgb(190, 200, 215), rgb(150, 162, 182)), hint_text=f"for {u.opponent}")
    else:
        start_button(frame, u, "REMATCH", holdstart.BUTTON, target="again", entrance=arrival(u, 0.6, 0.5))
    start_button(frame, u, "LEAVE", CHANGE_BUTTON, target="change", entrance=arrival(u, 0.7, 0.5), colors=(SKY, SKY_DARK))
