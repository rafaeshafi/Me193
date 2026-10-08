"""What the screens round the game share: the colours, the frosted court behind the menus, drifting bubbles, the title ribbon,
the pointer and how a button pops when the hand is on it."""

import math
import random
from functools import lru_cache

import cv2
import numpy as np

from pingpong import anim, court3d, fonts, scene, ui
from pingpong.cast import rgb

NAVY, SLATE, WHITE = rgb(24, 48, 96), rgb(112, 126, 150), (255, 255, 255)
SKY, SKY_DARK = rgb(70, 170, 255), rgb(30, 120, 225)
ORANGE, ORANGE_DARK = rgb(255, 168, 40), rgb(240, 120, 20)
GREEN, GREEN_DARK = rgb(90, 214, 130), rgb(40, 168, 92)
CORAL, CORAL_DARK = rgb(255, 124, 110), rgb(235, 76, 84)
GOLD, PURPLE, PURPLE_DARK = rgb(255, 196, 40), rgb(160, 110, 245), rgb(112, 70, 205)
PALE = rgb(214, 224, 240)
HOVER_SCALE = 1.07


def size_of(frame):
    return frame.shape[1], frame.shape[0]


@lru_cache(maxsize=4)
def _frost(w, h, level_name, state_class):
    """The empty court, blurred and brightened: the soft background of the menus (made once)."""
    layer = np.empty((h, w, 3), dtype=np.uint8)
    scene.draw_scene(layer, court3d.Camera.for_frame(w, h), state_class(level_name=level_name))
    blurred = cv2.GaussianBlur(layer, (0, 0), 7 * w / 1280)
    return cv2.addWeighted(blurred, 0.72, np.full_like(blurred, 255), 0.28, 0)


def frosted(frame, state):
    w, h = size_of(frame)
    frame[:] = _frost(w, h, state.level_name, type(state))


def bubbles(frame, t_s, *, seed=3, n=14):
    """Soft round lights drifting up the picture."""
    w, h = size_of(frame)
    rnd = random.Random(seed)
    for _ in range(n):
        x, r, speed, phase = rnd.uniform(0, w), rnd.uniform(14, 48) * w / 1280, rnd.uniform(14, 40), rnd.uniform(0, 600)
        y = h + r - (phase + speed * t_s) % (h + 2 * r)
        ui.glow(frame, x + 16 * math.sin(0.7 * t_s + phase), y, r, WHITE, 0.32)


def arrival(u, delay=0.0, length=0.6):
    """0..1 as a screen's things come in one after another from `delay` seconds."""
    return anim.progress(u.t_s, start=delay, length=length)


def active(u, target, focus_target):
    """The hand is on this button, or (with the hand elsewhere) it is the one the keys have in focus."""
    return u.hover == target or (u.hover is None and focus_target == target)


def button_scale(u, target, focus_target, entrance=1.0):
    """How big a button is: popping in as the screen comes up, bigger while the hand (or the keys) are on it."""
    return anim.pop(entrance) * (HOVER_SCALE if active(u, target, focus_target) else 1.0)


def progress_of(u, target):
    return u.progress if u.hover == target else 0.0


def pointer(frame, state):
    """The paddle-shaped pointer where the hand is over the screen, its ring filling as a hold goes on."""
    u = state.ui
    if u.cursor is None:
        return
    w, h = size_of(frame)
    ui.cursor(frame, min(1.0, max(0.0, u.cursor[0])) * (w - 1), (1.0 - min(1.0, max(0.0, u.cursor[1]))) * (h - 1),
              progress=u.progress if u.hover is not None else 0.0)


def ribbon(frame, cx, cy, text, colors, t_s, *, size=52, delay=0.0):
    """A banner with the title of a screen, popping in."""
    s = anim.pop(anim.progress(t_s, start=delay, length=0.6))
    width = fonts.measure(text, size)[0] + 3.4 * size
    for side in (-1, 1):                                              # the folded tails behind the ends
        x = cx + side * (width * s / 2 - 8)
        cv2.fillPoly(frame, [np.array([(x, cy - 0.3 * size * s), (x + side * 0.9 * size * s, cy - 0.3 * size * s),
                                       (x + side * 0.55 * size * s, cy + 0.3 * size * s), (x + side * 0.9 * size * s, cy + 0.9 * size * s),
                                       (x, cy + 0.7 * size * s)], dtype=np.int32)], colors[1], cv2.LINE_AA)
    ui.pill(frame, cx, cy, width, 1.7 * size, text, fill=colors, size=size, outline=WHITE, scale=max(0.01, s), gloss=True)


def hint(frame, text, *, cx=None, opacity=0.92):
    w, h = size_of(frame)
    fonts.draw(frame, text, w // 2 if cx is None else cx, h - 26, 21, WHITE, outline=NAVY, outline_px=3, opacity=opacity)


FILL = rgb(255, 214, 70)                       # what the hold on a button fills it with


def start_button(frame, u, label, rect, *, target="start", entrance=1.0, colors=(GREEN, GREEN_DARK), hint_text="hold the hub here", second=None):
    """A big pill button in a corner (the START button's rectangle and its mirror), with a hint under it; `second` is a smaller
    second line of its label."""
    w, h = size_of(frame)
    x, y, bw, bh = (f * v for f, v in zip(rect, (w, h, w, h)))
    scale = button_scale(u, target, None, entrance)
    cx, cy = x + bw / 2, y + bh * 0.4
    big = fonts.fit_size(label, bw * 0.82, max_size=round(bh * 0.27), min_size=14)
    ui.pill(frame, cx, cy, bw, bh * 0.62, "" if second else label, fill=colors, fill_progress=FILL, progress=progress_of(u, target),
            outline=WHITE, scale=scale, size=big)
    if second:
        fonts.draw(frame, label, cx, cy - bh * 0.10 * scale, round(big * scale), WHITE, anchor="mm", opacity=min(1.0, entrance))
        fonts.draw(frame, second, cx, cy + bh * 0.13 * scale, round(big * 0.74 * scale), WHITE, anchor="mm", opacity=min(1.0, entrance))
    fonts.draw(frame, hint_text, x + bw / 2, y + bh * 0.86, 19, WHITE, outline=NAVY, outline_px=3, opacity=min(1.0, entrance))


def card(frame, rect, scale, on):
    """A big white choice card standing in a menu's rectangle, popped to `scale`; an orange rim when it is the one in focus."""
    w, h = size_of(frame)
    x0, y0, x1, y1 = rect
    cx, cy, cw, ch = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h, (x1 - x0) * w * scale, (y1 - y0) * h * scale
    ui.panel(frame, cx - cw / 2, cy - ch / 2, cw, ch, radius=int(40 * scale), fill=WHITE, opacity=1.0,
             border=ORANGE if on else PALE, border_px=8 if on else 4)
    return cx, cy, cw, ch


def hold_bar(frame, cx, y, width, progress):
    """The bar along the bottom of a card that fills as the hand holds on it."""
    ui.panel(frame, cx - width / 2, y, width, 16, radius=8, fill=PALE, opacity=1.0, shadow=False)
    if progress > 0:
        ui.panel(frame, cx - width / 2, y, max(16.0, width * progress), 16, radius=8, fill=(ORANGE, ORANGE_DARK), opacity=1.0, shadow=False)


def back_button(frame, u, rect, *, label="BACK", target="back"):
    w, h = size_of(frame)
    x0, y0, x1, y1 = rect
    ui.pill(frame, (x0 + x1) / 2 * w, (y0 + y1) / 2 * h, (x1 - x0) * w, (y1 - y0) * h, label, fill=(rgb(190, 200, 215), rgb(150, 162, 182)),
            fill_progress=ORANGE, progress=progress_of(u, target), outline=WHITE, scale=button_scale(u, target, None, 1.0), size=30)
