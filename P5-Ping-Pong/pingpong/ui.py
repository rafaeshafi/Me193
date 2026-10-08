"""The shapes the menus and the HUD are made of: soft-shadowed panels, glossy pill buttons, progress rings, stars, bursts,
glow, the paddle-shaped pointer and confetti.  All of it draws into a numpy BGR frame, with anti-aliased edges (the masks
are drawn at four times the size and shrunk, once, then cached), in the clean bright style of a console sports game.
"""

import math
import random
from functools import lru_cache

import cv2
import numpy as np

from pingpong import canvas, fonts

WHITE = (255, 255, 255)
SUPERSAMPLE = 4
SHADOW_BLUR, SHADOW_DY, SHADOW_OPACITY = 7, 8, 0.30
CONFETTI = ((60, 90, 255), (60, 200, 255), (110, 220, 90), (255, 170, 60), (255, 90, 190), (255, 220, 120))


@lru_cache(maxsize=512)
def rounded_mask(w, h, radius):
    """A w x h rectangle with rounded corners as an anti-aliased 0..255 mask (shared: do not write to it)."""
    ss = SUPERSAMPLE
    r = max(0, min(int(radius), w // 2, h // 2))
    big = np.zeros((h * ss, w * ss), dtype=np.uint8)
    R, right, bottom = r * ss, w * ss - 1, h * ss - 1
    cv2.rectangle(big, (R, 0), (right - R, bottom), 255, -1)
    cv2.rectangle(big, (0, R), (right, bottom - R), 255, -1)
    for cx, cy in ((R, R), (right - R, R), (R, bottom - R), (right - R, bottom - R)):
        cv2.circle(big, (cx, cy), R, 255, -1)
    mask = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    mask.setflags(write=False)
    return mask


@lru_cache(maxsize=64)
def gradient(h, w, top, bottom):
    """A vertical gradient picture from the top colour to the bottom one."""
    ramp = np.linspace(top, bottom, h)[:, None, :]
    picture = np.repeat(ramp, w, axis=1).round().astype(np.uint8)
    picture.setflags(write=False)
    return picture


@lru_cache(maxsize=256)
def _shadow(w, h, radius):
    """-> (mask, pad): the panel's shape blurred, padded so the blur has room."""
    pad = 4 * SHADOW_BLUR
    big = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    big[pad:pad + h, pad:pad + w] = rounded_mask(w, h, radius)
    big = cv2.GaussianBlur(big, (0, 0), SHADOW_BLUR)
    big.setflags(write=False)
    return big, pad


def _fill(fill, h, w):
    """A flat colour stays a colour; a (top, bottom) pair of colours becomes a gradient picture."""
    if isinstance(fill[0], (tuple, list)):
        return gradient(h, w, tuple(fill[0]), tuple(fill[1]))
    return tuple(fill)


def panel(frame, x, y, w, h, *, radius=22, fill=WHITE, opacity=0.94, border=None, border_px=0, shadow=True):
    """A rounded panel with a soft shadow under it; fill is a colour or a (top, bottom) pair; the border is inside its edge."""
    x, y, w, h = round(x), round(y), round(w), round(h)
    if w < 2 or h < 2:
        return
    mask = rounded_mask(w, h, radius)
    if shadow:
        blurred, pad = _shadow(w, h, radius)
        fonts.blit(frame, (0, 0, 0), blurred, x - pad, y - pad + SHADOW_DY, SHADOW_OPACITY)
    bp = border_px if border and border_px else 0
    if bp and w > 2 * bp + 2 and h > 2 * bp + 2:
        inner = rounded_mask(w - 2 * bp, h - 2 * bp, max(0, radius - bp))
        ring = mask.astype(np.int16)
        ring[bp:h - bp, bp:w - bp] -= inner
        fonts.blit(frame, tuple(border), np.clip(ring, 0, 255).astype(np.uint8), x, y)
        fonts.blit(frame, _fill(fill, h - 2 * bp, w - 2 * bp), inner, x + bp, y + bp, opacity)
    else:
        fonts.blit(frame, _fill(fill, h, w), mask, x, y, opacity)


def pill(frame, cx, cy, w, h, label="", *, fill, text=WHITE, size=None, scale=1.0, progress=0.0, fill_progress=None,
         outline=None, shadow=True, opacity=1.0, gloss=True):
    """A button as long as it is round, centred on (cx, cy); `scale` pops it bigger; `progress` (0..1) sweeps
    `fill_progress` across it from the left, the dwell filling it up."""
    bw, bh = round(w * scale), round(h * scale)
    x0, y0 = round(cx - bw / 2), round(cy - bh / 2)
    panel(frame, x0, y0, bw, bh, radius=bh // 2, fill=fill, opacity=opacity, border=outline, border_px=max(2, bh // 18),
          shadow=shadow)
    if fill_progress is not None and progress > 0:
        swept = rounded_mask(bw, bh, bh // 2).copy()
        swept[:, round(bw * min(1.0, progress)):] = 0
        fonts.blit(frame, tuple(fill_progress), swept, x0, y0, opacity)
    if gloss and bh >= 24:
        fonts.blit(frame, WHITE, (rounded_mask(bw - bh // 2, bh // 2 - 3, bh // 4).astype(np.uint16) * 70 // 255).astype(np.uint8),
                   x0 + bh // 4, y0 + 4, opacity)
    if label:
        fonts.draw(frame, label, cx, cy + 1, size or round(bh * 0.46), text, anchor="mm", opacity=opacity)


def ring(frame, cx, cy, r, progress, color, thickness, track=None):
    """A circle drawn clockwise from the top as far as `progress` (0..1) says, over an optional full track."""
    centre = (round(cx), round(cy))
    if track is not None:
        cv2.circle(frame, centre, round(r), tuple(track), thickness, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, centre, (round(r), round(r)), 0, -90, -90 + 360.0 * min(1.0, progress), tuple(color), thickness,
                    cv2.LINE_AA)


def _polygon(frame, points, color, outline=None, outline_px=0):
    pts = np.array(points, dtype=np.int32)
    if outline and outline_px:
        cv2.polylines(frame, [pts], True, tuple(outline), outline_px * 2, cv2.LINE_AA)
    cv2.fillPoly(frame, [pts], tuple(color), cv2.LINE_AA)


def star(frame, cx, cy, r, color, outline=None, outline_px=0, angle_deg=0.0):
    """A five-pointed star of outer radius r."""
    pts = [(cx + (r if k % 2 == 0 else 0.45 * r) * math.cos(math.radians(angle_deg - 90 + 36 * k)),
            cy + (r if k % 2 == 0 else 0.45 * r) * math.sin(math.radians(angle_deg - 90 + 36 * k))) for k in range(10)]
    _polygon(frame, pts, color, outline, outline_px)


def burst(frame, cx, cy, r_out, r_in, spikes, color, angle_deg=0.0, outline=None, outline_px=0):
    """A spiky comic-book starburst: `spikes` points out to r_out, between them dips to r_in."""
    pts = [(cx + (r_out if k % 2 == 0 else r_in) * math.cos(math.radians(angle_deg + 180.0 * k / spikes)),
            cy + (r_out if k % 2 == 0 else r_in) * math.sin(math.radians(angle_deg + 180.0 * k / spikes)))
           for k in range(2 * spikes)]
    _polygon(frame, pts, color, outline, outline_px)


@lru_cache(maxsize=64)
def _glow_mask(r):
    d = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1])
    mask = (np.clip(1.0 - d / r, 0.0, 1.0) ** 2 * 255).astype(np.uint8)
    mask.setflags(write=False)
    return mask


def glow(frame, cx, cy, r, color, opacity=1.0):
    """A soft round glow that is brightest in the middle and gone at radius r."""
    r = max(2, round(r))
    fonts.blit(frame, tuple(color), _glow_mask(r), round(cx) - r, round(cy) - r, opacity)


def cursor(frame, x, y, *, progress=0.0, color=(60, 170, 255)):
    """The pointer: a small paddle with a halo, and a ring round it that fills as the hand holds on a button."""
    x, y = round(x), round(y)
    glow(frame, x, y, 60, (255, 255, 255), 0.6)
    cv2.circle(frame, (x, y), 40, WHITE, 3, cv2.LINE_AA)
    canvas.draw_paddle(frame, x, y - 6, 20, angle_deg=-18.0)
    ring(frame, x, y, 40, progress, color, 8)


def confetti(frame, t_s, *, seed, n=70):
    """Paper falling and spinning from the top of the picture, t_s seconds after it was thrown (the same seed, the same pieces)."""
    if t_s <= 0:
        return
    h, w = frame.shape[:2]
    rnd = random.Random(seed)
    for _ in range(n):
        x0, delay, speed = rnd.uniform(0, w), rnd.uniform(0, 1.2), rnd.uniform(160, 340)
        sway, rate, spin, size = rnd.uniform(20, 70), rnd.uniform(1.5, 3.5), rnd.uniform(-9, 9), rnd.uniform(7, 14)
        color, phase = CONFETTI[rnd.randrange(len(CONFETTI))], rnd.uniform(0, 6.3)
        age = t_s - delay
        if age <= 0:
            continue
        cx, cy = x0 + sway * math.sin(rate * age + phase), -20 + speed * age
        if cy > h + 20:
            continue
        a = spin * age
        c, s = math.cos(a), math.sin(a)
        corners = [(cx + c * px - s * py, cy + s * px + c * py) for px, py in ((-size, -size / 2), (size, -size / 2),
                                                                                  (size, size / 2), (-size, size / 2))]
        cv2.fillConvexPoly(frame, np.array(corners, dtype=np.int32), color, cv2.LINE_AA)
