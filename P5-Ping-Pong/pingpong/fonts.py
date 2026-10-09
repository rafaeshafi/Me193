"""Text in a rounded, friendly typeface (the Mac's own Arial Rounded Bold), drawn as cached sprites.

Each distinct text is rendered once with Pillow into a small picture with an alpha mask and then blended into the frame
(a few microseconds), so the menus and the HUD can use as much text as they like at 60 frames a second.  Where there is no
Pillow or no such typeface (a test machine, another OS) the old OpenCV Hershey font stands in with the same anchors and
sizes: the game never depends on the typeface.

Sizes are the height of a line of text in pixels; an anchor is two letters, across (l, m, r) then down (t, m, b).
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                                           # no Pillow: the plain font does everything
    Image = ImageDraw = ImageFont = None

FILES = ("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf", "/Library/Fonts/Arial Rounded Bold.ttf",
         "/System/Library/Fonts/SFNSRounded.ttf", "/System/Library/Fonts/Supplemental/Verdana Bold.ttf")
PLAIN = cv2.FONT_HERSHEY_SIMPLEX
SUBSTITUTES = {"\u00b7": "\u2022"}                  # the typeface has no middle dot: a bullet is its neighbour


@lru_cache(maxsize=1)
def _path():
    return next((p for p in FILES if Path(p).exists()), None)


@lru_cache(maxsize=64)
def _load(size):
    """The typeface at this size, or None when there is none (Pillow or the file missing)."""
    if ImageFont is None or _path() is None:
        return None
    try:
        return ImageFont.truetype(_path(), int(size))
    except OSError:
        return None


def available():
    return _load(20) is not None


@lru_cache(maxsize=1024)
def _lacks(ch):
    """The typeface has no picture for this character (it would be drawn as an empty box)."""
    font = _load(20)
    return font is not None and font.getmask(ch).getbbox() == font.getmask("\uffff").getbbox()


def _clean(text):
    """The text with every character that cannot be drawn replaced: by a close one if there is one, else a question mark."""
    if text.isascii():
        return text
    drawable = _load(20) is not None
    out = []
    for ch in text:
        ch = SUBSTITUTES.get(ch, ch) if drawable else ch
        out.append("?" if not ch.isascii() and (not drawable or _lacks(ch)) else ch)
    return "".join(out)


def measure(text, size):
    """(width, height) in pixels of one line of text."""
    if round(size) < 1:
        return 0, 0
    text = _clean(text)
    font = _load(round(size))
    if font is None:
        scale, thick = _plain_scale(size)
        (w, h), base = cv2.getTextSize(text, PLAIN, scale, thick)
        return w, h + base
    ascent, descent = font.getmetrics()
    return round(font.getlength(text)), ascent + descent


def wrap_lines(text, max_w, size, max_lines=3):
    """The text broken at spaces into lines no wider than max_w (cut with dots if it needs more than max_lines)."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and measure(candidate, size)[0] > max_w:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(1, len(lines[-1]) - 3)] + "..."
    for i, line in enumerate(lines):
        while len(line) > 1 and measure(line, size)[0] > max_w:
            line = line[:-1]
        lines[i] = line
    return lines


def fit_size(text, max_w, *, max_size, min_size):
    """The biggest size, from max_size down to min_size in steps of 2, at which the text is no wider than max_w."""
    size = max_size
    while size > min_size and measure(text, size)[0] > max_w:
        size -= 2
    return max(min_size, size)


@lru_cache(maxsize=1024)
def _sprite(text, size, color, outline, outline_px, anchor):
    """-> (bgr, alpha, (ax, ay)): the text as a picture and where its anchor point is inside it."""
    font = _load(size)
    rgb = lambda c: (c[2], c[1], c[0], 255)                   # noqa: E731  (the frame is BGR, Pillow's RGB)
    pad = outline_px + 2
    x0, y0, x1, y1 = font.getbbox(text, anchor=anchor, stroke_width=outline_px)
    image = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((pad - x0, pad - y0), text, font=font, fill=rgb(color), anchor=anchor,
                               stroke_width=outline_px, stroke_fill=rgb(outline) if outline_px else None)
    pixels = np.asarray(image)
    return pixels[:, :, 2::-1].copy(), pixels[:, :, 3].copy(), (pad - x0, pad - y0)


def blit(frame, color_or_bgr, alpha, x0, y0, opacity=1.0):
    """Blend a picture (or a flat colour through a mask) into the frame at (x0, y0); whatever is off the frame is cut.

    Done in whole numbers with OpenCV's vector operations: frame * (1 - a) + source * a, a being the mask times the opacity.  It
    agrees with the exact float blend to within two levels of 255 and takes half the time of it (the panels, shadows and glows of a
    menu are most of what a frame costs)."""
    h, w = alpha.shape
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sx0, sy0 = fx0 - x0, fy0 - y0
    weight = alpha[sy0:sy0 + fy1 - fy0, sx0:sx0 + fx1 - fx0]
    if opacity < 1.0:
        weight = cv2.convertScaleAbs(weight, alpha=max(0.0, opacity))
    roi = frame[fy0:fy1, fx0:fx1]
    weight3 = cv2.merge((weight, weight, weight))
    if isinstance(color_or_bgr, np.ndarray):
        top = cv2.multiply(color_or_bgr[sy0:sy0 + fy1 - fy0, sx0:sx0 + fx1 - fx0], weight3, scale=1 / 255.0)
    else:
        b, g, r = color_or_bgr
        top = cv2.multiply(weight3, (float(b), float(g), float(r), 0.0), scale=1 / 255.0)       # (a flat colour needs no picture)
    cv2.add(cv2.multiply(roi, cv2.bitwise_not(weight3), scale=1 / 255.0), top, dst=roi)


def draw(frame, text, x, y, size, color, *, anchor="mm", outline=None, outline_px=0, shadow=None, opacity=1.0):
    """Draw one line of text with its anchor point at (x, y).

    outline: a colour and a width in pixels round the letters; shadow: (dx, dy, colour, opacity) behind them;
    opacity: the whole text's, 0..1."""
    size = round(size)
    if size < 1:                                              # a thing still growing from nothing has no text yet
        return
    text = _clean(text)
    if _load(size) is None:
        return _draw_plain(frame, text, x, y, size, color, anchor, outline, outline_px, shadow, opacity)
    bgr, alpha, (ax, ay) = _sprite(text, size, tuple(color), tuple(outline) if outline and outline_px else None,
                                   outline_px if outline else 0, anchor)
    x0, y0 = round(x) - ax, round(y) - ay
    if shadow is not None:
        dx, dy, shade, shade_opacity = shadow
        blit(frame, tuple(shade), alpha, x0 + dx, y0 + dy, shade_opacity * opacity)
    blit(frame, bgr, alpha, x0, y0, opacity)


# --- the plain stand-in ------------------------------------------------------------------------------------------------
def _plain_scale(size):
    return size / 34.0, max(1, round(size / 14))


def _draw_plain(frame, text, x, y, size, color, anchor, outline, outline_px, shadow, opacity):
    scale, thick = _plain_scale(size)
    (w, h), base = cv2.getTextSize(text, PLAIN, scale, thick)
    across, down = anchor
    ox = round(x) - {"l": 0, "m": w // 2, "r": w}[across]
    oy = round(y) + {"t": h, "m": h // 2, "b": -base}[down]
    target = frame if opacity >= 1.0 else frame.copy()
    if shadow is not None:
        cv2.putText(target, text, (ox + shadow[0], oy + shadow[1]), PLAIN, scale, tuple(shadow[2]), thick, cv2.LINE_AA)
    if outline and outline_px:
        cv2.putText(target, text, (ox, oy), PLAIN, scale, tuple(outline), thick + 2 * outline_px, cv2.LINE_AA)
    cv2.putText(target, text, (ox, oy), PLAIN, scale, tuple(color), thick, cv2.LINE_AA)
    if target is not frame:
        cv2.addWeighted(target, opacity, frame, 1.0 - opacity, 0, frame)
