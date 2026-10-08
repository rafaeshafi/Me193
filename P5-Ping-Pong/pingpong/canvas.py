"""Drawing primitives (numpy BGR frames, no window): text, paddles, strip charts, panels.  The court itself is
scene.py and court3d.py."""

import math

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
RUBBER_RED, RUBBER_BLACK = (35, 35, 200), (44, 40, 38)          # BGR: a table-tennis paddle is red on one side, black on the other
WOOD, WOOD_DARK, RIM = (100, 165, 220), (45, 85, 140), (25, 25, 25)
SKIN, SKIN_DARK = (130, 175, 235), (70, 105, 165)


def new_frame(w, h):
    """A dark vertical-gradient backdrop."""
    ramp = np.linspace(28, 8, h, dtype=np.uint8)[:, None, None]
    return np.repeat(np.repeat(ramp, w, axis=1), 3, axis=2).copy()


def fit_background(image, w, h, darken=0.45):
    """Camera frame -> HUD-sized, darkened so the overlay stays readable."""
    return (cv2.resize(image, (w, h)).astype(np.float32) * darken).astype(np.uint8)


def draw_text(frame, text, org, scale=1.0, color=(255, 255, 255), thickness=2, anchor="left"):
    """Outlined text; anchor in {"left", "center", "right"} relative to org.x."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    if anchor == "center":
        x -= tw // 2
    elif anchor == "right":
        x -= tw
    cv2.putText(frame, text, (x, y), FONT, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def fit_text(text, max_w, *, max_scale=1.7, min_scale=0.7, thickness=2, max_lines=3):
    """-> (lines, scale) for a message that must fit max_w pixels.

    One line at the largest scale that fits (a short banner stays big); a long notice shrinks to min_scale and
    only then wraps into <= max_lines lines.  Text that still does not fit is cut with "..."."""
    def width(line, scale):
        return cv2.getTextSize(line, FONT, scale, thickness)[0][0]

    scale = max_scale
    while True:
        if width(text, scale) <= max_w:
            return [text], scale
        if scale <= min_scale + 1e-9:
            break
        scale = max(min_scale, scale * 0.95)
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and width(candidate, min_scale) > max_w:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    dropped = len(lines) > max_lines
    lines = lines[:max_lines]
    for i, line in enumerate(lines):                                    # trim whatever is still too wide (a long word)
        cut = dropped and i == len(lines) - 1
        while line and width(line + ("..." if cut else ""), min_scale) > max_w:
            line, cut = line[:-1], True
        lines[i] = line + ("..." if cut else "")
    return lines, min_scale


def draw_fitted(frame, text, center_x, top_y, max_w, *, max_scale=1.7, min_scale=0.7, color=(255, 255, 255),
                thickness=4, max_lines=3):
    """A message that always fits: shrunk and wrapped as needed, centred on center_x, first baseline at top_y."""
    lines, scale = fit_text(text, max_w, max_scale=max_scale, min_scale=min_scale, thickness=thickness,
                            max_lines=max_lines)
    line_h = int(34 * scale + 14)
    for i, line in enumerate(lines):
        draw_text(frame, line, (center_x, top_y + i * line_h), scale, color, max(1, round(thickness * scale / max_scale)),
                  anchor="center")
    return len(lines)


GRIP_RADII = 2.05                                                     # the grip is this many face radii from the face's centre


def paddle_grip(cx, cy, radius, angle_deg, *, handle_up=False):
    """Where a hand holds a paddle whose face is centred on (cx, cy): on the handle, below the face (above it for the
    player at the far end), turned with the paddle (a positive angle is clockwise as the player sees it)."""
    a, sign = math.radians(angle_deg), (-1.0 if handle_up else 1.0)
    return cx - sign * math.sin(a) * GRIP_RADII * radius, cy + sign * math.cos(a) * GRIP_RADII * radius


def draw_paddle(frame, cx, cy, radius, *, rubber=RUBBER_RED, angle_deg=0.0, handle_up=False, hand=False, skin=None):
    """A table-tennis paddle: a rubber face inside a black rim, on a short wooden handle (and, if asked, the fist
    holding it, in `skin`).

    (cx, cy) is the centre of the FACE, which is the surface that hits the ball; the handle hangs below it (above it
    for the player at the far end) and swings about the face's centre by angle_deg."""
    a, sign = math.radians(angle_deg), (-1.0 if handle_up else 1.0)
    d = (-sign * math.sin(a), sign * math.cos(a))                       # from the face towards the grip
    n = (-d[1], d[0])
    near, far = (cx + d[0] * 0.8 * radius, cy + d[1] * 0.8 * radius), (cx + d[0] * 2.5 * radius, cy + d[1] * 2.5 * radius)
    handle = np.array([(near[0] + n[0] * 0.22 * radius, near[1] + n[1] * 0.22 * radius),
                       (far[0] + n[0] * 0.19 * radius, far[1] + n[1] * 0.19 * radius),
                       (far[0] - n[0] * 0.19 * radius, far[1] - n[1] * 0.19 * radius),
                       (near[0] - n[0] * 0.22 * radius, near[1] - n[1] * 0.22 * radius)], dtype=np.int32)
    cv2.fillConvexPoly(frame, handle, WOOD, cv2.LINE_AA)
    cv2.polylines(frame, [handle], True, WOOD_DARK, 2, cv2.LINE_AA)
    if hand:                                                            # the fist round the handle
        fist = tuple(round(v) for v in paddle_grip(cx, cy, radius, angle_deg, handle_up=handle_up))
        colour = SKIN if skin is None else tuple(skin)
        cv2.circle(frame, fist, max(3, round(0.5 * radius)), tuple(int(c * 0.6) for c in colour), -1, cv2.LINE_AA)
        cv2.circle(frame, fist, max(2, round(0.42 * radius)), colour, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, RIM, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), round(0.9 * radius), rubber, -1, cv2.LINE_AA)
    shine = tuple(min(255, c + 80) for c in rubber)                     # a glossy arc on the upper left of the rubber
    cv2.ellipse(frame, (cx, cy), (round(0.72 * radius), round(0.72 * radius)), 0, 190, 245, shine, 2, cv2.LINE_AA)


def tint(frame, color, alpha):
    overlay = np.full_like(frame, color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_trace(frame, x0, y0, w, h, values, scale, threshold=0.0, color=(200, 200, 200), hot=(80, 220, 80)):
    """A strip chart: values of +-scale fill the panel's height; segments at or above `threshold` are `hot`."""
    panel = frame[y0:y0 + h, x0:x0 + w]
    panel[:] = (panel * 0.4).astype(np.uint8)
    mid = y0 + h // 2
    cv2.line(frame, (x0, mid), (x0 + w, mid), (110, 110, 110), 1)
    if threshold > 0:
        ty = int(mid - min(1.0, threshold / scale) * (h / 2 - 2))
        cv2.line(frame, (x0, ty), (x0 + w, ty), (40, 170, 255), 1, cv2.LINE_AA)
    n = len(values)
    pts = [(x0 + (i * (w - 1)) // max(1, n - 1), int(mid - max(-1.0, min(1.0, v / scale)) * (h / 2 - 2)))
           for i, v in enumerate(values)]
    if n == 1:
        cv2.circle(frame, pts[0], 3, color, -1, cv2.LINE_AA)
    for (a, va), (b, vb) in zip(zip(pts, values), zip(pts[1:], values[1:])):
        cv2.line(frame, a, b, hot if max(va, vb) >= threshold > 0 else color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (170, 170, 170), 1)


def panel(frame, x0, y0, w, h):
    """A score box: the picture behind it darkened, with a bright border (the arcade original's POINTS boxes)."""
    region = frame[y0:y0 + h, x0:x0 + w]
    region[:] = (region * 0.30).astype(np.uint8)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (235, 235, 235), 2)


def dim_rect(frame, x0, y0, w, h, factor=0.35):
    """Darken a rectangle so text on top stays readable over the court or the camera picture."""
    region = frame[y0:y0 + h, x0:x0 + w]
    region[:] = (region * factor).astype(np.uint8)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (170, 170, 170), 1)
