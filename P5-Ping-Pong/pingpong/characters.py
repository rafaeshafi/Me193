"""Drawing the cast (cast.py): a head-and-shoulders portrait with moods, and the opponent standing behind the far end of the table.

Everything is drawn with filled shapes and soft outlines at any size.  The portrait is laid out in "bust units" (the whole
portrait is 100 units tall, x across, y down from its centre) and scaled by `size`; a head bobs a little and blinks now and
then so a menu full of them feels alive.
"""

import math

import cv2
import numpy as np

from pingpong import anim, fonts, ui

MOODS = ("happy", "grin", "cheer", "sad", "surprised", "smug", "neutral")
BOB_PERIOD_S, BOB_UNITS = 1.6, 1.4
BLINK_PERIOD_S, BLINK_AT_S, BLINK_S = 3.2, 3.0, 0.12
EYE, WHITE, MOUTH, TONGUE, CHEEK, FRAME = (40, 30, 40), (255, 255, 255), (50, 40, 130), (110, 110, 235), (170, 150, 245), (50, 50, 60)
SHIFT, ONE = 3, 8                                           # sub-pixel drawing: coordinates are multiplied by 2 ** SHIFT

HEAD_Y_UNITS = -10.0
OPP_BUST_M, OPP_CENTER_Y_M, OPP_FOLLOW = 0.66, 0.45, 0.5     # the opponent: bust height, where its centre is above the table, how far it follows the paddle
SHOULDER = (0.34, 0.30)                                      # across and down from the bust centre, in bust heights


def eyes_closed(t_s):
    return BLINK_AT_S <= t_s % BLINK_PERIOD_S < BLINK_AT_S + BLINK_S


def _dark(color, k=0.62):
    return tuple(int(c * k) for c in color)


class _Pen:
    """Draws in bust units: (0, 0) is the centre of the portrait, one unit is size / 100 pixels."""

    def __init__(self, frame, cx, cy, u):
        self.frame, self.cx, self.cy, self.u = frame, cx, cy, u

    def _pt(self, x, y):
        return round((self.cx + x * self.u) * ONE), round((self.cy + y * self.u) * ONE)

    def ell(self, x, y, ax, ay, color, *, outline=True, start=0, end=360, edge=1.1):
        if outline:
            cv2.ellipse(self.frame, self._pt(x, y), (round((ax + edge) * self.u * ONE), round((ay + edge) * self.u * ONE)), 0,
                        start, end, _dark(color), -1, cv2.LINE_AA, SHIFT)
        cv2.ellipse(self.frame, self._pt(x, y), (round(ax * self.u * ONE), round(ay * self.u * ONE)), 0, start, end,
                    tuple(color), -1, cv2.LINE_AA, SHIFT)

    def arc(self, x, y, ax, ay, start, end, color, width):
        cv2.ellipse(self.frame, self._pt(x, y), (round(ax * self.u * ONE), round(ay * self.u * ONE)), 0, start, end,
                    tuple(color), max(1, round(width * self.u)), cv2.LINE_AA, SHIFT)

    def poly(self, points, color, *, outline=True):
        pts = np.array([self._pt(x, y) for x, y in points], dtype=np.int32)
        if outline:
            cv2.polylines(self.frame, [pts], True, _dark(color), max(1, round(2.2 * self.u)), cv2.LINE_AA, SHIFT)
        cv2.fillPoly(self.frame, [pts], tuple(color), cv2.LINE_AA, SHIFT)

    def line(self, x0, y0, x1, y1, color, width):
        cv2.line(self.frame, self._pt(x0, y0), self._pt(x1, y1), tuple(color), max(1, round(width * self.u)), cv2.LINE_AA, SHIFT)

    def blend_ell(self, x, y, ax, ay, color, alpha):
        layer = self.frame.copy()
        cv2.ellipse(layer, self._pt(x, y), (round(ax * self.u * ONE), round(ay * self.u * ONE)), 0, 0, 360, tuple(color), -1,
                    cv2.LINE_AA, SHIFT)
        cv2.addWeighted(layer, alpha, self.frame, 1 - alpha, 0, self.frame)


def _arc_points(cx, cy, ax, ay, start, end, n=14):
    return [(cx + ax * math.cos(math.radians(a)), cy + ay * math.sin(math.radians(a)))
            for a in np.linspace(start, end, n)]


# --- the parts of a portrait ------------------------------------------------------------------------------------------------
def _torso(p, look):
    p.ell(0, 50, 46, 30, look.shirt, start=180, end=360)
    p.arc(0, 50, 40, 24, 205, 335, look.trim, 3.2)                       # a sporty stripe along the shoulders
    p.poly([(-8, 6), (8, 6), (8, 26), (-8, 26)], _dark(look.skin, 0.88), outline=False)            # neck
    p.poly([(-12, 22), (12, 22), (0, 40)], look.skin, outline=False)                              # the V of the collar
    p.line(-12, 22, 0, 40, look.trim, 2.6)
    p.line(12, 22, 0, 40, look.trim, 2.6)


def _hairline(x):
    return -29.0 + 17.0 * (x / 32.0) ** 2


def _hair_cap(p, look, spikes=0):
    outer = _arc_points(0, -11, 34, 37, 180, 360, 20)
    inner = [(x, _hairline(x)) for x in np.linspace(32, -32, 12)]
    p.poly(outer + inner, look.hair)
    for k in range(spikes):
        a = math.radians(205 + k * 32.5)
        bx, by = 33 * math.cos(a), -11 + 36 * math.sin(a)
        tx, ty = 47 * math.cos(a), -11 + 50 * math.sin(a)
        side = (math.cos(a + math.pi / 2) * 7, math.sin(a + math.pi / 2) * 7)
        p.poly([(bx - side[0], by - side[1]), (tx, ty), (bx + side[0], by + side[1])], look.hair)


def _hair_back(p, look):
    if look.hair_style == "curly":
        for a in range(165, 380, 24):
            p.ell(36 * math.cos(math.radians(a)), -11 + 38 * math.sin(math.radians(a)), 11.5, 11.5, look.hair)
    elif look.hair_style == "bob":
        p.poly([(-36, -24)] + _arc_points(0, 12, 37, 17, 180, 0, 12) + [(36, -24)], look.hair)


def _hair_front(p, look):
    style = look.hair_style
    if style in ("short", "bob", "curly"):
        _hair_cap(p, look)
        if style == "curly":
            for x in (-14, 0, 14):
                p.ell(x, -33 + abs(x) * 0.25, 9.5, 9.5, look.hair)
    elif style == "spiky":
        _hair_cap(p, look, spikes=5)
    elif style == "bun":
        _hair_cap(p, look)
        p.ell(0, -54, 11, 10, look.hair)
        p.arc(0, -46, 10, 3, 0, 180, look.trim, 3)


def _brows(p, look, mood):
    inner, outer = {"sad": (-22.0, -18.5), "surprised": (-24.0, -24.5), "cheer": (-24.0, -24.5)}.get(mood, (-20.0, -21.5))
    color = _dark(look.hair, 0.7)
    for side in (-1, 1):
        raised = -4.5 if mood == "smug" and side == -1 else 2.0 if mood == "smug" else 0.0
        p.line(side * 6.5, inner + raised, side * 15, outer + raised, color, 2.6)


def _mouth(p, mood):
    if mood == "grin":
        p.ell(0, 9, 12, 9, MOUTH, start=0, end=180, edge=0.8)
        p.ell(0, 7.5, 9.5, 3.4, WHITE, outline=False, start=0, end=180)
    elif mood == "cheer":
        p.ell(0, 8, 13, 12, MOUTH, start=0, end=180, edge=0.8)
        p.ell(0, 15, 7.5, 4.2, TONGUE, outline=False, start=180, end=360)
    elif mood == "sad":
        p.arc(0, 19, 8.5, 5.5, 205, 335, MOUTH, 2.3)
    elif mood == "surprised":
        p.ell(0, 13, 4.6, 5.8, MOUTH, edge=0.8)
    elif mood == "smug":
        p.arc(3, 10, 10, 5.5, 8, 125, MOUTH, 2.3)
    elif mood == "neutral":
        p.line(-6, 13, 6, 13, MOUTH, 2.3)
    else:
        p.arc(0, 8, 10, 7, 22, 158, MOUTH, 2.4)


def _face(p, look, mood, closed):
    for side in (-1, 1):
        p.blend_ell(side * 18, 5, 6.5, 4.2, CHEEK, 0.55)
        if closed:
            p.arc(side * 11, -9, 4.6, 3.2, 200, 340, EYE, 2.0)
        else:
            p.ell(side * 11, -9, 4.1, 5.4, EYE, outline=False)
            p.ell(side * 11 - 1.4, -11.4, 1.5, 1.5, WHITE, outline=False)
    if look.freckles:
        for dx, dy in ((-13, 1), (-17, 3), (-14, 5), (13, 1), (17, 3), (14, 5)):
            p.ell(dx, dy, 0.9, 0.9, _dark(look.skin, 0.8), outline=False)
    _brows(p, look, mood)
    p.ell(0, 2, 2.6, 1.8, _dark(look.skin, 0.8), outline=False)
    _mouth(p, mood)


def _accessory(p, look):
    kind = look.accessory
    if kind == "cap":
        for side in (-1, 1):
            p.ell(side * 31, -9, 6.5, 8, look.hair)
        p.ell(0, -16, 35, 32, look.trim, start=180, end=360)
        p.ell(0, -16, 38, 9.5, _dark(look.trim, 0.82), start=0, end=180)
        p.ell(0, -48, 3.2, 3.2, _dark(look.trim, 0.7), outline=False)
    elif kind == "headband":
        pts = _arc_points(0, -27, 32, 7, 170, 10, 12)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            p.line(x0, y0 + 4, x1, y1 + 4, look.trim, 5.0)
    elif kind == "glasses":
        for side in (-1, 1):
            p.arc(side * 11, -9, 9.5, 8.5, 0, 360, FRAME, 2.0)
        p.line(-1.5, -10, 1.5, -10, FRAME, 2.0)
        p.line(-20.5, -10, -29, -12, FRAME, 1.6)
        p.line(20.5, -10, 29, -12, FRAME, 1.6)
    elif kind == "sunglasses":
        for side in (-1, 1):
            p.ell(side * 11.5, -9, 10.8, 8, (45, 40, 40))
            p.line(side * 11.5 - 5, -13, side * 11.5 - 1, -9.5, (210, 210, 210), 1.6)
        p.line(-1, -10.5, 1, -10.5, (45, 40, 40), 2.2)
        p.line(-22, -10.5, -29, -12.5, (45, 40, 40), 1.8)
        p.line(22, -10.5, 29, -12.5, (45, 40, 40), 1.8)


def draw_bust(frame, cx, cy, size, look, *, mood="happy", t=0.0, bob=True):
    """A head-and-shoulders portrait, `size` pixels tall, centred on (cx, cy)."""
    u = size / 100.0
    body = _Pen(frame, cx, cy, u)
    _torso(body, look)
    dy = anim.bob(t, period_s=BOB_PERIOD_S, amplitude=BOB_UNITS) if bob else 0.0
    head = _Pen(frame, cx, cy + dy * u, u)
    _hair_back(head, look)
    for side in (-1, 1):
        head.ell(side * 30, -8, 6, 7, look.skin)                          # ears
    head.ell(0, HEAD_Y_UNITS, 30, 31, look.skin)
    _face(head, look, mood, eyes_closed(t))
    _hair_front(head, look)
    _accessory(head, look)


# --- the opponent behind the table, and the crowd ------------------------------------------------------------------------------
FLOOR_Y_M = -0.76                                            # the deck the figures stand on
SPEC_BUST_M, SPEC_CENTER_Y_M = 0.62, 0.37                     # a spectator: bust height, and where its centre is above the table top


def head_px(cam, x_m, z_m):
    """-> (x, y, pixels per metre): where the opponent's portrait is centred when its paddle is at x_m (it follows a little)."""
    px, py, scale = cam.project(OPP_FOLLOW * x_m, OPP_CENTER_Y_M, z_m)
    return px, py, scale


def shoulder_px(cam, x_m, z_m):
    """The opponent's racket-arm shoulder (on the left of the picture: it is facing us)."""
    cx, cy, scale = head_px(cam, x_m, z_m)
    size = OPP_BUST_M * scale
    return cx - SHOULDER[0] * size, cy + SHOULDER[1] * size


def _p(x, y):
    return round(x * ONE), round(y * ONE)


def _body_block(view, cx, cy, size, bottom_row, look, half):
    """The body under a portrait: a block of shirt down to bottom_row (clipped to the view)."""
    rows = min(view.shape[0], max(0, round(bottom_row)))
    top = cy + 0.5 * size - 2
    if rows <= top:
        return
    block = np.array([(cx - half, top), (cx + half, top), (cx + half, rows), (cx - half, rows)])
    cv2.fillPoly(view, [(block * ONE).astype(np.int32)], tuple(look.shirt), cv2.LINE_AA, SHIFT)
    for x in (cx - half, cx + half):
        cv2.line(view, _p(x, cy + 0.5 * size), _p(x, rows), _dark(look.shirt), 2, cv2.LINE_AA, SHIFT)


def _limb(frame, a, b, look, size, bare=0.38):
    """An arm from a shoulder to a hand: a sleeve, and the last part of it bare, with a hand at the end."""
    mid = (a[0] + (1 - bare) * (b[0] - a[0]), a[1] + (1 - bare) * (b[1] - a[1]))
    for p0, p1, color, width in ((a, mid, look.shirt, 0.17 * size), (mid, b, look.skin, 0.12 * size)):
        cv2.line(frame, _p(*p0), _p(*p1), _dark(color), max(1, round(width)) + 3, cv2.LINE_AA, SHIFT)
        cv2.line(frame, _p(*p0), _p(*p1), tuple(color), max(1, round(width)), cv2.LINE_AA, SHIFT)


def _raised_hand(frame, hand, look, size):
    cv2.circle(frame, _p(*hand), round(0.075 * size * ONE), _dark(look.skin), -1, cv2.LINE_AA, SHIFT)
    cv2.circle(frame, _p(*hand), round(0.065 * size * ONE), tuple(look.skin), -1, cv2.LINE_AA, SHIFT)


def draw_opponent(frame, cam, look, *, x_m, z_m, grip_px, mood="happy", t=0.0, body_rows=None):
    """The opponent: a block of body up to the table's far edge (the first `body_rows` rows of the frame, or all of a view
    that ends there), the portrait on it, its racket arm reaching the paddle's grip and, cheering, the other arm up."""
    cx, cy, scale = head_px(cam, x_m, z_m)
    size = OPP_BUST_M * scale
    view = frame if body_rows is None else frame[:body_rows]
    _body_block(view, cx, cy, size, min(view.shape[0], cam.project(OPP_FOLLOW * x_m, FLOOR_Y_M, z_m)[1]), look, 0.46 * size)
    draw_bust(view, cx, cy, size, look, mood=mood, t=t)
    sx, sy = shoulder_px(cam, x_m, z_m)
    _limb(frame, (sx, sy), grip_px, look, size, bare=0.38)
    if mood == "cheer":                                                  # the other arm goes up
        hand = (cx + 0.75 * size, cy - 0.55 * size)
        _limb(frame, (cx + SHOULDER[0] * size, cy + SHOULDER[1] * size), hand, look, size, bare=0.2)
        _raised_hand(frame, hand, look, size)


def draw_spectator(frame, cam, look, *, x_m, z_m, mood="happy", t=0.0):
    """A person watching from the deck beside the court: standing on the floor, cheering with both arms up or calm with them down."""
    cx, cy, scale = cam.project(x_m, SPEC_CENTER_Y_M, z_m)
    size = SPEC_BUST_M * scale
    if size < 8 or cam.depth_of(x_m, SPEC_CENTER_Y_M, z_m) < 0.5:                     # too small to see, or behind the camera
        return
    _body_block(frame, cx, cy, size, cam.project(x_m, FLOOR_Y_M, z_m)[1], look, 0.36 * size)
    draw_bust(frame, cx, cy, size, look, mood=mood, t=t)
    for side in (-1, 1):
        shoulder = (cx + side * SHOULDER[0] * size, cy + SHOULDER[1] * size)
        up = mood == "cheer"
        hand = (cx + side * (0.78 if up else 0.52) * size, cy + (-0.62 if up else 0.62) * size)
        _limb(frame, shoulder, hand, look, size, bare=0.25)
        if up:
            _raised_hand(frame, hand, look, size)


def portrait(frame, cx, cy, diameter, look, *, mood="happy", t=0.0, bg=((255, 220, 160), (255, 245, 210))):
    """A round picture of someone on a soft background, with a white rim and a little shadow: the face on a card."""
    d = round(diameter)
    if d < 8:                                                  # still growing from nothing
        return
    picture = np.empty((d, d, 3), dtype=np.uint8)
    picture[:] = ui.gradient(d, d, tuple(bg[0]), tuple(bg[1]))
    draw_bust(picture, d / 2, d * 0.56, d * 0.98, look, mood=mood, t=t)
    x0, y0 = round(cx - d / 2), round(cy - d / 2)
    mask = ui.rounded_mask(d, d, d // 2)
    shadow, pad = ui._shadow(d, d, d // 2)
    fonts.blit(frame, (0, 0, 0), shadow, x0 - pad, y0 - pad + ui.SHADOW_DY, ui.SHADOW_OPACITY)
    fonts.blit(frame, picture, mask, x0, y0)
    rim = max(3, d // 24)
    ui.ring(frame, cx, cy, d / 2 - rim / 2, 1.0, (255, 255, 255), rim)
