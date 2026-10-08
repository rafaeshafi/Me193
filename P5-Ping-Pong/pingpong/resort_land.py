"""The land of the resort: the island with its palms, the walkway out to the terrace, the terrace itself (deck, railings,
bunting, potted plants) and the shadow of the table.  World metres as in physics.py (y up from the table top, z along the
table), the deck floor 76 cm below the table top, the sea 3 m below the deck."""

import math

import cv2
import numpy as np

from pingpong import court3d, physics, resort_geo as geo
from pingpong.cast import rgb
from pingpong.resort_sky import SEA_Y

DECK_Y = -0.76
DECK_X, DECK_Z0, DECK_Z1 = 5.0, -4.0, 5.5
RAIL_Z, RAIL_TOP_Y, RAIL_MID_Y = DECK_Z1, 0.30, -0.22
ISLAND = (0.0, -78.0, 42.0)
WALK_HALF, WALK_Z0 = 1.5, -36.0
HW, L = physics.HALF_WIDTH_M, physics.TABLE_LEN_M

DECK, DECK_ALT, DECK_SEAM, DECK_EDGE = rgb(238, 200, 148), rgb(230, 188, 134), rgb(196, 150, 100), rgb(176, 128, 84)
RAIL, RAIL_SHADE = rgb(250, 250, 252), rgb(196, 212, 232)
SAND, SAND_SHADE, SHALLOW, FOAM = rgb(250, 226, 170), rgb(232, 200, 140), rgb(122, 236, 226), rgb(255, 255, 255)
GRASS, GRASS_DARK = rgb(96, 204, 112), rgb(54, 150, 92)
HILLS = (rgb(84, 184, 104), rgb(104, 200, 112), rgb(124, 214, 122))
TRUNK, FROND, FROND_DARK, POT = rgb(176, 124, 84), rgb(64, 196, 104), rgb(34, 148, 84), rgb(222, 120, 84)
PILE = rgb(138, 100, 72)
BUNTING = (rgb(255, 90, 90), rgb(255, 210, 70), rgb(70, 160, 255), rgb(90, 210, 120), rgb(255, 130, 190))
SHADOW = (0, 0, 0)

_RNG = np.random.default_rng(21)


def _palm_sites():
    sites = []
    for _ in range(18):
        a, frac = _RNG.uniform(0, 2 * math.pi), _RNG.uniform(0.55, 0.97)
        y = SEA_Y + (0.6 if frac > 0.85 else 1.8 if frac > 0.62 else 5.5)
        sites.append((ISLAND[0] + ISLAND[2] * frac * math.cos(a), y, ISLAND[1] + ISLAND[2] * frac * math.sin(a),
                      float(_RNG.uniform(6.5, 10.5)), float(_RNG.uniform(-1.6, 1.6))))
    return sites


PALMS = _palm_sites()


# --- leaves, palms and plants (flat fans standing in the picture's plane, in the style of the rest) --------------------------------
def _leaf(x, y, z, angle_deg, length, width, droop):
    """A leaf as a world polygon in the x-y plane: from (x, y) out along `angle` (0 = straight up), bending down as it goes."""
    a = math.radians(angle_deg)
    nx, ny = math.cos(a), -math.sin(a)
    dx, dy = math.sin(a), math.cos(a)
    centre, px, py, steps = [], x, y, 7
    for _ in range(steps + 1):
        centre.append((px, py))
        px, py, dy = px + dx * length / steps, py + dy * length / steps, dy - droop / steps
    upper, lower = [], []
    for k, (cx, cy) in enumerate(centre):
        half = width * math.sin(math.pi * min(1.0, (k + 0.3) / (steps + 0.6))) ** 0.8
        upper.append((cx + nx * half, cy + ny * half, z))
        lower.append((cx - nx * half, cy - ny * half, z))
    return upper + lower[::-1]


def palm(frame, cam, x, y0, z, height, lean, t_s):
    depth = cam.depth_of(x, y0 + height / 2, z)
    if depth < 4.0 or depth > 700.0:
        return
    sway = 0.25 * math.sin(t_s * 1.3 + x)
    top = (x + lean, y0 + height)
    mid = (x + 0.15 * lean, y0 + 0.55 * height)
    pts = [((1 - u) ** 2 * x + 2 * (1 - u) * u * mid[0] + u ** 2 * top[0], (1 - u) ** 2 * y0 + 2 * (1 - u) * u * mid[1] + u ** 2 * top[1])
           for u in np.linspace(0, 1, 6)]
    for (xa, ya), (xb, yb), u in zip(pts, pts[1:], np.linspace(0, 1, 5)):
        geo.line(frame, cam, (xa, ya, z), (xb, yb, z), 0.5 - 0.22 * u, TRUNK)
    leaves = 9 if depth < 140 else 5 if depth < 320 else 3
    for k in range(leaves):
        angle = -100.0 + 200.0 * k / (leaves - 1) + sway * 12
        color = FROND if k % 2 == 0 else FROND_DARK
        geo.fill(frame, cam, _leaf(top[0], top[1], z, angle, 0.5 * height * 0.55, 0.28 * height * 0.14, 0.9), color)


def plant(frame, cam, x, z):
    """A potted plant by the railing: a pot and a fan of big leaves."""
    geo.fill(frame, cam, [(x - 0.2, DECK_Y, z), (x + 0.2, DECK_Y, z), (x + 0.27, DECK_Y + 0.45, z), (x - 0.27, DECK_Y + 0.45, z)], POT)
    for k, angle in enumerate((-72, -48, -24, 0, 24, 48, 72)):
        color = FROND if k % 2 == 0 else FROND_DARK
        geo.fill(frame, cam, _leaf(x, DECK_Y + 0.42, z, angle, 0.85 + 0.15 * (k % 3), 0.16, 0.5), color)


# --- the island -----------------------------------------------------------------------------------------------------------------------
def island(frame, cam, t_s):
    cx, cz, r = ISLAND
    if math.hypot(cx - cam.pos[0], cz - cam.pos[2]) > 1500:
        return
    flat = lambda ring, y: [(x, y, z) for x, z in ring]                              # noqa: E731
    geo.fill(frame, cam, flat(geo.blob(cx, cz, r * 1.26, 3), SEA_Y + 0.02), FOAM, 0.55)
    geo.fill(frame, cam, flat(geo.blob(cx, cz, r * 1.15, 3), SEA_Y + 0.04), SHALLOW, 0.9)
    for (frac, seed, y0, y1, side, top) in ((1.0, 3, SEA_Y, SEA_Y + 0.7, SAND_SHADE, SAND), (0.82, 4, SEA_Y + 0.7, SEA_Y + 2.0, GRASS_DARK, GRASS),
                                            (0.6, 5, SEA_Y + 2.0, SEA_Y + 5.5, GRASS_DARK, HILLS[0]),
                                            (0.42, 6, SEA_Y + 5.5, SEA_Y + 9.0, GRASS_DARK, HILLS[1]),
                                            (0.24, 7, SEA_Y + 9.0, SEA_Y + 11.5, GRASS_DARK, HILLS[2])):
        geo.prism(frame, cam, geo.blob(cx, cz, r * frac, seed), y0, y1, side, top)
    for x, y, z, height, lean in sorted(PALMS, key=lambda p: -cam.depth_of(p[0], p[1], p[2])):
        palm(frame, cam, x, y, z, height, lean, t_s)


def walkway(frame, cam):
    """The pier from the island's beach to the terrace, on piles, with a handrail on each side."""
    z0, z1 = WALK_Z0, DECK_Z0
    for z in np.arange(z0 + 4, z1, 6.0):
        for x in (-WALK_HALF + 0.2, WALK_HALF - 0.2):
            geo.line(frame, cam, (x, DECK_Y - 0.4, z), (x, SEA_Y, z), 0.3, PILE)
    ring = [(-WALK_HALF, z0), (WALK_HALF, z0), (WALK_HALF, z1), (-WALK_HALF, z1)]
    geo.prism(frame, cam, ring, DECK_Y - 0.4, DECK_Y, DECK_EDGE, DECK)
    near = math.hypot(cam.pos[0], cam.pos[2] - (z0 + z1) / 2) < 70
    if near:
        for z in np.arange(z0 + 1, z1, 1.0):
            geo.line(frame, cam, (-WALK_HALF, DECK_Y, z), (WALK_HALF, DECK_Y, z), 0.02, DECK_SEAM)
    for side in (-1, 1):
        x = side * WALK_HALF
        for z in np.arange(z0, z1 + 0.1, 3.0):
            geo.line(frame, cam, (x, DECK_Y, z), (x, DECK_Y + 1.05, z), 0.09, RAIL)
        geo.line(frame, cam, (x, DECK_Y + 1.05, z0), (x, DECK_Y + 1.05, z1), 0.09, RAIL)


# --- the terrace ----------------------------------------------------------------------------------------------------------------------
def deck(frame, cam):
    ring = [(-DECK_X, DECK_Z0), (DECK_X, DECK_Z0), (DECK_X, DECK_Z1), (-DECK_X, DECK_Z1)]
    for x in (-DECK_X + 0.3, DECK_X - 0.3):
        for z in np.arange(DECK_Z0 + 0.5, DECK_Z1, 3.0):
            geo.line(frame, cam, (x, DECK_Y - 1.0, z), (x, SEA_Y, z), 0.35, PILE)
    geo.prism(frame, cam, ring, DECK_Y - 1.0, DECK_Y, DECK_EDGE, DECK)
    if math.hypot(cam.pos[0], cam.pos[2] - 0.7) > 120:
        return
    planks = np.arange(-DECK_X, DECK_X + 0.01, 0.4)
    for k, (xa, xb) in enumerate(zip(planks, planks[1:])):
        if k % 2:
            geo.fill(frame, cam, [(xa, DECK_Y, DECK_Z0), (xb, DECK_Y, DECK_Z0), (xb, DECK_Y, DECK_Z1), (xa, DECK_Y, DECK_Z1)], DECK_ALT)
    for x in planks[1:-1]:
        geo.line(frame, cam, (x, DECK_Y, DECK_Z0), (x, DECK_Y, DECK_Z1), 0.012, DECK_SEAM)


def table_shadow(frame, cam):
    geo.fill(frame, cam, [(-HW - 0.15, DECK_Y, -0.1), (HW + 0.15, DECK_Y, -0.1), (HW + 0.15, DECK_Y, L + 0.15), (-HW - 0.15, DECK_Y, L + 0.15)],
             SHADOW, 0.24)


def _rail_run(frame, cam, p0, p1, posts):
    """One run of railing between two floor points: posts, a mid rail and a top rail."""
    (x0, z0), (x1, z1) = p0, p1
    geo.line(frame, cam, (x0, RAIL_MID_Y, z0), (x1, RAIL_MID_Y, z1), 0.05, RAIL_SHADE)
    for k in range(posts + 1):
        u = k / posts
        x, z = x0 + (x1 - x0) * u, z0 + (z1 - z0) * u
        geo.line(frame, cam, (x, DECK_Y, z), (x, RAIL_TOP_Y, z), 0.07, RAIL)
    geo.line(frame, cam, (x0, RAIL_TOP_Y, z0), (x1, RAIL_TOP_Y, z1), 0.08, RAIL)


def rails_far(frame, cam):
    """The railing behind the far end of the table and along both sides."""
    _rail_run(frame, cam, (-DECK_X, DECK_Z1), (DECK_X, DECK_Z1), 10)
    for side in (-1, 1):
        _rail_run(frame, cam, (side * DECK_X, DECK_Z0), (side * DECK_X, DECK_Z1), 9)


def rails_near(frame, cam):
    """The railing on the side the walkway comes in: in front of everything, so the intro draws it last."""
    _rail_run(frame, cam, (-DECK_X, DECK_Z0), (-1.9, DECK_Z0), 3)
    _rail_run(frame, cam, (1.9, DECK_Z0), (DECK_X, DECK_Z0), 3)


def _bunting_run(frame, cam, p0, p1, t_s):
    (x0, z0), (x1, z1) = p0, p1
    length = math.hypot(x1 - x0, z1 - z0)
    ax, az = (x1 - x0) / length, (z1 - z0) / length
    y_top, flags = RAIL_TOP_Y + 0.30, int(length / 0.3)
    sag = lambda u: 0.13 * 4 * ((u * length) % 1.0) * (1 - (u * length) % 1.0)        # noqa: E731  (one sag per metre between posts)
    string = [(x0 + (x1 - x0) * u, y_top - sag(u), z0 + (z1 - z0) * u) for u in np.linspace(0, 1, flags * 2 + 1)]
    for a, b in zip(string, string[1:]):
        geo.line(frame, cam, a, b, 0.012, (70, 70, 80))
    for k in range(flags):
        u = (k + 0.5) / flags
        x, y, z = x0 + (x1 - x0) * u, y_top - sag(u), z0 + (z1 - z0) * u
        wave = 0.04 * math.sin(t_s * 3.0 + k)
        geo.fill(frame, cam, [(x - 0.11 * ax, y, z - 0.11 * az), (x + 0.11 * ax, y, z + 0.11 * az),
                              (x + wave * ax, y - 0.25, z + wave * az)], BUNTING[k % len(BUNTING)])


def bunting(frame, cam, t_s):
    _bunting_run(frame, cam, (-DECK_X, DECK_Z1), (DECK_X, DECK_Z1), t_s)
    for side in (-1, 1):
        _bunting_run(frame, cam, (side * DECK_X, DECK_Z1), (side * DECK_X, DECK_Z0), t_s)
