"""The sky of the island: its gradient, the sun, drifting clouds, the sea out to the horizon with its sparkles, seagulls, and the
far islands, lighthouse and sailboats on the horizon.

The sky and the sea are painted a row at a time (a row of the picture is a fixed angle above or below the horizon, so its
colour and, for the sea, its distance follow from the camera's pitch and height alone), which is fast and right from any
height: the same code serves the game's view over the railing and the fly-in from the clouds."""

import math

import cv2
import numpy as np

from pingpong import resort_geo, ui
from pingpong.cast import rgb

SEA_Y = -4.0
SKY_TOP, SKY_HAZE = rgb(58, 156, 255), rgb(208, 238, 255)
SEA_NEAR, SEA_FAR = rgb(0, 170, 206), rgb(178, 233, 245)
HAZE_M = 380.0
CLOUD, CLOUD_SHADE = rgb(255, 255, 255), rgb(204, 226, 250)
GULL = rgb(70, 82, 104)
_RAW_SUN = (-0.45, 0.50, 0.74)
SUN_DIR = tuple(v / math.sqrt(sum(c * c for c in _RAW_SUN)) for v in _RAW_SUN)

_RNG = np.random.default_rng(11)
_SPARK_X, _SPARK_Z = _RNG.uniform(-260, 260, 240), _RNG.uniform(-260, 260, 240)
_SPARK_PHASE, _SPARK_RATE = _RNG.uniform(0, 2 * math.pi, 240), _RNG.uniform(1.2, 3.4, 240)
_CLOUDS = [(float(x), float(z), float(y), float(s)) for x, z, y, s in zip(
    _RNG.uniform(-400, 400, 30), _RNG.uniform(-300, 380, 30), _RNG.uniform(70, 135, 30), _RNG.uniform(10, 24, 30))]
PUFFS = ((-1.3, 0.05, 0.75), (-0.6, -0.30, 0.95), (0.25, -0.42, 1.10), (1.1, -0.15, 0.90), (1.7, 0.10, 0.65), (0.2, 0.12, 1.0))
GULLS = ((-60.0, 22.0, -60.0, 7.0, 3.0, 0.0), (30.0, 18.0, -40.0, -6.0, 4.0, 1.7), (-10.0, 30.0, -90.0, 8.0, 2.0, 3.1),
         (70.0, 14.0, -20.0, -7.5, 3.5, 4.4))
FAR_ISLANDS = ((-210.0, 330.0, 70.0, 42.0), (270.0, 430.0, 90.0, 60.0), (60.0, 560.0, 120.0, 38.0))
BOATS = ((-90.0, 150.0), (130.0, 260.0), (-40.0, 360.0))
HILL = rgb(76, 176, 122)


def _elevation(cam, h):
    """The angle above the horizon (radians, negative below) of every row of the picture."""
    rows = np.arange(h)
    return np.arctan((cam.cy - rows) / cam.focal) - math.radians(cam.pitch_deg)


def sky(frame, cam):
    elev = _elevation(cam, frame.shape[0])
    t = np.clip(elev / math.radians(55.0), 0.0, 1.0) ** 0.75
    top, haze = np.array(SKY_TOP, dtype=float), np.array(SKY_HAZE, dtype=float)
    frame[:] = (haze + (top - haze) * t[:, None]).astype(np.uint8)[:, None, :]


def sea(frame, cam, t_s):
    """The sea from the horizon to the bottom of the picture: turquoise near, washed out with distance, with slow wave bands."""
    elev = _elevation(cam, frame.shape[0])
    below = np.where(elev < -1e-4)[0]
    if below.size == 0:
        return
    dist = max(1.0, cam.pos[1] - SEA_Y) / np.maximum(1e-3, np.abs(np.tan(-elev[below])))
    haze = 1.0 - np.exp(-dist / HAZE_M)
    near, far = np.array(SEA_NEAR, dtype=float), np.array(SEA_FAR, dtype=float)
    rows = (near + (far - near) * haze[:, None]) * (1.0 + 0.035 * np.sin(dist * 0.45 - t_s * 1.6) * np.exp(-dist / 160.0))[:, None]
    frame[below] = np.clip(rows, 0, 255).astype(np.uint8)[:, None, :]


def sparkles(frame, cam, t_s):
    """Glints on the water near the camera, each twinkling on its own beat."""
    right, up, depth = cam.view_many(_SPARK_X, np.full_like(_SPARK_X, SEA_Y), _SPARK_Z)
    bright = (np.sin(t_s * _SPARK_RATE + _SPARK_PHASE) * 0.5 + 0.5) ** 5
    for k in np.where((depth > 12) & (depth < 450) & (bright > 0.3))[0]:
        scale = cam.focal / depth[k]
        x, y = cam.cx + right[k] * scale, cam.cy - up[k] * scale
        half = min(10.0, max(1.5, 0.8 * scale))
        if 0 <= y < frame.shape[0] and -10 <= x < frame.shape[1] + 10:
            cv2.line(frame, (round(x - half), round(y)), (round(x + half), round(y)), (255, 255, 255), 1, cv2.LINE_AA)


def sun_px(cam):
    """Where the sun is in the picture (even when that is off it), or None if it is behind the camera."""
    point = tuple(c + 1000.0 * d for c, d in zip(cam.pos, SUN_DIR))
    right, up, depth = cam.view(*point)
    if depth < 1.0:
        return None
    return cam.cx + right * cam.focal / depth, cam.cy - up * cam.focal / depth


def sun_elevation_deg():
    return math.degrees(math.asin(SUN_DIR[1]))


def sun(frame, cam):
    seen = sun_px(cam)
    if seen is None:
        return
    k = frame.shape[1] / 1280.0
    ui.glow(frame, seen[0], seen[1], 380 * k, rgb(255, 244, 200), 0.55)
    ui.glow(frame, seen[0], seen[1], 120 * k, rgb(255, 252, 232), 0.85)
    cv2.circle(frame, (round(seen[0]), round(seen[1])), max(3, round(30 * k)), rgb(255, 253, 228), -1, cv2.LINE_AA)


def clouds(frame, cam, t_s, *, far):
    """Puffy flat clouds drifting across; `far` ones are drawn before the island, near ones in front of it."""
    seen = []
    for x0, z0, y, size in _CLOUDS:
        x = ((x0 + 2.2 * t_s + 400.0) % 800.0) - 400.0
        right, up, depth = cam.view(x, y, z0)
        if depth > 90.0 and (depth >= 300.0) == far:
            seen.append((depth, cam.cx + right * cam.focal / depth, cam.cy - up * cam.focal / depth, size * cam.focal / depth))
    h, w = frame.shape[:2]
    for depth, cx, cy, s in sorted(seen, reverse=True):
        if s < 1.5 or cx < -3 * s or cx > w + 3 * s or cy < -2 * s or cy > h + 2 * s:
            continue
        for color, drop in ((CLOUD_SHADE, 0.2), (CLOUD, 0.0)):
            for dx, dy, r in PUFFS:
                cv2.circle(frame, (round(cx + dx * s), round(cy - dy * s + drop * r * s)), max(1, round(r * s)), color, -1, cv2.LINE_AA)


def birds(frame, cam, t_s):
    """A few seagulls crossing, wings flapping."""
    for x0, y0, z0, vx, vz, phase in GULLS:
        x, z = x0 + vx * (t_s % 40.0), z0 + vz * (t_s % 40.0)
        px, py, scale = cam.project(x, y0 + 0.8 * math.sin(t_s * 0.7 + phase), z)
        if cam.depth_of(x, y0, z) < 8.0 or scale < 1.5:
            continue
        s, flap = 0.5 * scale, 0.45 * math.sin(t_s * 9.0 + phase)
        wing = np.array([(px - 2 * s, py - s * (0.5 + flap)), (px - s, py - s * (0.15 + flap * 0.5)), (px, py),
                         (px + s, py - s * (0.15 + flap * 0.5)), (px + 2 * s, py - s * (0.5 + flap))])
        cv2.polylines(frame, [np.round(wing).astype(np.int32)], False, GULL, max(1, round(0.3 * s)), cv2.LINE_AA)


def _hill(frame, cam, cx, cz, r, height):
    d = math.hypot(cx - cam.pos[0], cz - cam.pos[2])
    for frac, up, shade in ((1.0, 0.0, 0.0), (0.72, 0.4, 0.12), (0.42, 0.75, 0.24)):
        ring = resort_geo.circle_ring(cx, cz, r * frac, 20)
        base = resort_geo.hazed(resort_geo.mix(HILL, (255, 255, 255), shade), d, SKY_HAZE, 900.0)
        resort_geo.prism(frame, cam, ring, SEA_Y + height * up * 0.5, SEA_Y + height * (up + 0.4), base, base)


def lighthouse(frame, cam, x, z):
    d = math.hypot(x - cam.pos[0], z - cam.pos[2])
    if cam.depth_of(x, SEA_Y + 30, z) < 5:
        return
    scale = cam.focal / max(1.0, cam.depth_of(x, SEA_Y + 30, z))
    bx, by, _ = cam.project(x, SEA_Y + 36, z)
    top = cam.project(x, SEA_Y + 70, z)
    for (y0, y1, color) in ((36, 50, rgb(250, 250, 250)), (50, 58, rgb(235, 70, 70)), (58, 66, rgb(250, 250, 250)),
                            (66, 70, rgb(255, 214, 90))):
        a, b = cam.project(x, SEA_Y + y0, z), cam.project(x, SEA_Y + y1, z)
        w = max(1.0, (7.0 - 0.07 * (y0 - 36)) * scale / 2)
        cv2.fillPoly(frame, [np.round(np.array([(a[0] - w, a[1]), (a[0] + w, a[1]), (b[0] + w * 0.9, b[1]), (b[0] - w * 0.9, b[1])]))
                     .astype(np.int32)], resort_geo.hazed(color, d, SKY_HAZE, 900.0), cv2.LINE_AA)


def boats(frame, cam, t_s):
    for k, (x, z) in enumerate(BOATS):
        bob = 0.25 * math.sin(t_s * 1.1 + k)
        px, py, scale = cam.project(x, SEA_Y + 0.4 + bob, z)
        if cam.depth_of(x, SEA_Y, z) < 15 or scale < 0.8:
            continue
        s = scale * 4.0
        cv2.fillPoly(frame, [np.round(np.array([(px - 1.6 * s, py), (px + 1.6 * s, py), (px + 1.1 * s, py + 0.55 * s),
                                                (px - 1.1 * s, py + 0.55 * s)])).astype(np.int32)], rgb(235, 90, 80), cv2.LINE_AA)
        cv2.fillPoly(frame, [np.round(np.array([(px, py - 0.1 * s), (px, py - 3.2 * s), (px + 1.3 * s, py - 0.1 * s)])).astype(np.int32)],
                     rgb(255, 255, 255), cv2.LINE_AA)
        cv2.fillPoly(frame, [np.round(np.array([(px - 0.2 * s, py - 0.1 * s), (px - 0.2 * s, py - 2.4 * s), (px - 1.1 * s, py - 0.1 * s)])).astype(np.int32)],
                     rgb(255, 228, 120), cv2.LINE_AA)


def far_scenery(frame, cam, t_s):
    """What lies out at sea: two or three green islands with a lighthouse, and sailboats."""
    for cx, cz, r, height in sorted(FAR_ISLANDS, key=lambda i: -math.hypot(i[0] - cam.pos[0], i[1] - cam.pos[2])):
        _hill(frame, cam, cx, cz, r, height)
    lighthouse(frame, cam, FAR_ISLANDS[0][0] + 20.0, FAR_ISLANDS[0][1] - 5.0)
    boats(frame, cam, t_s)
