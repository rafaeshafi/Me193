"""Drawing helpers for the resort: filled polygons and thick lines in world metres through any camera, rings for islands,
and extruded shapes ("prisms") whose silhouette is the hull of their bottom and top rings."""

import math

import cv2
import numpy as np


def _int(points):
    return np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)


def fill(frame, cam, points, color, alpha=1.0):
    """A polygon given in world points (x, y, z); alpha < 1 blends it over what is there."""
    pts = cam.project_poly(points)
    if len(pts) < 3:
        return
    pts = np.clip(np.asarray(pts), -30000, 30000)
    if alpha >= 1.0:
        cv2.fillPoly(frame, [_int(pts)], tuple(color), cv2.LINE_AA)
        return
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(pts[:, 0].min()) - 2), max(0, int(pts[:, 1].min()) - 2)
    x1, y1 = min(w, int(pts[:, 0].max()) + 3), min(h, int(pts[:, 1].max()) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.fillPoly(layer, [_int(pts - np.array([x0, y0]))], tuple(color), cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)


def line(frame, cam, p0, p1, width_m, color, min_px=1):
    """A straight bar of the given width in metres between two world points (thinner and thinner as it recedes)."""
    seg = cam.project_segment(p0, p1)
    if seg is None:
        return
    mid = tuple((a + b) / 2 for a, b in zip(p0, p1))
    scale = cam.focal / max(0.5, cam.depth_of(*mid))
    (xa, ya), (xb, yb) = np.clip(np.asarray(seg), -30000, 30000)
    cv2.line(frame, (round(xa), round(ya)), (round(xb), round(yb)), tuple(color), max(min_px, round(width_m * scale)), cv2.LINE_AA)


def circle_ring(cx, cz, r, n=28):
    return [(cx + r * math.cos(2 * math.pi * k / n), cz + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def blob(cx, cz, r, seed, n=30, jitter=0.10):
    """An irregular, roundish outline (an island's shore) as (x, z) points; the same seed, the same shore."""
    rnd = np.random.default_rng(seed)
    wobble = rnd.uniform(-jitter, jitter, 5)
    ring = []
    for k in range(n):
        a = 2 * math.pi * k / n
        bump = sum(wobble[j] * math.sin((j + 2) * a + j) for j in range(5))
        ring.append((cx + r * (1 + bump) * math.cos(a), cz + r * (1 + bump) * math.sin(a)))
    return ring


def prism(frame, cam, ring, y0, y1, side, top):
    """An extruded shape from y0 up to y1: its walls (the hull of the bottom and top rings) in `side`, its top in `top`."""
    low = cam.project_poly([(x, y0, z) for x, z in ring])
    high = cam.project_poly([(x, y1, z) for x, z in ring])
    both = np.clip(np.asarray(low + high, dtype=np.float32), -30000, 30000)
    if len(both) < 3:
        return
    cv2.fillPoly(frame, [_int(cv2.convexHull(both)[:, 0, :])], tuple(side), cv2.LINE_AA)
    if cam.pos[1] > y1 and len(high) >= 3:
        cv2.fillPoly(frame, [_int(np.clip(np.asarray(high), -30000, 30000))], tuple(top), cv2.LINE_AA)


def mix(a, b, t):
    """Colour a blended towards colour b by t (0..1)."""
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def hazed(color, distance_m, haze, scale_m=700.0):
    """A colour as seen through the air at this distance: washed towards the haze colour."""
    return mix(color, haze, 1.0 - math.exp(-max(0.0, distance_m) / scale_m))
