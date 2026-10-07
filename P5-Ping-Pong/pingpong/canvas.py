"""Drawing primitives and the court projection (numpy BGR frames, no window)."""

import cv2
import numpy as np

HALF_WIDTH_M = 0.7625
FONT = cv2.FONT_HERSHEY_SIMPLEX


def project(x_m, depth, h_m, w, h):
    """(lateral metres, depth 0=far/CPU .. 1=near/player, height metres) -> (px, py, scale)."""
    far_y, near_y = 0.30 * h, 0.86 * h
    scale = 0.38 + 0.62 * depth
    px = w / 2 + (x_m / HALF_WIDTH_M) * 0.40 * w * scale
    py = far_y + depth * (near_y - far_y) - h_m * 0.55 * h * scale
    return int(px), int(py), scale


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


def draw_court(frame):
    h, w = frame.shape[:2]
    corners = [canvas_pt(-HALF_WIDTH_M, 0.0, w, h), canvas_pt(HALF_WIDTH_M, 0.0, w, h),
               canvas_pt(HALF_WIDTH_M, 1.0, w, h), canvas_pt(-HALF_WIDTH_M, 1.0, w, h)]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], (86, 52, 20))
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.polylines(frame, [np.array(corners, dtype=np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, canvas_pt(0, 0.0, w, h), canvas_pt(0, 1.0, w, h), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.line(frame, canvas_pt(-HALF_WIDTH_M * 1.08, 0.5, w, h), canvas_pt(HALF_WIDTH_M * 1.08, 0.5, w, h),
             (230, 230, 230), 4, cv2.LINE_AA)                                  # the net


def canvas_pt(x_m, depth, w, h, h_m=0.0):
    px, py, _ = project(x_m, depth, h_m, w, h)
    return px, py


def draw_ball(frame, x_m, depth, h_m, color=(40, 160, 255)):
    h, w = frame.shape[:2]
    gx, gy, _ = project(x_m, depth, 0.0, w, h)
    bx, by, scale = project(x_m, depth, h_m, w, h)
    cv2.ellipse(frame, (gx, gy), (int(16 * scale), int(6 * scale)), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (bx, by), int(8 + 14 * scale), color, -1, cv2.LINE_AA)
    cv2.circle(frame, (bx, by), int(8 + 14 * scale), (255, 255, 255), 2, cv2.LINE_AA)


def draw_ring(frame, x_m, depth, h_m, radius, color, thickness=3):
    h, w = frame.shape[:2]
    px, py, scale = project(x_m, depth, h_m, w, h)
    cv2.circle(frame, (px, py), int(radius * scale), color, thickness, cv2.LINE_AA)


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


def dim_rect(frame, x0, y0, w, h, factor=0.35):
    """Darken a rectangle so text on top stays readable over the court or the camera picture."""
    region = frame[y0:y0 + h, x0:x0 + w]
    region[:] = (region * factor).astype(np.uint8)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (170, 170, 170), 1)
