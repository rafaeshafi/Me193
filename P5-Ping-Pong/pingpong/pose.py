"""Paddle point from MediaPipe pose landmarks.

(u, v) are in SHOULDER-WIDTH units relative to the shoulder midpoint, so they do not
depend on how far the player stands from the camera.  Frames go to MediaPipe
UNMIRRORED (so left/right are anatomical); u is defined positive toward the PLAYER'S
RIGHT and v positive UP, so the on-screen paddle moves the way the hand does.
The paddle point is the HAND: the index, pinky and thumb points, i.e. the fist that holds the hub (the
wrist, a few centimetres behind it, is only the fallback when no finger point is seen).  The wrist's visibility
still decides whether the arm is in view at all.
"""

import math

L_SHOULDER, R_SHOULDER = 11, 12
HANDS = {"right": {"wrist": 16, "index": 20, "pinky": 18, "thumb": 22},
         "left": {"wrist": 15, "index": 19, "pinky": 17, "thumb": 21}}
HAND_WEIGHTS = {"index": 0.4, "pinky": 0.4, "thumb": 0.2}       # the fist: a knuckle line either side, and the thumb
MIN_VISIBILITY = 0.5
MIN_SHOULDER_FRAC = 0.08       # shoulders narrower than this fraction of the frame are rejected


def _shoulders(lm):
    left, right = lm[L_SHOULDER], lm[R_SHOULDER]
    if min(left.visibility, right.visibility) < MIN_VISIBILITY:
        return None
    return left, right


def hand_xy(lm, hand):
    """The centre of the hand as (x, y) in image fractions: the visible ones of the index, pinky and thumb points;
    the wrist only when none of them is seen."""
    idx = HANDS[hand]
    total = px = py = 0.0
    for part, weight in HAND_WEIGHTS.items():
        point = lm[idx[part]]
        if point.visibility >= MIN_VISIBILITY:
            total += weight
            px += weight * point.x
            py += weight * point.y
    if total == 0.0:
        wrist = lm[idx["wrist"]]
        return wrist.x, wrist.y
    return px / total, py / total


def shoulder_width_norm(lm, width, height):
    """Shoulder width as a fraction of the frame width (None if a shoulder is not visible)."""
    shoulders = _shoulders(lm)
    if shoulders is None:
        return None
    left, right = shoulders
    return math.hypot((left.x - right.x) * width, (left.y - right.y) * height) / width


def paddle_uv(lm, hand, width, height, unit=None):
    """-> (u, v, confidence) or None when the frame is not worth reading.

    `unit` is the shoulder width, as a fraction of the frame width, that one unit of u and v is measured in: the
    player's calibrated width.  The width of the moment shrinks when the torso turns (every stroke), which would
    stretch the hand coordinates by 1 / cos(turn); None measures in the width of the moment (no calibration yet)."""
    shoulders = _shoulders(lm)
    if shoulders is None:
        return None
    left, right = shoulders
    if unit is None:
        sw_px = math.hypot((left.x - right.x) * width, (left.y - right.y) * height)
        if sw_px < MIN_SHOULDER_FRAC * width:
            return None                           # side-on or far away: it would amplify noise
    else:
        sw_px = unit * width                      # (a body too far from the calibrated width is the lock's to refuse)
    idx = HANDS[hand]
    wrist = lm[idx["wrist"]]
    if wrist.visibility < MIN_VISIBILITY:
        return None
    hx, hy = hand_xy(lm, hand)
    px, py = hx * width, hy * height
    cx = (left.x + right.x) / 2 * width
    cy = (left.y + right.y) / 2 * height
    conf = min(wrist.visibility, left.visibility, right.visibility)
    return (cx - px) / sw_px, (cy - py) / sw_px, conf


class PoseLock:
    """One player only: after calibration, a body whose shoulders look far wider or narrower than the calibrated
    width is not the player (a spectator walking into frame must not move the paddle).

    The tolerance is lopsided on purpose: turning the torso (every stroke, and a side-on stance) makes the shoulders
    look narrower, down to about half, while only stepping towards the camera makes them wider."""

    def __init__(self, lo=0.5, hi=1.3):
        self.lo, self.hi, self._ref = lo, hi, None

    @property
    def unit(self):
        """The calibrated shoulder width (a fraction of the frame width) the hand is measured in; None before it."""
        return self._ref

    def calibrate(self, shoulder_width):
        self._ref = shoulder_width

    def accepts(self, shoulder_width):
        if shoulder_width is None:
            return False
        if self._ref is None:
            return True
        return self.lo <= shoulder_width / self._ref <= self.hi
