"""The body as the hand's reference: where the shoulders are and how big a shoulder width is, steady and robust.

The hand is measured from the shoulders, in shoulder widths, so that it does not depend on how far the player stands.
Reading the shoulders afresh in every frame has three faults the first live games showed: their noise is added to the
hand's, a width captured once at calibration is the wrong unit for a player who stands somewhere else (a lock built on it
threw away 20-39% of the frames, mostly one at a time), and a stroke turns the torso so the shoulders look narrow just
when the hand moves fastest.  BodyTracker follows the shoulder midpoint smoothly, takes the unit from the upper part of
the recent widths (a turn narrows the shoulders for a moment, a step towards the camera widens them for good), holds the
anchor through a hidden shoulder, and refuses only a body that really is someone else: one that jumps, or that is a
very different size and stays that way for a second (then it is accepted: nobody is locked out for good).

HandTracker places the hand at the wrist plus the fist's offset from it, and keeps the last offset of a finger point
that is not seen, so the hand does not jump each time a point flickers in and out of sight.
"""

import math
from collections import deque
from dataclasses import dataclass

from pingpong import pose

HELD_CONF = 0.7             # how sure a held anchor is (the judge wants 0.6)
STALE_S = 3.0               # a body not accepted for this long is forgotten: the next one is a new body


@dataclass(frozen=True)
class Body:
    cx: float               # the shoulder midpoint, a fraction of the frame width
    cy: float               # ... of the frame height
    unit: float             # one shoulder width, a fraction of the frame width
    conf: float             # 0..1
    held: bool = False      # the shoulders were not seen: this is where they were


class BodyTracker:
    def __init__(self, ref=None, *, window_s=2.0, percentile=80.0, lo=0.5, hi=1.6, hold_s=1.0, jump=0.08,
                 reacquire_s=1.0, tau_s=0.06):
        self.ref, self.window_s, self.percentile = ref, window_s, percentile
        self.lo, self.hi, self.hold_s, self.jump = lo, hi, hold_s, jump
        self.reacquire_s, self.tau_s = reacquire_s, tau_s
        self.counts = {"ok": 0, "held": 0, "rejected": 0, "no_shoulders": 0, "size": 0, "jump": 0}
        self.reason = ""                      # why the last frame was refused or held
        self.last_width = None                # the last frame's shoulder width, for the diagnostics
        self.reset()

    def reset(self):
        self._cx = self._cy = self._unit = self._t_last = None
        self._conf = 0.0
        self._widths = deque()
        self._alt = None                      # a different body that keeps being seen: [t_first, t_prev, cx, cy, width]

    # --- one frame --------------------------------------------------------------------------------------------------
    def update(self, lm, t_s, aspect):
        """The Body for this frame, or None.  `aspect` is the frame's height over its width."""
        left, right = lm[pose.L_SHOULDER], lm[pose.R_SHOULDER]
        if min(left.visibility, right.visibility) < pose.MIN_VISIBILITY:
            return self._hold(t_s, "no_shoulders")
        w = math.hypot(left.x - right.x, (left.y - right.y) * aspect)
        cx, cy, conf = (left.x + right.x) / 2, (left.y + right.y) / 2, min(left.visibility, right.visibility)
        self.last_width = w
        if self._unit is not None and t_s - self._t_last > STALE_S:
            self.reset()
        if self._unit is None:                                    # nobody yet: compare with the calibrated width, if any
            if self.ref is not None and not self.lo * self.ref <= w <= self.hi * self.ref:
                return self._refuse("size", t_s, cx, cy, w, conf)
            return self._accept(t_s, cx, cy, w, conf, count_width=True)
        dist = math.hypot(cx - self._cx, (cy - self._cy) * aspect)
        if dist > self.jump + 0.25 * (t_s - self._t_last):
            return self._refuse("jump", t_s, cx, cy, w, conf)
        ratio = w / self._unit
        if ratio > self.hi:
            return self._refuse("size", t_s, cx, cy, w, conf)
        return self._accept(t_s, cx, cy, w, conf, count_width=ratio >= self.lo)   # (a turned torso is no measure of size)

    def _accept(self, t_s, cx, cy, w, conf, *, count_width):
        self._alt = None
        if self._cx is None:
            self._cx, self._cy = cx, cy
        else:
            a = 1.0 - math.exp(-max(0.0, t_s - self._t_last) / self.tau_s)
            self._cx, self._cy = self._cx + a * (cx - self._cx), self._cy + a * (cy - self._cy)
        if count_width:
            self._widths.append((t_s, w))
        while self._widths and self._widths[0][0] < t_s - self.window_s:
            self._widths.popleft()
        self._unit = self._current_unit(w)
        self._t_last, self._conf = t_s, conf
        self.counts["ok"] += 1
        self.reason = ""
        return Body(self._cx, self._cy, self._unit, conf)

    def _current_unit(self, fallback):
        widths = sorted(wd for _, wd in self._widths)
        if not widths:
            return self._unit if self._unit is not None else fallback
        prior = [self.ref] * max(0, 6 - len(widths)) if self.ref is not None else []    # the calibrated width until the window fills
        widths = sorted(widths + prior)
        return widths[min(len(widths) - 1, int(self.percentile / 100.0 * len(widths)))]

    def _hold(self, t_s, reason):
        if self._unit is not None and t_s - self._t_last <= self.hold_s:
            self.counts["held"] += 1
            self.reason = reason
            return Body(self._cx, self._cy, self._unit, min(self._conf, HELD_CONF), held=True)
        self.counts[reason] += 1
        self.reason = reason
        return None

    def _refuse(self, reason, t_s, cx, cy, w, conf):
        """A body that is not the one being followed; if it stays where it is for reacquire_s it is the player now."""
        alt = self._alt
        if alt is not None and t_s - alt[1] <= 0.2 and math.hypot(cx - alt[2], cy - alt[3]) <= 0.05 \
                and abs(w / alt[4] - 1.0) <= 0.35:
            alt[1:] = [t_s, cx, cy, w]
        else:
            alt = self._alt = [t_s, t_s, cx, cy, w]
        if t_s - alt[0] >= self.reacquire_s:
            self.reset()
            return self._accept(t_s, cx, cy, w, conf, count_width=True)
        self.counts["rejected"] += 1
        self.counts[reason] += 1
        self.reason = reason
        return None


class HandTracker:
    def __init__(self, hand="right", hold_s=1.0):
        self.idx, self.hold_s = pose.HANDS[hand], hold_s
        self._offsets = {}                    # part -> (dx, dy, when seen): the fist's points, from the wrist

    def update(self, lm, t_s):
        """(x, y, confidence) of the hand in image fractions, or None when the wrist is not seen."""
        wrist = lm[self.idx["wrist"]]
        if wrist.visibility < pose.MIN_VISIBILITY:
            return None
        for part in pose.HAND_WEIGHTS:
            point = lm[self.idx[part]]
            if point.visibility >= pose.MIN_VISIBILITY:
                self._offsets[part] = (point.x - wrist.x, point.y - wrist.y, t_s)
        live = {part: (dx, dy) for part, (dx, dy, seen) in self._offsets.items() if t_s - seen <= self.hold_s}
        if not live:
            return wrist.x, wrist.y, wrist.visibility
        total = sum(pose.HAND_WEIGHTS[part] for part in live)
        ox = sum(pose.HAND_WEIGHTS[part] * dx for part, (dx, _) in live.items()) / total
        oy = sum(pose.HAND_WEIGHTS[part] * dy for part, (_, dy) in live.items()) / total
        return wrist.x + ox, wrist.y + oy, wrist.visibility


def hand_uv(body, hx, hy, aspect):
    """The hand in shoulder widths from the body's anchor: u positive to the player's right, v positive up."""
    return (body.cx - hx) / body.unit, (body.cy - hy) * aspect / body.unit
