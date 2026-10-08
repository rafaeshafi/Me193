"""Pointing with the hub and holding on a button to press it: the engine behind every menu and the START button.

Your hand is a cursor over the whole screen: its place in the calibrated reach box, (a, b) in 0..1 across and up, is where the
cursor stands, so reaching to the top right of your reach is reaching the top right of the screen.  A target is a rectangle of
the screen (fractions of it, y down) with the seconds it takes to press it.  Holding the hand on one for that long presses it.

The rules that make that safe at 1.8 m from the screen, where a reading flickers and a hand rests wherever it was:
  * the hand must have been seen somewhere else since the screen came up before a hold counts (a hand that ended the last
    game up in the corner must not press the corner button of the next screen);
  * the hand may drop out, or its reading, for GRACE_S without the hold starting over;
  * a stall between two updates is not time held;
  * a press takes the arming away again, so one hold is one press.
Pure logic with the clock passed in.
"""

GRACE_S = 0.3             # the hand (or its reading) may drop out this long without the hold starting over
MAX_STEP_S = 0.1          # a stall between two updates is not time held


def hit_test(ab, targets):
    """The id of the first target the hand points at, or None (also when the hand is not seen).  A hand beyond the edge of
    its reach counts as at the edge, so a button in a corner can always be reached."""
    if ab is None:
        return None
    x, y = min(1.0, max(0.0, ab[0])), 1.0 - min(1.0, max(0.0, ab[1]))
    for target_id, (x0, y0, x1, y1), *_ in targets:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return target_id
    return None


class Dwell:
    """Holding on one thing: the clock of a hold, with its arming and its grace."""

    def __init__(self, hold_s, grace_s=GRACE_S):
        self.hold_s = hold_s
        self.grace_ns, self.max_step_ns = round(grace_s * 1e9), round(MAX_STEP_S * 1e9)
        self.reset()

    def reset(self):
        self._t = None
        self._held_ns = self._away_ns = 0
        self.armed = False            # the hand has been somewhere else since the screen came up: only then is a hold on purpose
        self.inside = False

    def restart_hold(self):
        """The hold starts again from nothing (the hand slid from one target to the next); the arming stays."""
        self._held_ns = self._away_ns = 0

    def progress(self):
        return min(1.0, self._held_ns / max(1, round(self.hold_s * 1e9)))

    def step(self, now_ns, inside, seen, active, hold_s=None):
        """-> True at the moment the hold is complete.  inside: the hand is on the thing; seen: there is a hand reading at all;
        active: the screen is one the thing works on."""
        if hold_s is not None:
            self.hold_s = hold_s
        if not active:
            self.reset()
            return False
        dt = 0 if self._t is None else max(0, now_ns - self._t)
        self._t = now_ns
        self.inside = inside
        if not inside:
            self.armed = self.armed or seen
            self._away_ns += dt
            if self._away_ns > self.grace_ns:
                self._held_ns = 0
            return False
        self._away_ns = 0
        if not self.armed:
            return False
        self._held_ns += min(dt, self.max_step_ns)
        if self._held_ns >= round(self.hold_s * 1e9):
            self.reset()
            return True
        return False


class Pointer:
    """A hand over a screen of buttons: which one it is on, how far the hold has come, and when one is pressed."""

    def __init__(self, grace_s=GRACE_S):
        self._dwell = Dwell(1.0, grace_s)
        self.hovered = None

    @property
    def progress(self):
        return self._dwell.progress()

    @property
    def armed(self):
        return self._dwell.armed

    def reset(self):
        """A new screen: nothing is held, and the hand has to be seen elsewhere before it can press anything."""
        self._dwell.reset()
        self.hovered = None

    def update(self, now_ns, ab, targets, active):
        """-> the id of the target pressed at this moment, else None.  ab: the hand's (a, b) in the reach box, None when it is
        not seen; targets: (id, (x0, y0, x1, y1), hold_s) of this screen; active: this screen takes presses."""
        if not active:
            self.reset()
            return None
        hovered = hit_test(ab, targets)
        if hovered is not None and self.hovered is not None and hovered != self.hovered:
            self._dwell.restart_hold()
        self.hovered = hovered
        hold_s = next((t[2] for t in targets if t[0] == hovered), None)
        return hovered if self._dwell.step(now_ns, hovered is not None, ab is not None, True, hold_s) else None
