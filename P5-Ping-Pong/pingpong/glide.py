"""The hand as the screen draws it, smoothed between the camera's readings.  Display only: the game, the judge and the contact test
use the readings themselves.

The camera gives a hand reading about 30 times a second and the screen draws 60, so a hand held where it was last read moves in steps,
one every other picture.  On the player's real recordings the velocity of a hand drawn that way wobbles by 2.2 shoulder widths a second:
a paddle that judders.  Two one-pole filters in a row (each of half the time constant), updated exactly for the time between two
pictures (so 60 or 120 a second draw the same hand), took 61% of that out at 25 ms for 16 ms more delay than a held reading has (a held
reading is on average half an interval old already).  Longer smoothing took out more and cost more: 30 ms removed 66% for 20 ms, 40 ms
74% for 30, 50 ms 79% for 38.
"""

import math


class Glide:
    SNAP_S = 0.25                  # after a silence this long the hand is drawn where it is, not swept there from where it was
    JUMP = 1.5                     # ... and so is a hand that is further than this (shoulder widths) from where the glide was

    def __init__(self, tau_s=0.025):
        self.tau_s = tau_s
        self._stage = self._t_ns = self._out = None

    def update(self, t_ns, target):
        """The hand to draw at t_ns, given the reading to draw it at: a tuple of floats, or None for a hand that is not seen.
        Asking again for the same moment gives the same answer."""
        if target is None:
            self._stage = self._t_ns = self._out = None
            return None
        if self.tau_s <= 0:
            return tuple(target)
        if self._t_ns is not None and t_ns == self._t_ns:
            return self._out
        dt = None if self._t_ns is None else (t_ns - self._t_ns) / 1e9
        if dt is None or dt < 0 or dt > self.SNAP_S or math.dist(self._out, target) > self.JUMP:
            self._stage, self._out = tuple(target), tuple(target)
        else:
            a = dt / (self.tau_s / 2)                      # exact for a reading that holds over the step: two equal poles
            e = math.exp(-a)
            self._out = tuple(x + ((o - x) + (s - x) * a) * e for o, s, x in zip(self._out, self._stage, target))
            self._stage = tuple(x + (s - x) * e for s, x in zip(self._stage, target))
        self._t_ns = t_ns
        return self._out
