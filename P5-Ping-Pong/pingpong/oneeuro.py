"""One-Euro filter (Casiez et al.): smooth when slow, nearly lag-free when fast.

A low-pass filter whose cutoff rises with the signal's speed, so a still hand is
steady and a swinging hand is not delayed -- which matters because pose is already
70-150 ms late.  Defaults come from the plan: min_cutoff 1.2 Hz, beta 5, d_cutoff 1 Hz.
"""

import math


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=5.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t_s):
        if self._t is None:
            self._x, self._dx, self._t = x, 0.0, t_s
            return x
        dt = t_s - self._t
        if dt < 0:
            raise ValueError("timestamps must not go backwards")
        if dt == 0:
            return self._x
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t_s
        return self._x


class OneEuro2D:
    def __init__(self, **kw):
        self._fx, self._fy = OneEuro(**kw), OneEuro(**kw)

    def reset(self):
        self._fx.reset()
        self._fy.reset()

    def __call__(self, xy, t_s):
        return self._fx(xy[0], t_s), self._fy(xy[1], t_s)
