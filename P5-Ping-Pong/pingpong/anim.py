"""Easing curves for the menus and the intro: pure functions of a progress 0..1 (held at the ends outside it)."""

import math


def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def lerp(a, b, t):
    return a + (b - a) * t


def progress(t_s, *, start, length):
    """How far along a step that begins at `start` seconds and lasts `length` is at t_s, as 0..1 (a step of no length is over)."""
    if length <= 0:
        return 1.0 if t_s >= start else 0.0
    return clamp01((t_s - start) / length)


def smoothstep(t):
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t):
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t):
    t = clamp01(t)
    return 4.0 * t ** 3 if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out_back(t, overshoot=1.70158):
    """Runs past the end and settles back on it: the spring of a thing arriving."""
    t = clamp01(t) - 1.0
    return 1.0 + (overshoot + 1.0) * t ** 3 + overshoot * t ** 2


def pop(t):
    """A button or a banner popping in: 0 -> about 1.2 -> 1."""
    return ease_out_back(t, 2.6)


def bob(t_s, *, period_s, amplitude):
    """A gentle up-and-down: amplitude * sin, starting from nothing."""
    return amplitude * math.sin(2.0 * math.pi * t_s / period_s)
