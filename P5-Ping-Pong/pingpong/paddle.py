"""Paddle (hand) coordinates: shoulder-width units <-> the calibrated reach box.

The pose gives the hand as (u, v) in shoulder widths relative to the shoulder
midpoint (distance-invariant).  Calibration records the four corners of the
area the player can comfortably reach; the game addresses it as (a, b) in 0..1.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReachBox:
    u_min: float
    u_max: float
    v_min: float
    v_max: float

    def __post_init__(self):
        if not (self.u_max > self.u_min and self.v_max > self.v_min):
            raise ValueError("reach box must have positive width and height")

    def to_ab(self, u, v):
        return ((u - self.u_min) / (self.u_max - self.u_min),
                (v - self.v_min) / (self.v_max - self.v_min))

    def to_uv(self, a, b):
        return (self.u_min + a * (self.u_max - self.u_min),
                self.v_min + b * (self.v_max - self.v_min))

    @classmethod
    def fit(cls, corners, margin=0.05):
        """Box around the corner samples, padded by `margin` x the half-span on each side."""
        us = [c[0] for c in corners]
        vs = [c[1] for c in corners]
        pad_u = margin * (max(us) - min(us)) / 2
        pad_v = margin * (max(vs) - min(vs)) / 2
        return cls(min(us) - pad_u, max(us) + pad_u, min(vs) - pad_v, max(vs) + pad_v)
