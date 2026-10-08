"""The table in perspective: a camera behind the player's end, above the table, looking down it.

World metres as in physics.py: x across the table (positive to the player's right), y up from the table top, z along the
table from the player's edge (0) to the computer's (2.74).  project() is a plain pinhole camera, so sizes shrink and
positions converge towards the far end the way they do in an arcade table tennis game, and a shadow on the table under
anything in the air shows how high it is.
"""

import math
from dataclasses import dataclass

REF_W, REF_H = 1280, 720
POS = (0.0, 1.4, -1.8)           # the camera: behind the player's edge, 1.4 m above the table top
PITCH_DEG = 28.0                 # looking down this far
FOCAL_PX = 950.0                 # at 1280 px wide
CENTRE_Y = 392.0                 # the optical centre's row at 720 px high (puts the far edge at 210 and your edge at 558)


NEAR = 0.3                       # polygons are cut where they come closer to the camera than this (metres)


@dataclass(frozen=True)
class Camera:
    cx: float
    cy: float
    focal: float
    pos: tuple = POS
    pitch_deg: float = PITCH_DEG
    yaw_deg: float = 0.0         # turned to the right of straight down the table by this much (the intro flies the camera)

    @classmethod
    def for_frame(cls, w, h):
        k = w / REF_W
        return cls(cx=w / 2, cy=CENTRE_Y * h / REF_H, focal=FOCAL_PX * k)

    def view(self, x, y, z):
        """A world point in the camera's own frame: (right, up, depth), metres."""
        rx, ry, rz = x - self.pos[0], y - self.pos[1], z - self.pos[2]
        if self.yaw_deg:
            yaw = math.radians(self.yaw_deg)
            rx, rz = rx * math.cos(yaw) - rz * math.sin(yaw), rx * math.sin(yaw) + rz * math.cos(yaw)
        pitch = math.radians(self.pitch_deg)
        return rx, ry * math.cos(pitch) + rz * math.sin(pitch), -ry * math.sin(pitch) + rz * math.cos(pitch)

    def view_many(self, x, y, z):
        """`view` for numpy arrays of points: (right, up, depth) arrays."""
        import numpy as np

        rx, ry, rz = x - self.pos[0], y - self.pos[1], z - self.pos[2]
        if self.yaw_deg:
            yaw = math.radians(self.yaw_deg)
            rx, rz = rx * math.cos(yaw) - rz * math.sin(yaw), rx * math.sin(yaw) + rz * math.cos(yaw)
        pitch = math.radians(self.pitch_deg)
        return rx, ry * np.cos(pitch) + rz * np.sin(pitch), -ry * np.sin(pitch) + rz * np.cos(pitch)

    def depth_of(self, x, y, z):
        """How far in front of the camera a point is (negative: behind it)."""
        return self.view(x, y, z)[2]

    def project(self, x, y, z):
        """World (x, y, z) -> (px, py, scale): the pixel, and how many pixels one metre is across at that depth."""
        rx, up, depth = self.view(x, y, z)
        scale = self.focal / max(0.2, depth)
        return self.cx + rx * scale, self.cy - up * scale, scale

    def _screen(self, point):
        rx, up, depth = point
        return self.cx + rx * self.focal / depth, self.cy - up * self.focal / depth

    def project_poly(self, points, near=NEAR):
        """A polygon in the world -> its screen points, cut where it crosses the near plane (none if it is all behind us)."""
        view = [self.view(*p) for p in points]
        kept = []
        for a, b in zip(view, view[1:] + view[:1]):
            a_in, b_in = a[2] >= near, b[2] >= near
            if a_in:
                kept.append(a)
            if a_in != b_in:
                t = (near - a[2]) / (b[2] - a[2])
                kept.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near))
        return [self._screen(p) for p in kept]

    def project_segment(self, p0, p1, near=NEAR):
        """A line in the world -> its two screen points, cut at the near plane; None if it is all behind us."""
        a, b = self.view(*p0), self.view(*p1)
        if a[2] < near and b[2] < near:
            return None
        if a[2] < near or b[2] < near:
            inside, outside = (b, a) if a[2] < near else (a, b)
            t = (near - outside[2]) / (inside[2] - outside[2])
            cut = (outside[0] + (inside[0] - outside[0]) * t, outside[1] + (inside[1] - outside[1]) * t, near)
            a, b = (cut, inside) if a[2] < near else (inside, cut)
        return self._screen(a), self._screen(b)

    def ground_ellipse(self, x, z, rx, rz, n=36):
        """An ellipse lying on the table top around (x, z), rx across and rz along it, as screen points."""
        return [self.project(x + rx * math.cos(2 * math.pi * k / n), 0.0, z + rz * math.sin(2 * math.pi * k / n))[:2]
                for k in range(n)]

    def ground_circle(self, x, z, radius, n=28):
        """A circle of this radius lying on the table top around (x, z), as screen points (it looks like an ellipse)."""
        return self.ground_ellipse(x, z, radius, radius, n)
