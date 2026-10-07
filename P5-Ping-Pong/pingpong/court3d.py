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


@dataclass(frozen=True)
class Camera:
    cx: float
    cy: float
    focal: float
    pos: tuple = POS
    pitch_deg: float = PITCH_DEG

    @classmethod
    def for_frame(cls, w, h):
        k = w / REF_W
        return cls(cx=w / 2, cy=CENTRE_Y * h / REF_H, focal=FOCAL_PX * k)

    def project(self, x, y, z):
        """World (x, y, z) -> (px, py, scale): the pixel, and how many pixels one metre is across at that depth."""
        pitch = math.radians(self.pitch_deg)
        rx, ry, rz = x - self.pos[0], y - self.pos[1], z - self.pos[2]
        depth = max(0.2, -ry * math.sin(pitch) + rz * math.cos(pitch))
        up = ry * math.cos(pitch) + rz * math.sin(pitch)
        scale = self.focal / depth
        return self.cx + rx * scale, self.cy - up * scale, scale

    def ground_ellipse(self, x, z, rx, rz, n=36):
        """An ellipse lying on the table top around (x, z), rx across and rz along it, as screen points."""
        return [self.project(x + rx * math.cos(2 * math.pi * k / n), 0.0, z + rz * math.sin(2 * math.pi * k / n))[:2]
                for k in range(n)]

    def ground_circle(self, x, z, radius, n=28):
        """A circle of this radius lying on the table top around (x, z), as screen points (it looks like an ellipse)."""
        return self.ground_ellipse(x, z, radius, radius, n)
