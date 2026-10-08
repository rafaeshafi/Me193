"""A hand that moves into the ball hits it: where the paddle meets the ball, from the hand's readings and the ball's flight.

The paddle stands where the hand puts it: across the table by the hand's sideways place in the reach box, up the table by its
height (stage.rest_z: raise the hand to meet the ball sooner).  The ball comes down the table, its depth z falling.  It meets
the paddle the moment the two depths are equal, whichever of them moved there: a still paddle is met by the ball, a paddle
that moves up the table meets it sooner.  If the hand is level with the ball across the table at that moment (within the
level's radius) it is a hit; if not, the ball goes by.  How high the ball is does not matter (as it never did).
"""

from dataclasses import dataclass

from pingpong import physics, stage


@dataclass(frozen=True)
class Contact:
    t_ns: int            # the moment the ball's depth met the paddle's
    ball: tuple          # (x, y, z): where the ball was
    d_sw: float          # how far across the table the hand was from it, in shoulder widths
    hit: bool
    u: float = 0.0       # where the hand was then, in shoulder widths (the game aims the return from it)
    v: float = 0.0


def ball_u(box, x_m):
    """Where the ball is across the table, in the hand's own units (shoulder widths)."""
    return box.to_uv(physics.a_of_x(x_m), 0.5)[0]


def crossing(prev, cur, leg, box, radius_sw):
    """prev, cur: consecutive hand readings (t_ns, u, v).  -> a Contact when the ball's depth crossed the paddle's between
    them (the ball falling from in front of the paddle to behind it), else None."""
    zb0, zb1 = leg.position(prev[0])[2], leg.position(cur[0])[2]
    d0, d1 = zb0 - stage.rest_z(box, prev[2]), zb1 - stage.rest_z(box, cur[2])
    if not (d0 > 0.0 >= d1):
        return None
    f = d0 / (d0 - d1)
    t_ns = round(prev[0] + f * (cur[0] - prev[0]))
    u, v = prev[1] + f * (cur[1] - prev[1]), prev[2] + f * (cur[2] - prev[2])
    ball = leg.position(t_ns)
    d = abs(u - ball_u(box, ball[0]))
    return Contact(t_ns=t_ns, ball=ball, d_sw=d, hit=d <= radius_sw, u=u, v=v)
