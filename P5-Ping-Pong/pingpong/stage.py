"""Where your paddle stands in the scene and how a stroke moves it.

Your hand's position across the reach box is where the paddle stands across the table; its height is how far up the
table the paddle stands (raise the hand to meet the ball earlier, hang back to meet it later, like the arcade original's
joystick up and down).  A swing is a lunge: the paddle reaches out to meet the ball wherever the ball is at the contact
(a hit) or straight ahead (a swing at nothing), holds a moment, and comes back to where your hand is.
"""

from dataclasses import dataclass

from pingpong import physics

Z_REST_MIN, Z_REST_MAX = 0.05, 0.55      # the paddle stands this far from your edge when the hand is lowest / highest
PADDLE_Y_M = 0.16                        # the face's centre above the table
PADDLE_MIN_Y_M, PADDLE_MAX_Y_M = 0.08, 0.50
A_RANGE = (-0.1, 1.1)                    # the paddle may hang a little over the edge of the table
MISS_LUNGE = (0.0, 0.08, 0.40)           # a swing at nothing reaches this far ahead (across, up, along)
MISS_LUNGE_S = 0.12
SNAP_S = 0.04                            # a contact that is already due is met this quickly
HOLD_S, RECOVER_S = 0.05, 0.25


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rest_z(box, v):
    """How far up the table the paddle stands for a hand at height v."""
    b = _clamp((v - box.v_min) / (box.v_max - box.v_min), 0.0, 1.0)
    return Z_REST_MIN + b * (Z_REST_MAX - Z_REST_MIN)


def rest_position(box, u, v):
    """The paddle's (x, y, z) in metres for a hand at (u, v) shoulder widths."""
    a = _clamp(box.to_ab(u, v)[0], *A_RANGE)
    return physics.x_of_a(a), PADDLE_Y_M, rest_z(box, v)


def _ease(x):
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lerp(a, b, s):
    return tuple(p + (q - p) * s for p, q in zip(a, b))


@dataclass(frozen=True)
class Stroke:
    t0_ns: int            # the swing was seen: the lunge starts
    contact_ns: int       # the paddle meets the ball (a hit) or is fully stretched (a swing at nothing)
    target: tuple         # (x, y, z): where it reaches
    hit: bool = False

    @property
    def end_ns(self):
        return self.contact_ns + round((HOLD_S + RECOVER_S) * 1e9)

    def done(self, t_ns):
        return t_ns >= self.end_ns

    def pose(self, rest, t_ns):
        """The paddle at t_ns, given where the hand has it (its rest place) at that moment."""
        hold_end = self.contact_ns + round(HOLD_S * 1e9)
        if t_ns <= self.t0_ns:
            return rest
        if t_ns < self.contact_ns:
            return _lerp(rest, self.target, _ease((t_ns - self.t0_ns) / (self.contact_ns - self.t0_ns)))
        if t_ns < hold_end:
            return self.target
        if t_ns < self.end_ns:
            return _lerp(self.target, rest, _ease((t_ns - hold_end) / (RECOVER_S * 1e9)))
        return rest


def hit_stroke(t0_ns, contact_ns, ball):
    """The lunge that meets the ball (x, y, z) at the contact."""
    x, y, z = ball
    return Stroke(t0_ns, max(contact_ns, t0_ns + round(SNAP_S * 1e9)), (x, _clamp(y, PADDLE_MIN_Y_M, PADDLE_MAX_Y_M), z),
                  hit=True)


def miss_stroke(rest, t0_ns):
    """The lunge of a swing that hits nothing."""
    return Stroke(t0_ns, t0_ns + round(MISS_LUNGE_S * 1e9), tuple(r + d for r, d in zip(rest, MISS_LUNGE)))


def zone(leg, sweet_ns, level):
    """The stretch of table (far z, near z) over which a ball can be hit: where it is a level's early window before
    the moment it passes your paddle, and where it is a late window after it."""
    z_far = leg.position(sweet_ns - round(level.early_s * 1e9))[2]
    z_near = leg.position(sweet_ns + round(level.late_s * 1e9))[2]
    return z_far, z_near
