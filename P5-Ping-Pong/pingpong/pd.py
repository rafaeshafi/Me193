"""PD control: the computer's paddle chases the predicted landing point (the class PD controller, P2).

The computer's paddle is not a coin flip.  When you hit, it needs its reaction time before it moves at all;
then a PD controller turns the distance still to cover into a velocity command, saturated at the level's
paddle speed:

    command = Kp * error + Kd * d(error)/dt          (the derivative is low-passed, like P2's D_FILTER)

Because the command is a velocity the paddle cannot overshoot; it either gets there or it does not.  What
is left of the distance when the ball arrives is the "reach deficit" that the Match miss model uses, and
paddle_x() is also what the screen draws, so you SEE the computer chase your shot and fall short of a
smash.  Gains Kp 15 / Kd 0.3 leave under 3 cm for a ball at half the paddle's reach, at every level.
"""

KP, KD, D_FILTER, DT = 15.0, 0.3, 0.5, 0.005


class PDController:
    def __init__(self, kp=KP, kd=KD, limit=float("inf"), d_filter=D_FILTER):
        self.kp, self.kd, self.limit, self.d_filter = kp, kd, limit, d_filter
        self.reset()

    def reset(self):
        self.de, self._prev = 0.0, None

    def step(self, error, dt):
        """One control step: the error (m) in, a velocity command (m/s, saturated) out."""
        if self._prev is not None:
            self.de += self.d_filter * ((error - self._prev) / dt - self.de)       # low-passed d(error)/dt
        self._prev = error
        return max(-self.limit, min(self.limit, self.kp * error + self.kd * self.de))


def paddle_x(level, x_land_m, x_cpu_m, t_s, dt=DT):
    """Where the computer's paddle is t_s seconds after you hit: still for the reaction time, then chasing."""
    controller, x = PDController(limit=level.cpu_speed_ms), x_cpu_m
    steps = int(t_s / dt + 1e-9)
    for k in range(steps):
        if k * dt >= level.tau_s - 1e-12:
            x += controller.step(x_land_m - x, dt) * dt
    leftover = t_s - steps * dt
    if leftover > 1e-9 and steps * dt >= level.tau_s - 1e-12:
        x += controller.step(x_land_m - x, leftover) * leftover
    return x


def paddle_error_m(level, x_land_m, x_cpu_m, flight_s):
    """Metres between the paddle and the landing point when the ball arrives: the reach deficit."""
    return abs(x_land_m - paddle_x(level, x_land_m, x_cpu_m, flight_s))
