"""The moment of the game at which the picture shows the ball.

The picture is drawn ahead of the game by the display's delay (view = now + lead), so that when the light reaches the eye it shows the
world as it is then.  That is exact for a ball whose flight is known.  It is not at a junction, where the game only learns what the
ball does next some time after the picture needs to know: the hand meeting the ball (a camera frame and a pose reading later), the
ball being let go after the hold (the wrist's flick has to be read first), the computer meeting your return (its answer comes at that
moment).  By then the ball had been drawn flying on past the paddle and it jumped back onto it: 100-340 px in one picture.

So at a junction the ball waits (is `pinned`) where the paddle is, and when the game has spoken it runs a little quicker than the clock
until the picture has caught up.  Pure arithmetic on nanoseconds; View says what to pin.
"""

S = 1_000_000_000
STALL_NS = S // 4                # a gap in the pictures longer than this (or time running backwards) forgets the wait
CATCH_UP = 0.5                   # after a wait the ball runs this much faster than the clock (1.5 x) until it is back with the picture


class BallClock:
    def __init__(self, catch_up=CATCH_UP):
        self.catch_up = catch_up
        self.reset()

    def reset(self):
        self._shown = self._last = None

    def tick(self, view_ns, pin_ns=None, floor_ns=None):
        """-> the moment of the game to draw the ball at.  view_ns: the picture's own time; pin_ns: the ball may not be shown beyond
        this moment (it waits to hear what happens there); floor_ns: the ball is shown from at least this moment (a new flight
        starts there)."""
        shown, last, self._last = self._shown, self._last, view_ns
        if shown is None or view_ns < last or view_ns - last > STALL_NS:
            shown = view_ns if pin_ns is None else min(view_ns, pin_ns)
        else:
            ahead = min(shown + round((1.0 + self.catch_up) * (view_ns - last)), view_ns)
            shown = ahead if pin_ns is None else max(shown, min(ahead, pin_ns))        # it can wait, but not go back
        if floor_ns is not None:
            shown = min(view_ns, max(shown, floor_ns))
        self._shown = shown
        return shown
