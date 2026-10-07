"""Injectable clocks: every time-dependent module takes one so tests are exact."""

import time


class Clock:
    """Real monotonic clock in integer nanoseconds."""

    def now_ns(self):
        return time.monotonic_ns()


class FakeClock:
    """Deterministic clock for tests and --fake runs."""

    def __init__(self, start_ns=0):
        self._t = int(start_ns)

    def now_ns(self):
        return self._t

    def advance_s(self, dt):
        if dt < 0:
            raise ValueError("a clock cannot go backwards")
        self._t += round(dt * 1e9)
