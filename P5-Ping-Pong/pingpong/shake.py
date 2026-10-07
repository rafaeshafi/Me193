"""ShakeMonitor (judge gate J6): an FFT of the last second of gyro tells shaking from swinging.

Waving the hub produces swing-like peaks; the swing detector's oscillation guard already rejects
most of that, and this is the second line of defence: when the last second of rotation is a
sustained, narrow-band oscillation the paddle is locked for a second, so the last wave of a shake
can never score.

A window counts as shaking when ALL of these hold (each one rules out a real swing or a glitch):
  * the motion is big enough to matter (RMS above rms_min_dps),
  * the dominant FFT bin (not just a bin in the band) lies between 3 and 8 Hz,
  * that bin holds a large share of the non-DC power (a narrow peak, not the smooth spectrum of a
    single stroke),
  * the principal-axis signal has several sign reversals (full cycles).  A swing plus its
    backswing also looks periodic for a moment, and a false lock would cost the player their
    next hit, so a swing must never reach this count.

The hub's samples arrive at irregular times, so the last window_s seconds are resampled onto a
uniform grid before the FFT.  Samples inside a haptic blank window are ignored: the motors'
own vibration is not the player shaking.
"""

from collections import deque

import numpy as np


class ShakeMonitor:
    def __init__(self, *, gyro_per_dps, window_s=1.0, n=64, eval_period_s=0.1, band=(3, 8), rms_min_dps=60.0,
                 min_peakedness=0.3, min_crossings=4, lock_s=1.0, max_gap_s=0.25):
        self.gpd, self.window_s, self.n, self.eval_period_s = gyro_per_dps, window_s, n, eval_period_s
        self.band, self.rms_min, self.min_peakedness = band, rms_min_dps, min_peakedness
        self.min_crossings, self.lock_ns, self.max_gap_s = min_crossings, round(lock_s * 1e9), max_gap_s
        self._buf = deque()                   # (t_s, gx, gy, gz) in dps
        self._blanks = []
        self._last_eval = None
        self._hann = np.hanning(n)[:, None]

    # --- haptic blank windows ---------------------------------------------------------------------
    def blank(self, start_ns, end_ns):
        self._blanks.append((start_ns, end_ns))

    def _blanked(self, t_ns):
        self._blanks = [b for b in self._blanks if b[1] >= t_ns - 5_000_000_000]
        return any(a <= t_ns <= b for a, b in self._blanks)

    # --- main entry -----------------------------------------------------------------------------------
    def feed(self, sample):
        """-> the time (ns) the paddle should stay locked until, or None while nothing is shaking."""
        if self._blanked(sample.t_ns):
            return None
        t = sample.t_ns / 1e9
        self._buf.append((t, *(v / self.gpd for v in sample.g)))
        while self._buf and self._buf[0][0] < t - 1.3 * self.window_s:
            self._buf.popleft()
        if self._last_eval is not None and t - self._last_eval < self.eval_period_s:
            return None
        window = [row for row in self._buf if row[0] >= t - self.window_s]
        if not self._covers(window, t):
            return None
        self._last_eval = t
        return sample.t_ns + self.lock_ns if self._shaking(window, t) else None

    def _covers(self, window, t):
        if len(window) < 20 or t - window[0][0] < 0.9 * self.window_s:
            return False
        return all(b[0] - a[0] <= self.max_gap_s for a, b in zip(window, window[1:]))

    def _shaking(self, window, t):
        times = np.array([row[0] for row in window])
        grid = np.linspace(t - self.window_s + self.window_s / self.n, t, self.n)
        r = np.stack([np.interp(grid, times, [row[k] for row in window]) for k in (1, 2, 3)], axis=1)
        r -= r.mean(axis=0)
        if float(np.sqrt((r ** 2).sum(axis=1).mean())) < self.rms_min:
            return False
        power = (np.abs(np.fft.rfft(r * self._hann, axis=0)) ** 2).sum(axis=1)
        total = power[1:].sum()
        if total <= 0.0:
            return False
        k = 1 + int(np.argmax(power[1:]))                  # the dominant bin overall, DC excluded
        if not self.band[0] <= k <= self.band[1] or power[k] / total < self.min_peakedness:
            return False
        principal = r @ np.linalg.svd(r, full_matrices=False)[2][0]
        return _reversals(principal, 0.25 * float(np.max(np.abs(principal)))) >= self.min_crossings


def _reversals(x, hysteresis):
    """Sign changes of x that swing clear of +-hysteresis (so noise around zero does not count)."""
    count, state = 0, 0
    for v in x:
        if v > hysteresis:
            sign = 1
        elif v < -hysteresis:
            sign = -1
        else:
            continue
        if state and sign != state:
            count += 1
        state = sign
    return count
