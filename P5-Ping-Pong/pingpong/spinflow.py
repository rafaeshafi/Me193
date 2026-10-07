"""SpinCollector: guided collection of labelled swings for the spin classifier (pure; the window is in tools/).

The player is asked for flat, top and back swings in turn (interleaved, so arm fatigue and drift are spread
over all three classes instead of piling onto the last one).  Every swing the SwingDetector recognises,
with the player's own calibration, gives one 12-feature sample labelled with the class that was asked for:
exactly the features the game computes at every IMPACT.
"""

from pingpong import spin
from pingpong.swing import SwingDetector

HINTS = {"flat": "FLAT: push the paddle straight through the ball",
         "top": "TOPSPIN: brush UP over the ball, wrist rolling forward",
         "back": "BACKSPIN: chop DOWN under the ball"}


class SpinCollector:
    def __init__(self, params, n_per_class=12):
        self.detector = SwingDetector(params)
        self.order = [label for _ in range(n_per_class) for label in spin.CLASSES]
        self.samples, self._notes = [], []

    def finished(self):
        return len(self.samples) >= len(self.order)

    def progress(self):
        return len(self.samples), len(self.order)

    def prompt(self):
        if self.finished():
            return "Done: every swing is recorded"
        return f"Swing {len(self.samples) + 1} of {len(self.order)}: {HINTS[self.order[len(self.samples)]]}"

    def take_notes(self):
        notes, self._notes = self._notes, []
        return notes

    def feed_imu(self, sample):
        for event in self.detector.feed(sample):
            if event.kind == "IMPACT" and not self.finished():
                label = self.order[len(self.samples)]
                self.samples.append(event.feat)
                self._notes.append(f"{label} swing {len(self.samples)}/{len(self.order)} captured "
                                   f"(peak {event.w_pk:.0f} dps)")

    def result(self):
        """-> (features, labels) once every swing is in."""
        if not self.finished():
            raise ValueError(f"only {len(self.samples)} of {len(self.order)} swings are recorded")
        return list(self.samples), list(self.order)
