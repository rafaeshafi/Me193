"""The lock policy: how the safe decides, from one probability vector per camera frame.

    IDLE ──button──▶ LISTENING ──3 digits──▶ UNLOCKED ──unlock_s──▶ IDLE
                        │   │                  (code matches)
                        │   └──3 digits──────▶ DENIED ──deny_s──▶ IDLE  (attempts + 1)
                        └──entry_timeout_s──▶ IDLE
    max_attempts failures in a row ──▶ LOCKOUT ──lockout_s──▶ IDLE (button ignored meanwhile)

A digit is accepted when the same class 1..5 is the arg-max with confidence >= min_confidence for hold_frames frames
in a row. After an acceptance the hand must release: gap_frames frames of class 0 (no hand) or of another class before
the next digit can start counting, so one held hand is one digit. The policy never sees the picture, only the
probabilities, so the Mac, the tests and the board all decide the same way.
"""
import time

from lock.config import CODE_LEN, DIGITS

IDLE, LISTENING, UNLOCKED, DENIED, LOCKOUT = "IDLE", "LISTENING", "UNLOCKED", "DENIED", "LOCKOUT"


def argmax(probs):
    best, best_p = 0, -1.0
    for i, p in enumerate(probs):
        if p > best_p:
            best, best_p = i, p
    return best, best_p


class DigitReader:
    """Turns a stream of per-frame class guesses into accepted digits (hold to accept, release between digits)."""

    def __init__(self, hold_frames, gap_frames, min_confidence):
        self.hold, self.gap, self.min_conf = int(hold_frames), int(gap_frames), float(min_confidence)
        self.reset()

    def reset(self):
        self.candidate, self.run, self.released, self.gap_run = 0, 0, True, 0

    @property
    def progress(self):
        """0..1 of the way to accepting the current candidate."""
        return min(self.run / self.hold, 1.0) if self.candidate in DIGITS and self.released else 0.0

    def observe(self, probs):
        """One frame; returns the accepted digit (1..5) or None."""
        cls, p = argmax(probs)
        confident = p >= self.min_conf and cls in DIGITS
        if not self.released:  # waiting for the hand to go away (or change) after an acceptance
            if not confident or cls != self.candidate:
                self.gap_run += 1
                if self.gap_run >= self.gap:
                    self.released, self.candidate, self.run = True, 0, 0
            else:
                self.gap_run = 0
            return None
        if confident and cls == self.candidate:
            self.run += 1
        elif confident:
            self.candidate, self.run = cls, 1
        else:
            self.candidate, self.run = 0, 0
        if self.run >= self.hold:
            digit = self.candidate
            self.released, self.gap_run, self.run = False, 0, 0
            return digit
        return None


class Lock:
    def __init__(self, cfg, now=time.monotonic, on_event=None):
        self.cfg, self._now, self.on_event = cfg, now, on_event or (lambda kind, info: None)
        self.code = list(cfg["code"])
        self.reader = DigitReader(cfg["hold_frames"], cfg["gap_frames"], cfg["min_confidence"])
        self.state, self.since, self.entered, self.attempts = IDLE, now(), [], 0
        self.last_probs, self.last_digit, self.unlocks, self.denials = None, None, 0, 0

    # ----- inputs -----
    def press_button(self):
        """The physical button (or 'b' on the Mac). Only matters when idle; a press while listening restarts the entry."""
        if self.state == LOCKOUT:
            self.on_event("button_ignored", {"state": self.state})
            return False
        if self.state in (IDLE, LISTENING):
            self._go(LISTENING)
            self.entered, self.last_digit = [], None
            self.reader.reset()
            self.on_event("listening", {})
            return True
        return False

    def observe(self, probs, image=None):
        """One camera frame's class probabilities (index = finger count). image is only kept for the intruder log."""
        self.last_probs = list(probs)
        if self.state != LISTENING:
            return None
        digit = self.reader.observe(probs)
        if digit is not None:
            self.entered.append(digit)
            self.last_digit = digit
            self.on_event("digit", {"digit": digit, "n": len(self.entered)})
            if len(self.entered) == CODE_LEN:
                self._decide(image)
        return digit

    def tick(self):
        """Time-outs; call it every loop."""
        elapsed = self._now() - self.since
        if self.state == LISTENING and elapsed >= self.cfg["entry_timeout_s"]:
            self.on_event("timeout", {"entered": list(self.entered)})
            self._go(IDLE)
        elif self.state == UNLOCKED and elapsed >= self.cfg["unlock_s"]:
            self._go(IDLE)
        elif self.state == DENIED and elapsed >= self.cfg["deny_s"]:
            self._go(LOCKOUT if self.attempts >= self.cfg["max_attempts"] else IDLE)
        elif self.state == LOCKOUT and elapsed >= self.cfg["lockout_s"]:
            self.attempts = 0
            self._go(IDLE)

    # ----- outputs -----
    def snapshot(self):
        elapsed = self._now() - self.since
        remaining = {LISTENING: self.cfg["entry_timeout_s"], UNLOCKED: self.cfg["unlock_s"], DENIED: self.cfg["deny_s"],
                     LOCKOUT: self.cfg["lockout_s"]}.get(self.state)
        return {
            "state": self.state, "entered": list(self.entered), "progress": self.reader.progress,
            "candidate": self.reader.candidate if self.reader.released else 0, "released": self.reader.released,
            "attempts": self.attempts, "elapsed": round(elapsed, 2),
            "remaining": None if remaining is None else round(max(remaining - elapsed, 0.0), 2),
            "leds": self.leds(), "unlocks": self.unlocks, "denials": self.denials,
            "probs": None if self.last_probs is None else [round(float(p), 3) for p in self.last_probs],
        }

    def leds(self):
        """(red, green) as the two LEDs should be right now; LOCKOUT blinks red at 2 Hz."""
        if self.state == LISTENING:
            return (True, True)
        if self.state == UNLOCKED:
            return (False, True)
        if self.state == DENIED:
            return (True, False)
        if self.state == LOCKOUT:
            return (int((self._now() - self.since) * 4) % 2 == 0, False)
        return (False, False)

    # ----- internals -----
    def _go(self, state):
        self.state, self.since = state, self._now()

    def _decide(self, image):
        if self.entered == self.code:
            self.attempts, self.unlocks = 0, self.unlocks + 1
            self._go(UNLOCKED)
            self.on_event("unlock", {"entered": list(self.entered)})
        else:
            self.attempts, self.denials = self.attempts + 1, self.denials + 1
            self._go(DENIED)
            info = {"entered": list(self.entered), "attempts": self.attempts}
            if self.attempts >= self.cfg["max_attempts"]:
                info["image"] = image  # the intruder log keeps this picture
                self.on_event("intruder", info)
            self.on_event("deny", info)
