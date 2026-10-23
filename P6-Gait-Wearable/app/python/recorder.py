"""Labelled CSV recording on the board: the guided script, single-activity takes, and an always-on unlabelled rolling log.

Rows are raw 100 Hz samples, not features, so features can change later without re-recording.
CSV columns: t (seconds since the session started), label ("" during a get-ready gap, "-" in the rolling log),
ax ay az (g), gx gy gz (deg/s). Files: <YYYYmmdd-HHMMSS>-guided[-who].csv, <stamp>-walk[-who].csv, rolling-<stamp>.csv.
"""
import csv
import math
import re
import time
from pathlib import Path

LABELS = ("STAND", "WALK", "RUN", "BIKE")
SCRIPT = [("STAND", 20), ("WALK", 30), ("RUN", 30), ("BIKE", 30)]
GAP_S = 5
HEADER = ["t", "label", "ax", "ay", "az", "gx", "gy", "gz"]
ROLLING_MAX_S = 3600  # the rolling log stops after an hour so a forgotten board does not fill the disk


def clean(who):
    return re.sub(r"[^A-Za-z0-9_-]", "", who or "")[:16]


class Recorder:
    def __init__(self, data_dir, clock=time.monotonic, rolling=True):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._steps = []          # [(label, seconds)], label "" = unlabelled gap
        self._t0 = 0.0
        self._step_end = 0.0
        self._file = None
        self._writer = None
        self.path = None
        self.rows = 0
        self.who = ""
        self._rolling = None
        self._rolling_t0 = self._clock()
        if rolling:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._rolling_path = self.dir / ("rolling-%s.csv" % stamp)
            self._rolling = open(self._rolling_path, "w", newline="")
            self._rolling_writer = csv.writer(self._rolling)
            self._rolling_writer.writerow(HEADER)

    # --- starting and stopping ---------------------------------------------------------------------------
    def start_guided(self, who=""):
        steps = []
        for label, seconds in SCRIPT:
            steps += [("", GAP_S), (label, seconds)]
        self._start(steps, who, "guided")

    def start_single(self, label, seconds=30, who=""):
        label = label.upper()
        if label not in LABELS:
            raise ValueError("unknown label %r" % label)
        self._start([("", GAP_S), (label, seconds)], who, label.lower())

    def _start(self, steps, who, kind):
        self.stop()
        self.who = clean(who)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s-%s%s.csv" % (stamp, kind, "-" + self.who if self.who else "")
        self.path = self.dir / name
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)
        self._steps = list(steps)
        self._t0 = self._clock()
        self._step_end = self._t0 + self._steps[0][1]
        self.rows = 0

    def stop(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
        self._steps = []

    def close(self):
        """Stop everything, including the rolling log (shutdown)."""
        self.stop()
        if self._rolling:
            self._rolling.close()
            self._rolling = None

    @property
    def active(self):
        return self._file is not None

    # --- the script ----------------------------------------------------------------------------------------
    def current(self, now=None):
        """-> (label, remaining_s) of the step active at `now`, or None when idle. Advances the script."""
        if not self.active:
            return None
        now = self._clock() if now is None else now
        while self._steps and now >= self._step_end:
            self._steps.pop(0)
            if self._steps:
                self._step_end += self._steps[0][1]
        if not self._steps:
            self.stop()
            return None
        return self._steps[0][0], self._step_end - now

    def prompt(self, now=None):
        cur = self.current(now)
        if cur is None:
            return ""
        label, remaining = cur
        return "%s %ds" % (label or "GET READY", int(math.ceil(remaining)))

    # --- data ------------------------------------------------------------------------------------------------
    def on_samples(self, t, samples, rate=100):
        """Write a batch; `t` is the clock time of the LAST sample, earlier ones are spaced 1/rate before it."""
        n = len(samples)
        if self._rolling:
            if t - self._rolling_t0 > ROLLING_MAX_S:
                self._rolling.close()
                self._rolling = None
            else:
                for i, s in enumerate(samples):
                    ts = t - (n - 1 - i) / rate
                    self._rolling_writer.writerow(["%.3f" % (ts - self._rolling_t0), "-"] + ["%.4f" % v for v in s])
        if not self.active:
            return
        for i, s in enumerate(samples):
            ts = t - (n - 1 - i) / rate
            cur = self.current(ts)
            if cur is None:
                return
            self._writer.writerow(["%.3f" % (ts - self._t0), cur[0]] + ["%.4f" % v for v in s])
            self.rows += 1
        self._file.flush()

    def sessions(self):
        """Recorded session files (not the rolling logs), oldest first."""
        return sorted(p.name for p in self.dir.glob("*.csv") if not p.name.startswith("rolling-"))

    def status(self):
        cur = self.current()
        return {"active": self.active, "file": self.path.name if self.path else None, "rows": self.rows,
                "label": cur[0] if cur else None, "remaining_s": round(cur[1], 1) if cur else None,
                "prompt": self.prompt(), "who": self.who}
