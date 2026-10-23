"""Labelled CSV recording on the board: the guided script, single-activity takes, and an unlabelled rolling log.

Rows are raw 100 Hz samples, not features, so features can change later without re-recording.
CSV columns: t (seconds since the session started), label ("" during a get-ready gap, "-" in the rolling log),
ax ay az (g), gx gy gz (deg/s). Files: <YYYYmmdd-HHMMSS>-guided[-who].csv, <stamp>-walk[-who].csv, rolling-<stamp>.csv.

Threads: on_samples() runs on the Bridge thread at 10 Hz; start/stop/status come from the web and main threads, so
every public method takes the lock. The script's position is a pure function of time (bisect over the step ends):
asking for the prompt never moves it, and only on_samples() closes a finished session, after writing its last rows.
`t` values are seconds on the same monotonic clock the samples are stamped with. The rolling log runs only while no
session is active, starts a new file every hour and keeps the last two.
"""
import bisect
import csv
import math
import re
import threading
import time
from pathlib import Path

LABELS = ("STAND", "WALK", "RUN", "BIKE")
SCRIPT = [("STAND", 20), ("WALK", 30), ("RUN", 30), ("BIKE", 30)]
GAP_S = 5
HEADER = ["t", "label", "ax", "ay", "az", "gx", "gy", "gz"]
ROLLING_ROTATE_S = 3600
ROLLING_KEEP = 2


def clean(who):
    return re.sub(r"[^A-Za-z0-9_-]", "", who or "")[:16]


def _row(t, label, sample):
    return ["%.3f" % t, label] + ["%.4f" % v for v in sample]


class Recorder:
    def __init__(self, data_dir, clock=time.monotonic, rolling=True):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions = None  # cached listing, cleared when a session starts or ends
        self._labels = []     # one per step, "" = unlabelled gap
        self._ends = []       # cumulative end time of each step (clock seconds)
        self._t0 = 0.0
        self._file = None
        self._writer = None
        self.path = None
        self.rows = 0
        self.who = ""
        self._rolling_on = rolling
        self._rolling = None
        self._rolling_writer = None
        self._rolling_t0 = 0.0

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
        with self._lock:
            self._close_session()
            self.who = clean(who)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = "%s-%s%s" % (stamp, kind, "-" + self.who if self.who else "")
            path = self.dir / (base + ".csv")
            try:
                f = open(path, "x", newline="")
            except FileExistsError:  # two takes within one second
                path = self.dir / (base + "-2.csv")
                f = open(path, "x", newline="")
            self._file, self.path = f, path
            self._writer = csv.writer(f)
            self._writer.writerow(HEADER)
            self._t0 = self._clock()
            self._labels = [label for label, _ in steps]
            self._ends = []
            end = self._t0
            for _, seconds in steps:
                end += seconds
                self._ends.append(end)
            self.rows = 0
            self._sessions = None

    def _close_session(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
        self._labels, self._ends = [], []
        self._sessions = None

    def stop(self):
        with self._lock:
            self._close_session()

    def close(self):
        """Stop everything, including the rolling log (shutdown)."""
        with self._lock:
            self._close_session()
            if self._rolling:
                self._rolling.close()
                self._rolling = None

    @property
    def active(self):
        return self._file is not None

    # --- the script, read-only -----------------------------------------------------------------------------
    def _step(self, now):
        """-> (label, remaining_s) of the step active at `now`, or None when idle or past the end. Never mutates."""
        if not self._ends:
            return None
        i = bisect.bisect_right(self._ends, now)
        if i >= len(self._ends):
            return None
        return self._labels[i], self._ends[i] - now

    def current(self, now=None):
        with self._lock:
            return self._step(self._clock() if now is None else now)

    def prompt(self, now=None):
        return self._prompt(self.current(now))

    @staticmethod
    def _prompt(cur):
        if cur is None:
            return ""
        label, remaining = cur
        return "%s %ds" % (label or "GET READY", int(math.ceil(remaining)))

    # --- data ------------------------------------------------------------------------------------------------
    def on_samples(self, t, samples, rate=100):
        """Write a batch; `t` is the clock time of the LAST sample, earlier ones are spaced 1/rate before it.
        Closes the session once the batch reaches the end of the script; feeds the rolling log while idle."""
        n = len(samples)
        with self._lock:
            if self._file is None:
                self._rolling_write(t, samples, rate)
                return
            rows = []
            for i, s in enumerate(samples):
                ts = t - (n - 1 - i) / rate
                if ts < self._t0:
                    continue  # stamped before the session started
                cur = self._step(ts)
                if cur is None:
                    break  # past the end of the script
                rows.append(_row(ts - self._t0, cur[0], s))
            self._writer.writerows(rows)
            self.rows += len(rows)
            self._file.flush()
            if t >= self._ends[-1]:
                self._close_session()

    def _rolling_write(self, t, samples, rate):
        if not self._rolling_on:
            return
        if self._rolling is None or t - self._rolling_t0 > ROLLING_ROTATE_S:
            self._rotate_rolling(t)
        n = len(samples)
        self._rolling_writer.writerows(_row(t - (n - 1 - i) / rate - self._rolling_t0, "-", s)
                                       for i, s in enumerate(samples))
        self._rolling.flush()

    def _rotate_rolling(self, t):
        if self._rolling:
            self._rolling.close()
        current = self.dir / ("rolling-%s.csv" % time.strftime("%Y%m%d-%H%M%S"))
        self._rolling = open(current, "w", newline="")
        self._rolling_writer = csv.writer(self._rolling)
        self._rolling_writer.writerow(HEADER)
        self._rolling_t0 = t
        # prune by modification time, never the file just opened: the board's clock can come up earlier after a reboot
        others = sorted((p for p in self.dir.glob("rolling-*.csv") if p != current), key=lambda p: p.stat().st_mtime)
        for old in others[:-(ROLLING_KEEP - 1)] if ROLLING_KEEP > 1 else others:
            old.unlink()

    def sessions(self):
        """Finished session files (not the rolling logs, not the one being written), oldest first."""
        with self._lock:
            if self._sessions is None:
                busy = self.path.name if self._file else None
                self._sessions = sorted(p.name for p in self.dir.glob("*.csv")
                                        if not p.name.startswith("rolling-") and p.name != busy)
            return list(self._sessions)

    def status(self):
        with self._lock:
            cur = self._step(self._clock())
            active = self._file is not None
            name = self.path.name if self.path else None
            rows, who = self.rows, self.who
        return {"active": active, "file": name, "rows": rows,
                "label": cur[0] if cur else None, "remaining_s": round(cur[1], 1) if cur else None,
                "prompt": self._prompt(cur), "who": who}
