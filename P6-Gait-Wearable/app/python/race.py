"""Racing on the spot: a speed from the gait, and a race against ghost runners.

speed = cadence/60 x stride, stride_m = clamp(0.6 + 0.5 x (peak_g - 1), 0.5, 2.2). The classifier gates it: STAND, BIKE
and no data give 0, WALK is capped at 7 km/h, RUN is uncapped. SpeedModel smooths it with a 1 s exponential average.
Race: start -> 3 s countdown -> running (distance integrates the smoothed speed every tick) -> finished when the line is
crossed (finish time interpolated inside the last step). Ghosts run at fixed paces, plus a replay of the runner's best
profile for that distance. ResultStore keeps every finish in a JSON file and serves leaderboards.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

STRIDE_AT_1G, STRIDE_PER_G, STRIDE_MIN, STRIDE_MAX = 0.6, 0.5, 0.5, 2.2
WALK_CAP = 7.0 / 3.6          # m/s
SMOOTH_TAU = 1.0              # s
COUNTDOWN_S = 3.0
DISTANCES = (100, 400, 1000)
GHOSTS = (("Jogger", 9.0), ("Runner", 13.0), ("Sprinter", 22.0))   # name, km/h
LEADERBOARD_N = 10
PROFILE_STEP_S = 0.5


def stride_m(peak_g):
    return max(STRIDE_MIN, min(STRIDE_MAX, STRIDE_AT_1G + STRIDE_PER_G * (peak_g - 1.0)))


def raw_speed(act, cad, pk):
    """m/s from one classified window, before smoothing."""
    if act not in ("WALK", "RUN") or not cad or pk is None:
        return 0.0
    v = cad / 60.0 * stride_m(pk)
    return min(v, WALK_CAP) if act == "WALK" else v


def replay_distance(profile, t):
    """Distance (m) a recorded (elapsed_s, speed_m_s) profile has covered after t seconds."""
    dist, last_t = 0.0, 0.0
    for elapsed, speed in profile:
        if elapsed >= t:
            dist += speed * (t - last_t)
            return dist
        dist += speed * (elapsed - last_t)
        last_t = elapsed
    return dist


class SpeedModel:
    def __init__(self):
        self.speed = 0.0
        self._t = None

    def update(self, act, cad, pk, now):
        target = raw_speed(act, cad, pk)
        if self._t is None:
            self.speed = target
        else:
            a = 1.0 - math.exp(-max(0.0, now - self._t) / SMOOTH_TAU)
            self.speed += a * (target - self.speed)
        self._t = now
        return self.speed


class ResultStore:
    """Every finished race, in a JSON list on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._boards = {}
        self._entries = []
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                ok = isinstance(data, list) and all(
                    isinstance(e, dict) and "distance" in e and "time" in e and "who" in e for e in data)
            except (ValueError, OSError):
                ok = False
            if ok:
                self._entries = data
            else:
                try:
                    self.path.rename(self.path.with_name(self.path.name + ".bad-" + time.strftime("%Y%m%d-%H%M%S")))
                except OSError:
                    pass

    def save(self, distance, who, seconds, profile):
        entry = {"who": who or "anon", "distance": int(distance), "time": round(float(seconds), 2),
                 "avg_kmh": round(distance / seconds * 3.6, 2) if seconds > 0 else 0.0,
                 "when": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": [list(p) for p in profile]}
        with self._lock:
            self._entries.append(entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries))
            os.replace(str(tmp), str(self.path))
            self._boards = {}
            rank = 1 + sum(1 for e in self._entries if e["distance"] == entry["distance"] and e["time"] < entry["time"])
        return dict(entry, rank=rank)

    def leaderboard(self, distance, n=LEADERBOARD_N):
        with self._lock:
            key = (distance, n)
            if key not in self._boards:
                rows = sorted((e for e in self._entries if e["distance"] == distance), key=lambda e: e["time"])[:n]
                self._boards[key] = [{k: e[k] for k in ("who", "time", "avg_kmh", "when")} for e in rows]
            return [dict(r) for r in self._boards[key]]

    def best(self, distance, who):
        if not who or who == "anon":
            return None
        with self._lock:
            mine = [e for e in self._entries if e["distance"] == distance and e["who"] == who]
        return min(mine, key=lambda e: e["time"]) if mine else None


class Race:
    """idle -> countdown -> running -> finished; abort() returns to idle without saving."""

    def __init__(self, store, clock=time.monotonic):
        self.store = store
        self._clock = clock
        self.distance = DISTANCES[1]
        self._lock = threading.RLock()
        self._gen = 0
        self._reset()

    def _reset(self):
        self._gen += 1
        self.phase = "idle"
        self.who = ""
        self.t0 = None
        self.dist = 0.0
        self.elapsed = 0.0
        self.speed = 0.0
        self.profile = []
        self.result = None
        self.best = None
        self._last = None

    def start(self, distance, who=""):
        with self._lock:
            try:
                distance = int(distance)
            except (TypeError, ValueError):
                raise ValueError("distance must be one of %s" % (DISTANCES,))
            if distance not in DISTANCES:
                raise ValueError("distance must be one of %s" % (DISTANCES,))
            self._reset()
            self.distance = distance
            self.who = who or "anon"
            self.best = self.store.best(self.distance, self.who)
            self.t0 = self._clock() + COUNTDOWN_S
            self.phase = "countdown"

    def abort(self):
        with self._lock:
            self._reset()

    def tick(self, speed, now):
        """Advance with the current smoothed speed (m/s); call every loop pass."""
        pending = None
        with self._lock:
            self.speed = speed
            if self.phase == "countdown":
                if now >= self.t0:
                    self.phase = "running"
                    self._last = self.t0
                    self.tick(speed, now)
            elif self.phase == "running":
                dt = max(0.0, now - self._last)
                self._last = now
                self.dist += speed * dt
                self.elapsed = now - self.t0
                if self.dist >= self.distance:
                    if speed > 0:
                        self.elapsed -= (self.dist - self.distance) / speed
                    self.dist = float(self.distance)
                    self.phase = "finished"
                    self.profile.append((round(self.elapsed, 2), round(speed, 3)))
                    pending = (self._gen, self.distance, self.who, self.elapsed, list(self.profile))
                elif not self.profile or self.elapsed - self.profile[-1][0] >= PROFILE_STEP_S:
                    self.profile.append((round(self.elapsed, 2), round(speed, 3)))
        if pending:
            gen, distance, who, elapsed, profile = pending
            saved = self.store.save(distance, who, elapsed, profile)
            with self._lock:
                if self._gen == gen:
                    beat = [n for n, kmh in GHOSTS if distance / (kmh / 3.6) > elapsed]
                    self.result = {"time": saved["time"], "avg_kmh": saved["avg_kmh"], "rank": saved["rank"], "beat": beat}

    def ghosts(self):
        with self._lock:
            return self._ghosts()

    def _ghosts(self):
        t = max(0.0, self.elapsed) if self.phase in ("running", "finished") else 0.0
        out = [{"name": n, "kmh": k, "dist": min(float(self.distance), k / 3.6 * t)} for n, k in GHOSTS]
        if self.best:
            out.append({"name": "Your best", "kmh": self.best["avg_kmh"],
                        "dist": min(float(self.distance), replay_distance(self.best["profile"], t))})
        return out

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        now = self._clock()
        return {"phase": self.phase, "distance": self.distance, "who": self.who,
                "countdown": round(max(0.0, self.t0 - now), 1) if self.phase == "countdown" else None,
                "elapsed": round(self.elapsed, 2), "dist": round(self.dist, 1), "speed_kmh": round(self.speed * 3.6, 1),
                "ghosts": self._ghosts(), "result": self.result, "leaderboard": self.store.leaderboard(self.distance)}
