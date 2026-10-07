"""Recorder: a live session written as JSON lines, so a bad run becomes an offline fixture.

    recordings/<session>/session.json   seed, level, mode, calibration, units, start time
                         imu.jsonl      every hub sample, raw counts, arrival-stamped
                         pose.jsonl     every accepted paddle pose (already lag-stamped)
                         events.jsonl   game events, tags, pauses -- verdicts with all six gates

That is enough to replay the session through the real code (pingpong/replay.py) and to write a
failing test for whatever went wrong, instead of relying on a verbal description.

Producers (the IMU thread, the game loop) only append to deques; the game loop calls tick()
which writes at most once per flush interval, so a crash loses at most a second.  Recording must
never break the game: the first write error is reported once and recording switches itself off.
"""

import dataclasses
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import config
from pingpong import profile
from pingpong.events import ImuSample, PaddlePose

VERSION = 1
STREAMS = ("imu", "pose", "events")


def default_root():
    return config.HERE / "recordings"


def session_name(player, wall=None):
    """e.g. 20261007-090503-rafae  (wall = (Y, M, D, h, m, s), default: now)."""
    y, mo, d, h, mi, s = wall if wall is not None else time.localtime()[:6]
    return f"{y:04d}{mo:02d}{d:02d}-{h:02d}{mi:02d}{s:02d}-{profile.slug(player)}"


def session_meta(*, source, player, seed, level, mode, target, scope, t0_ns, calibration, gyro_per_dps,
                 accel_per_g, fs_raw, lag_s, stale_ms, no_motor, learn=False):
    """Everything a replay needs to rebuild the same game."""
    return {"source": source, "player": player, "seed": seed, "level": level, "mode": mode, "target": target,
            "scope": scope, "t0_ns": t0_ns, "calibration": json.loads(calibration.to_json()),
            "units": {"gyro_per_dps": gyro_per_dps, "accel_per_g": accel_per_g, "fs_raw": fs_raw},
            "camera_lag_s": lag_s, "stale_ms": stale_ms, "no_motor": no_motor, "learn": learn}


def _plain(obj):
    """Anything the game puts in an event -> something json.dumps accepts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):       # numpy scalars
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)


def _dump(obj):
    return json.dumps(_plain(obj), separators=(",", ":"), allow_nan=False)


class Recorder:
    def __init__(self, directory, meta, *, flush_interval_s=1.0, clock=None, log=print):
        self.dir = Path(directory)
        if (self.dir / "session.json").exists():
            raise FileExistsError(f"{self.dir} already holds a recorded session; never overwrite one")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "session.json").write_text(json.dumps(_plain({"version": VERSION, **meta}), indent=2) + "\n")
        self.flush_interval_ns = round(flush_interval_s * 1e9)
        self.clock, self.log = clock, log
        self._queues = {name: deque() for name in STREAMS}
        self._files = {name: open(self.dir / f"{name}.jsonl", "a", buffering=1 << 16) for name in STREAMS}
        self._lock = threading.Lock()
        self._last_flush_ns = clock.now_ns() if clock is not None else 0
        self._dead = self._closed = False

    # --- producers (any thread) --------------------------------------------------------------------
    def imu(self, sample):
        if not self._dead:
            self._queues["imu"].append(sample)

    def pose(self, pose):
        if not self._dead:
            self._queues["pose"].append(pose)

    def event(self, kind, t_ns, data=None):
        if not self._dead:
            self._queues["events"].append({"t": t_ns, "k": kind, "d": data or {}})

    def game_events(self, events):
        for e in events:
            self.event(e.kind, e.t_ns, e.data)

    # --- writing -----------------------------------------------------------------------------------------
    def tick(self, now_ns):
        if now_ns - self._last_flush_ns >= self.flush_interval_ns:
            self.flush()

    def flush(self):
        with self._lock:
            self._last_flush_ns = self.clock.now_ns() if self.clock is not None else self._last_flush_ns
            if self._dead or self._closed:
                return
            try:
                for name in STREAMS:
                    out, queue_ = self._files[name], self._queues[name]
                    while queue_:
                        out.write(_dump(self._row(name, queue_.popleft())) + "\n")
                    out.flush()
            except Exception as exc:
                self._dead = True
                self.log(f"recording stopped: {exc} (the game carries on; nothing more is recorded)")

    @staticmethod
    def _row(name, item):
        if name == "imu":
            return {"t": item.t_ns, "g": item.g, "a": item.a}
        if name == "pose":
            return {"t": item.t_scene_ns, "u": item.u, "v": item.v, "c": item.conf, "h": item.hand}
        return item

    def close(self):
        if self._closed:
            return
        self.flush()
        with self._lock:
            self._closed = True
            for out in self._files.values():
                try:
                    out.close()
                except Exception:
                    pass


@dataclass
class Loaded:
    meta: dict
    imu: list
    poses: list
    events: list


def load(directory):
    """Read a recorded session back: the description, IMU samples, poses and event dicts."""
    directory = Path(directory)
    meta = json.loads((directory / "session.json").read_text())

    def rows(name):
        path = directory / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []

    imu = [ImuSample(t_ns=r["t"], g=tuple(r["g"]), a=tuple(r["a"])) for r in rows("imu")]
    poses = [PaddlePose(t_scene_ns=r["t"], u=r["u"], v=r["v"], conf=r["c"], hand=r["h"]) for r in rows("pose")]
    return Loaded(meta=meta, imu=imu, poses=poses, events=rows("events"))
