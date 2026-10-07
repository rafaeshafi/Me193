"""replay: a recorded session fed back through the real pipeline on a simulated clock.

The recording holds every raw hub sample, every pose and the exact start times, so the same code
(swing detector, judge, game rules) can be run again offline:

  * to prove a recording reproduces the decisions made live, and
  * to ask counterfactuals on real data -- "what if the late window were 0.4 s?" -- without
    anybody swinging again.  `overrides` changes the swing detector, the judge or the level.

Nothing here can reach the network or the hub: the hub and camera are replay stand-ins and no
broker client is created.  Pump times of a real recording are unknown, so a 60 Hz grid anchored at
the session start is used (exact for fake-rig recordings, within one frame for live ones).
"""

import dataclasses
import json
from dataclasses import dataclass
from queue import SimpleQueue

from pingpong import levels, live
from pingpong import recorder as recorder_mod
from pingpong.clock import FakeClock
from pingpong.profile import Calibration
from pingpong.sources_fake import FakeDoubleMotor
from pingpong.swing import SwingParams

S = 1_000_000_000
STEP_NS = S // 240
PUMP_NS = S // 60
JUDGE_SETTINGS = ("t_pk", "d95_s", "min_dur_ms", "max_dur_ms", "max_reversals", "min_conf", "refractory_s",
                  "max_hits_per_s")
SECTIONS = {"swing": tuple(f.name for f in dataclasses.fields(SwingParams)), "judge": JUDGE_SETTINGS,
            "level": tuple(f.name for f in dataclasses.fields(levels.Level))}


def parse_overrides(items):
    """["level.late_s=0.4", ...] -> {"level": {"late_s": 0.4}}."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        section, dot, name = key.partition(".")
        if not (sep and dot and name):
            raise ValueError(f"expected section.name=value, got {item!r}")
        out.setdefault(section, {})[name] = float(value) if "." in value or "e" in value.lower() else int(value)
    return out


def check_overrides(overrides):
    for section, settings in overrides.items():
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}: choose from {', '.join(SECTIONS)}")
        for name in settings:
            if name not in SECTIONS[section]:
                raise ValueError(f"unknown {section} setting {name!r}: choose from {', '.join(SECTIONS[section])}")
    return overrides


class ReplayHub:
    """HubLink-shaped: samples appear when the replay clock reaches their recorded arrival time."""

    def __init__(self):
        self.imu, self.gestures = SimpleQueue(), SimpleQueue()
        self.dev = FakeDoubleMotor()
        self.dev.connected = True                  # haptic commands go to a recording stand-in, not a hub
        self.last_rx_ns = None
        self._clock = None

    def deliver(self, sample):
        self.last_rx_ns = sample.t_ns
        self.imu.put_nowait(sample)

    def is_stale(self, threshold_ms):
        return self.last_rx_ns is None or (self._clock.now_ns() - self.last_rx_ns) / 1e6 > threshold_ms

    def reconnect(self):
        pass                                       # the recording is what it is

    def close(self):
        pass


class ReplayVision:
    """VisionWorker-shaped: poses appear when the frame they came from would have been read."""

    def __init__(self, lag_ns):
        self.lag_ns, self._poses, self._last_read_ns = lag_ns, (), None

    def deliver(self, pose):
        self._poses = (self._poses + (pose,))[-64:]
        self._last_read_ns = pose.t_scene_ns + self.lag_ns

    def snapshot(self):
        return self._poses

    def pose_age_s(self, now_ns):
        return None if self._last_read_ns is None else (now_ns - self._last_read_ns) / 1e9

    def poll_tags(self):
        return []

    def latest_frame(self):
        return None

    def stats(self):
        return {}

    def stop(self):
        pass


class _Collector:
    """Recorder-shaped: keeps every event in memory and forwards to a real Recorder if there is one."""

    def __init__(self, inner=None):
        self.events, self.inner = [], inner

    def imu(self, sample):
        if self.inner:
            self.inner.imu(sample)

    def pose(self, pose):
        if self.inner:
            self.inner.pose(pose)

    def event(self, kind, t_ns, data=None):
        self.events.append((kind, t_ns, data or {}))
        if self.inner:
            self.inner.event(kind, t_ns, data)

    def game_events(self, events):
        for e in events:
            self.event(e.kind, e.t_ns, e.data)

    def tick(self, now_ns):
        if self.inner:
            self.inner.tick(now_ns)

    def close(self):
        if self.inner:
            self.inner.close()


@dataclass
class ReplayResult:
    rig: object
    events: list                     # [(kind, t_ns, data)] from the replayed game

    def count(self, kind):
        return sum(1 for k, _, _ in self.events if k == kind)

    @property
    def hits(self):
        return self.count("hit")

    @property
    def misses(self):
        return self.count("miss")

    @property
    def record(self):
        return self.rig.session.game.tracker.record


def _apply_settings(rig, overrides):
    game = rig.session.game
    if "swing" in overrides:
        rig.imu.set_params(dataclasses.replace(rig.imu.detector.p, **overrides["swing"]))
        if "t_pk" in overrides["swing"] and "t_pk" not in overrides.get("judge", {}):
            game.judge.t_pk = overrides["swing"]["t_pk"]               # the judge's J3 follows the detector
    for name, value in overrides.get("judge", {}).items():
        setattr(game.judge, name, value)


def _start(rig, record, overrides):
    game = rig.session.game
    level = levels.LEVELS[record["level"]]
    if "level" in overrides:
        level = dataclasses.replace(level, **overrides["level"])
    game.set_level(level)
    game.set_mode(record["mode"])
    game.start(record["started_at_ns"])


def replay(loaded, *, overrides=None, record_dir=None):
    """Run a recorded session again; -> ReplayResult (and a new recording if record_dir is given)."""
    overrides = check_overrides(overrides or {})
    meta, units = loaded.meta, loaded.meta["units"]
    clock = FakeClock(start_ns=meta["t0_ns"])
    hub, lag_ns = ReplayHub(), round(meta["camera_lag_s"] * S)
    hub._clock = clock
    vision = ReplayVision(lag_ns)
    inner = recorder_mod.Recorder(record_dir, {**meta, "source": "replay", "overrides": overrides}, clock=clock) \
        if record_dir else None
    collector = _Collector(inner)
    rig = live.assemble(
        hub=hub, capture=None, landmarker=None, vision=vision, clock=clock, recorder=collector,
        calibration=Calibration.from_json(json.dumps(meta["calibration"])), level=meta["level"], mode=meta["mode"],
        target=meta["target"], seed=meta["seed"], source="replay", scope=meta["scope"], no_motor=meta["no_motor"],
        gyro_per_dps=units["gyro_per_dps"], accel_per_g=units["accel_per_g"], fs_raw=units["fs_raw"],
        stale_ms=meta["stale_ms"], threaded=False, log=lambda *_: None)
    _apply_settings(rig, overrides)
    poses = sorted(loaded.poses, key=lambda p: p.t_scene_ns)
    starts, seen = [], set()
    for e in loaded.events:
        d = e["d"]
        if e["k"] == "phase" and d["phase"] == "COUNTDOWN" and d["started_at_ns"] not in seen:
            seen.add(d["started_at_ns"])
            starts.append(d)
    end_ns = max([meta["t0_ns"]] + [s.t_ns for s in loaded.imu[-1:]] + [p.t_scene_ns + lag_ns for p in poses[-1:]]
                 + [e["t"] for e in loaded.events])       # stop with the data: a silent tail would "pause" again
    i_imu = i_pose = i_start = 0
    next_pump = meta["t0_ns"]
    while clock.now_ns() < end_ns:
        clock.advance_s(STEP_NS / S)
        now = clock.now_ns()
        while i_imu < len(loaded.imu) and loaded.imu[i_imu].t_ns <= now:
            hub.deliver(loaded.imu[i_imu])
            i_imu += 1
        while i_pose < len(poses) and poses[i_pose].t_scene_ns + lag_ns <= now:
            vision.deliver(poses[i_pose])
            i_pose += 1
        while i_start < len(starts) and starts[i_start]["started_at_ns"] <= now:
            _start(rig, starts[i_start], overrides)
            i_start += 1
        while next_pump <= now:
            rig.pump()
            next_pump += PUMP_NS
    rig.close()
    return ReplayResult(rig=rig, events=collector.events)
