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

from pingpong import levels, live, posegyro
from pingpong import overrides as overrides_mod
from pingpong import recorder as recorder_mod
from pingpong.clock import FakeClock
from pingpong.profile import Calibration
from pingpong.sources_fake import FakeDoubleMotor

S = 1_000_000_000
STEP_NS = S // 240
PUMP_NS = S // 60
parse_overrides, check_overrides = overrides_mod.parse, overrides_mod.check      # the names the tools and tests use


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

    def battery_pct(self):
        return None

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
    warnings: list = dataclasses.field(default_factory=list)

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


def _start(rig, record):
    game = rig.session.game
    game.set_level(levels.LEVELS[record["level"]])                  # the game applies any level setting itself
    game.set_mode(record["mode"])
    game.start(record["started_at_ns"])


def replay(loaded, *, overrides=None, record_dir=None):
    """Run a recorded session again; -> ReplayResult (and a new recording if record_dir is given)."""
    meta, units = loaded.meta, loaded.meta["units"]
    overrides = overrides_mod.merge(meta.get("overrides"), check_overrides(overrides or {}))   # the recording's own, then yours
    clock = FakeClock(start_ns=meta["t0_ns"])
    hub, lag_ns = ReplayHub(), round(meta["camera_lag_s"] * S)
    hub._clock = clock
    vision = ReplayVision(lag_ns)
    probs_by_feat = {tuple(e["d"]["feat"]): e["d"]["spin_probs"] for e in loaded.events
                     if e["k"] == "swing" and "spin_probs" in e["d"]}
    inner = recorder_mod.Recorder(record_dir, {**meta, "source": "replay", "overrides": overrides}, clock=clock) \
        if record_dir else None
    collector = _Collector(inner)
    rig = live.assemble(
        hub=hub, capture=None, landmarker=None, vision=vision, clock=clock, recorder=collector,
        calibration=Calibration.from_json(json.dumps(meta["calibration"])), level=meta["level"], mode=meta["mode"],
        target=meta["target"], seed=meta["seed"], source="replay", scope=meta["scope"], no_motor=meta["no_motor"],
        gyro_per_dps=units["gyro_per_dps"], accel_per_g=units["accel_per_g"], fs_raw=units["fs_raw"],
        stale_ms=meta["stale_ms"], threaded=False, log=lambda *_: None,
        spin_probs_fn=(lambda feat: probs_by_feat.get(tuple(feat))) if probs_by_feat else None, overrides=overrides)
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
    camera = rig.swing_source == "pose"
    # A camera sample is made from a pose the moment that pose is read: lag + estimator delay after its stamp,
    # together with the pose.  A hub sample is stamped when it arrives.
    imu_delay_ns = round((meta["camera_lag_s"] + posegyro.DELAY_S) * S) if camera else 0

    def deliver_imu(now):
        nonlocal i_imu
        while i_imu < len(loaded.imu) and loaded.imu[i_imu].t_ns + imu_delay_ns <= now:
            hub.deliver(loaded.imu[i_imu])
            i_imu += 1

    def deliver_poses(now):
        nonlocal i_pose
        while i_pose < len(poses) and poses[i_pose].t_scene_ns + lag_ns <= now:
            vision.deliver(poses[i_pose])
            i_pose += 1

    while clock.now_ns() < end_ns:
        clock.advance_s(STEP_NS / S)
        now = clock.now_ns()
        for deliver in ((deliver_poses, deliver_imu) if camera else (deliver_imu, deliver_poses)):
            deliver(now)
        while i_start < len(starts) and starts[i_start]["started_at_ns"] <= now:
            _start(rig, starts[i_start])
            i_start += 1
        while next_pump <= now:
            rig.pump()
            next_pump += PUMP_NS
    rig.close()
    warnings = ["the session used the learning opponent (--learn): the computer's serves depend on what it had "
                "learned, so this replay may diverge from what happened"] if meta.get("learn") else []
    return ReplayResult(rig=rig, events=collector.events, warnings=warnings)
