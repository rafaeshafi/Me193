"""LiveRig: sensors -> Session each frame, pause on sensor loss, reconnect, ordered teardown.

These tests use stand-in sensors so every behaviour is pinned down on its own; the whole
pipeline on fake hardware is exercised in tests/test_fakerig.py.
"""

import numpy as np
import pytest

from pingpong import app, live
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose, SwingEvent, TagEvent
from pingpong.keys import fake_swing

S = 1_000_000_000


class Hub:
    """HubLink-shaped; silence is simulated by not calling beat()."""

    def __init__(self, clock, log):
        self.clock, self.log = clock, log
        self.last_rx_ns = clock.now_ns()
        self.reconnects, self.fail_reconnect = 0, False

    def beat(self):
        self.last_rx_ns = self.clock.now_ns()

    def stale_ms(self):
        return None if self.last_rx_ns is None else (self.clock.now_ns() - self.last_rx_ns) / 1e6

    def is_stale(self, threshold_ms):
        age = self.stale_ms()
        return age is None or age > threshold_ms

    def battery_pct(self):
        return None

    def reconnect(self):
        self.reconnects += 1
        self.log.append("hub.reconnect")
        if self.fail_reconnect:
            raise ConnectionError("not found")
        self.beat()

    def close(self):
        self.log.append("hub.close")


class Vision:
    def __init__(self, clock, log):
        self.clock, self.log = clock, log
        self.poses, self.tags, self.last_read_ns, self.frame = (), [], clock.now_ns(), None

    def beat(self):
        self.last_read_ns = self.clock.now_ns()

    def snapshot(self):
        return self.poses

    def poll_tags(self):
        out, self.tags = self.tags, []
        return out

    def pose_age_s(self, now_ns):
        return None if self.last_read_ns is None else (now_ns - self.last_read_ns) / 1e9

    def latest_frame(self):
        return self.frame

    def start(self):
        self.log.append("vision.start")

    def stop(self):
        self.log.append("vision.stop")


class Imu:
    def __init__(self, log):
        self.log, self.events, self.steps, self.locks = log, [], 0, []

    def step(self):
        self.steps += 1

    def poll(self):
        out, self.events = self.events, []
        return out

    def blank(self, a, b):
        pass

    def trace(self, seconds):
        return [(0.0, 100.0), (0.1, 450.0)]

    def poll_locks(self):
        out, self.locks = self.locks, []
        return out

    def start(self):
        self.log.append("imu.start")

    def stop(self):
        self.log.append("imu.stop")


class Actuator:
    def __init__(self, log):
        self.log, self.names = log, []

    def submit(self, name, fire_at_ns=None):
        self.names.append(name)
        return True

    def start(self):
        self.log.append("actuator.start")

    def stop(self):
        self.log.append("actuator.stop")


class Rec:
    """Recorder-shaped: remembers what it was given."""

    def __init__(self, log):
        self.log, self.poses, self.events, self.ticks = log, [], [], 0

    def pose(self, pose):
        self.poses.append(pose)

    def event(self, kind, t_ns, data=None):
        self.events.append((kind, data or {}))

    def game_events(self, events):
        for e in events:
            self.events.append((e.kind, e.data))

    def tick(self, now_ns):
        self.ticks += 1

    def close(self):
        self.log.append("recorder.close")


class Rig:
    """A LiveRig on stand-ins, plus helpers that advance time while beating chosen sensors."""

    def __init__(self, **kw):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.log = []
        self.hub, self.vision = Hub(self.clock, self.log), Vision(self.clock, self.log)
        self.imu, self.actuator = Imu(self.log), Actuator(self.log)
        self.recorder = Rec(self.log) if kw.pop("record", False) else None
        if self.recorder is not None:
            kw["recorder"] = self.recorder
        self.session = app.make_session(clock=self.clock, actuator=self.actuator, client=kw.get("mqtt_client"))
        self.rig = live.LiveRig(self.session, hub=self.hub, imu=self.imu, vision=self.vision,
                                actuator=self.actuator, clock=self.clock, **kw)
        self.game = self.session.game

    def advance(self, seconds, hub=True, pose=True, dt=0.05):
        trace = []
        for _ in range(round(seconds / dt)):
            self.clock.advance_s(dt)
            if hub:
                self.hub.beat()
            if pose:
                self.vision.beat()
            self.rig.pump()
            trace.append((self.clock.now_ns(), self.game.paused))
        return trace

    def into_rally(self):
        self.session.on_start()
        self.advance(3.2)
        assert self.game.phase == "RALLY" and self.game.incoming is not None


# --- feeding the session ------------------------------------------------------------------------
def test_each_new_pose_reaches_the_session_exactly_once_in_order():
    r = Rig()
    p1 = PaddlePose(t_scene_ns=1, u=0.1, v=0.2, conf=0.9, hand="right")
    p2 = PaddlePose(t_scene_ns=2, u=0.2, v=0.3, conf=0.9, hand="right")
    r.vision.poses = (p1,)
    r.rig.pump()
    r.vision.poses = (p1, p2)
    r.rig.pump()
    r.rig.pump()
    assert list(r.session.poses) == [p1, p2]


def test_a_start_tag_from_the_camera_starts_the_countdown():
    r = Rig()
    r.vision.tags = [TagEvent("START", 0, r.clock.now_ns())]
    r.rig.pump()
    assert r.game.phase == "COUNTDOWN"


def test_a_level_tag_from_the_camera_sets_the_level():
    r = Rig()
    r.vision.tags = [TagEvent("LEVEL", 3, r.clock.now_ns())]
    r.rig.pump()
    assert r.game.level.name == "Pro"


def test_only_impacts_reach_the_judge_never_swing_starts():
    r = Rig()
    seen = []
    r.session.on_swing = seen.append
    impact = fake_swing(r.clock.now_ns(), 600.0)
    start = SwingEvent(**{**impact.__dict__, "kind": "SWING_START"})
    r.imu.events = [start, impact]
    r.rig.pump()
    assert seen == [impact]
    assert r.rig.n_impacts == 1


def test_sync_mode_drains_the_imu_queue_itself_and_threaded_mode_leaves_it_to_the_thread():
    sync, threaded = Rig(threaded=False), Rig(threaded=True)
    sync.rig.pump()
    threaded.rig.pump()
    assert (sync.imu.steps, threaded.imu.steps) == (1, 0)


def test_a_shake_lock_from_the_imu_worker_reaches_the_judge():
    r = Rig()
    locked = []
    r.game.judge.lock_paddle = locked.append
    until = r.clock.now_ns() + S
    r.imu.locks = [until]
    r.rig.pump()
    assert locked == [until]


def test_the_display_frame_is_the_mirror_image_of_the_camera_frame():
    r = Rig()
    assert r.rig.display_frame() is None
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 0] = 255                                        # a bright left edge
    r.vision.frame = frame
    shown = r.rig.display_frame()
    assert shown[0, 5, 0] == 255 and shown[0, 0, 0] == 0     # ... appears on the right: like a mirror


# --- sensor loss pauses the game, never scores a miss -----------------------------------------------
def test_hub_silence_pauses_a_running_rally_and_the_ball_resumes_exactly_where_it_was():
    # "Where it was" means: where it was when the hub was last heard, not when the silence was
    # noticed -- the stale threshold must not eat into the player's flight time.
    r = Rig(stale_ms=300.0)
    r.hub.fail_reconnect = True                              # the silence lasts: no rescue by a reconnect
    r.into_rally()
    r.advance(0.2)
    ball, last_heard = r.game.incoming, r.clock.now_ns()
    r.advance(1.0, hub=False)
    assert r.game.paused and r.game.pause_reasons == {"hub"}
    assert r.game.phase == "RALLY" and r.game.tracker.streak == 0            # no miss, no fault
    r.advance(4.0, hub=False)                                                # far past the miss deadline
    assert r.game.phase == "RALLY"
    r.advance(0.05)                                                          # the hub is back
    resumed_at = r.clock.now_ns()
    assert not r.game.paused
    assert r.game.incoming.t_c_ns - resumed_at == pytest.approx(ball.t_c_ns - last_heard, abs=1)


def test_losing_the_pose_for_longer_than_the_limit_pauses_and_it_resumes_when_seen_again():
    r = Rig()
    r.into_rally()
    r.advance(live.POSE_STALE_S - 0.2, pose=False)
    assert not r.game.paused                                 # a short dropout is just a dropout
    r.advance(0.6, pose=False)
    assert r.game.paused and r.game.pause_reasons == {"pose"}
    assert "pose" in r.session.hud_state().message
    r.advance(0.1)
    assert not r.game.paused and r.game.phase == "RALLY"


def test_both_sensors_lost_pause_for_both_reasons_and_resume_only_when_both_are_back():
    r = Rig(stale_ms=300.0)
    r.hub.fail_reconnect = True
    r.into_rally()
    r.advance(1.5, hub=False, pose=False)
    assert r.game.pause_reasons == {"hub", "pose"}
    r.hub.beat()
    r.rig.pump()
    assert r.game.pause_reasons == {"pose"}
    r.advance(0.1)
    assert not r.game.paused


def test_silent_sensors_do_not_pause_the_lobby_so_the_start_tag_can_still_be_shown():
    r = Rig(stale_ms=300.0)
    r.advance(5.0, hub=False, pose=False)
    assert not r.game.paused and r.game.phase == "LOBBY"


def test_the_countdown_waits_for_the_sensors_instead_of_serving_a_ball_nobody_can_hit():
    r = Rig(stale_ms=300.0)
    r.hub.fail_reconnect = True
    r.session.on_start()
    r.advance(6.0, hub=False)
    assert r.game.phase == "COUNTDOWN" and r.game.paused
    r.advance(3.5)
    assert r.game.phase == "RALLY"


def test_a_pose_that_was_never_seen_counts_as_lost_once_the_game_is_running():
    r = Rig()
    r.vision.last_read_ns = None
    r.session.on_start()
    r.rig.pump()
    assert r.game.pause_reasons == {"pose"}


def test_a_hub_gap_around_the_swing_window_is_a_pause_never_a_miss():
    # The pause only starts after STALE_MS of silence; by then the wall clock is past the miss
    # deadline.  The rig tells the game how far the IMU data actually reaches, so the missing
    # swing is not mistaken for no swing.
    r = Rig(stale_ms=500.0)
    r.hub.fail_reconnect = True
    r.into_rally()
    ball = r.game.incoming
    r.advance((ball.t_c_ns - r.clock.now_ns()) / S - 0.1)            # healthy until 100 ms before arrival
    r.advance(0.9, hub=False)                                        # ... then the hub goes quiet
    assert r.game.phase == "RALLY" and r.game.paused
    r.advance(0.1)
    assert not r.game.paused and r.game.phase == "RALLY"


def test_an_impact_that_arrives_while_paused_is_not_a_hit():
    r = Rig(stale_ms=300.0)
    r.into_rally()
    r.advance(1.0, hub=False)
    r.imu.events = [fake_swing(r.clock.now_ns(), 600.0)]
    r.clock.advance_s(0.05)
    r.rig.pump()
    assert r.game.tracker.streak == 0 and r.game.paused


# --- hub status text and reconnecting ---------------------------------------------------------------------
def test_the_hud_hub_indicator_follows_the_link_health():
    r = Rig(stale_ms=300.0)
    r.session.bind_status(hub=r.rig.hub_status)
    assert r.session.hud_state().hub_status == "ok"
    r.advance(0.6, hub=False)
    assert r.session.hud_state().hub_status == "stale"
    r.advance(0.1)
    assert r.session.hud_state().hub_status == "ok"


def test_a_silent_hub_is_reconnected_after_the_grace_period_and_then_only_after_the_cooldown():
    r = Rig(stale_ms=300.0)
    r.hub.fail_reconnect = True
    r.advance(live.RECONNECT_AFTER_S - 0.5, hub=False)
    assert r.hub.reconnects == 0
    r.advance(1.0, hub=False)
    assert r.hub.reconnects == 1
    r.advance(live.RECONNECT_COOLDOWN_S - 2.0, hub=False)
    assert r.hub.reconnects == 1
    r.advance(3.0, hub=False)
    assert r.hub.reconnects == 2


def test_a_successful_reconnect_clears_the_attempt_counter_and_the_status():
    r = Rig(stale_ms=300.0)
    r.advance(live.RECONNECT_AFTER_S + 0.5, hub=False)
    assert r.hub.reconnects == 1
    r.advance(1.0)                                           # samples flow again
    assert r.rig.hub_status() == "ok"
    r.advance(live.RECONNECT_AFTER_S + 0.5, hub=False)       # a second, separate outage: a fresh budget
    assert r.hub.reconnects == 2


def test_it_gives_up_after_the_attempt_limit_until_the_player_asks_again():
    r = Rig(stale_ms=300.0)
    r.hub.fail_reconnect = True
    r.advance(live.RECONNECT_COOLDOWN_S * (live.MAX_RECONNECTS + 2), hub=False, dt=0.25)
    assert r.hub.reconnects == live.MAX_RECONNECTS
    assert r.rig.hub_status() == "lost"
    r.rig.reconnect_now()
    r.advance(0.5, hub=False)
    assert r.hub.reconnects == live.MAX_RECONNECTS + 1


def test_a_failing_reconnect_is_logged_and_never_raised():
    messages = []
    r = Rig(stale_ms=300.0, log=messages.append)
    r.hub.fail_reconnect = True
    r.advance(live.RECONNECT_AFTER_S + 0.5, hub=False)
    assert any("reconnect" in m for m in messages)


# --- teardown ---------------------------------------------------------------------------------------------------
def test_close_tears_down_in_the_documented_order():
    r = Rig()
    r.rig.close()
    assert r.log == ["actuator.stop", "vision.stop", "imu.stop", "hub.close"]


def test_close_is_idempotent_and_a_failing_step_never_stops_the_rest():
    r = Rig(log=lambda *_: None)

    def boom():
        raise RuntimeError("stuck")

    r.vision.stop = boom
    r.rig.close()
    r.rig.close()
    assert r.log == ["actuator.stop", "imu.stop", "hub.close"]


def test_close_hangs_up_the_mqtt_client_after_the_sensors_are_stopped():
    from pingpong.sources_fake import FakeMqttClient

    client = FakeMqttClient()
    r = Rig(mqtt_client=client)
    r.rig.close()
    assert ("disconnect",) in client.log and r.log[-1] == "hub.close"


def test_starting_a_threaded_rig_starts_every_worker_and_a_sync_rig_starts_none():
    threaded, sync = Rig(threaded=True), Rig(threaded=False)
    threaded.rig.start()
    sync.rig.start()
    assert sorted(threaded.log) == ["actuator.start", "imu.start", "vision.start"]
    assert sync.log == []


def test_the_hud_state_carries_the_imu_trace_and_the_thresholds_it_is_judged_against():
    r = Rig()
    state = r.rig.hud_state()
    assert state.swing_trace == (100.0, 450.0)
    assert state.swing_threshold == r.game.judge.t_pk and state.swing_scale == r.game.omega_hi
    assert state.phase == r.session.hud_state().phase


def test_the_rig_records_poses_swings_game_events_tags_pauses_and_phase_changes():
    r = Rig(record=True, stale_ms=300.0)
    r.hub.fail_reconnect = True
    pose = PaddlePose(t_scene_ns=1, u=0.1, v=0.2, conf=0.9, hand="right")
    r.vision.poses = (pose,)
    r.vision.tags = [TagEvent("START", 0, r.clock.now_ns())]
    r.rig.pump()
    kinds = [k for k, _ in r.recorder.events]
    assert r.recorder.poses == [pose] and "tag" in kinds and "phase" in kinds
    phase = [d for k, d in r.recorder.events if k == "phase"][-1]
    assert phase["phase"] == "COUNTDOWN" and phase["started_at_ns"] == r.game.started_at_ns
    assert phase["level"] == 1 and phase["mode"] == "survival"
    r.advance(3.2)                                            # the first ball is served
    r.imu.events = [fake_swing(r.clock.now_ns(), 600.0)]
    r.advance(0.05)
    swing = [d for k, d in r.recorder.events if k == "swing"]
    assert swing and swing[0]["w_pk"] == 600.0 and len(swing[0]["feat"]) == 12
    assert "serve" in [k for k, _ in r.recorder.events]
    r.advance(1.0, hub=False)
    assert [d["reasons"] for k, d in r.recorder.events if k == "pause"][-1] == ["hub"]
    assert r.recorder.ticks > 0


def test_a_recorder_is_closed_after_the_sensors_stop_and_before_the_hub_lets_go():
    r = Rig(record=True)
    r.rig.close()
    assert r.log == ["actuator.stop", "vision.stop", "imu.stop", "recorder.close", "hub.close"]


def test_pump_timings_are_collected_for_the_report():
    r = Rig()
    for _ in range(5):
        r.rig.pump()
    stats = r.rig.loop_stats()
    assert stats["n"] == 5 and stats["p95_ms"] >= 0.0


def test_the_store_is_closed_after_the_recorder_and_before_the_hub():
    r = Rig(record=True)

    class Store:
        def close(self):
            r.log.append("store.close")

    r.rig.store = Store()
    r.rig.close()
    assert r.log == ["actuator.stop", "vision.stop", "imu.stop", "recorder.close", "store.close", "hub.close"]
