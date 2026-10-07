"""The live pipeline in REAL time with REAL threads (what `./pp play` runs on the Mac), on fake hardware.

fakerig.FakeRig steps a simulated clock on one thread: that tests the logic, not the concurrency.  Here a "BLE" thread
notifies the hub callback at 66 Hz, the camera thread blocks in read() at 30 fps, the actuator and IMU workers are
the real threads and the game thread pumps at ~60 Hz.  A deadlock, a race on a shared list, an exception that a worker
swallows or a thread that survives close() would show up here and nowhere else short of the hardware.
"""

import threading
import time

import pytest

import config
from pingpong.threadrig import ThreadedFakeRig

OFFICIAL = config.SCORE_TOPIC
OUR_THREADS = ("imu", "vision", "actuator", "fake-ble")


def our_live_threads():
    return [t.name for t in threading.enumerate() if t.name in OUR_THREADS and t.is_alive()]


@pytest.fixture
def rig():
    made = []

    def make(**kw):
        made.append(ThreadedFakeRig(**kw))
        return made[-1]

    yield make
    for r in made:                                       # a failing test must not leave threads behind
        r.close()


def test_a_rally_in_real_time_survives_a_quiet_hub_and_shuts_down_cleanly_with_a_pattern_still_queued(rig):
    r = rig(level=1, stale_ms=300.0)
    r.start()                                                              # the START card starts it, via the vision thread
    assert {"imu", "vision", "actuator", "fake-ble"} <= set(our_live_threads())
    r.run(until=lambda: r.game.phase == "RALLY", max_s=30)
    now = r.now_s()
    r.hub_blackouts.append((now + 0.3, now + 1.6))                         # the hub goes silent mid-ball
    r.run(until=lambda: r.game.tracker.streak >= 2 and "hub" in r.pause_reasons_seen, max_s=40)
    r.session.actuator.submit("record")                                    # a 5-pulse pattern in the queue
    began = time.monotonic()
    r.close()
    r.close()                                                              # closing twice is safe
    assert time.monotonic() - began < 3.0                                  # every join returned
    assert our_live_threads() == []
    assert r.thread_errors == [] and r.logs == []                          # nothing died, nothing was swallowed
    assert [p["payload"] for p in r.client.published if p["topic"] == OFFICIAL][:2] == ["1.0", "2.0"]
    assert r.rig.loop_stats()["p95_ms"] < 40.0
    assert [c[0] for c in r.dev.calls][-2:] == ["motor_stop", "disconnect"]


def test_the_camera_alone_is_enough_in_real_time_too(rig):
    r = rig(level=1, swing_source="pose", hz=8.0, cards=[])
    r.start()
    r.session.on_start()                                                   # a key press instead of the card
    r.run(until=lambda: r.game.tracker.streak >= 2, max_s=45)
    r.close()
    assert r.thread_errors == [] and r.logs == [] and r.game.tracker.record >= 2 and r.rig.swing_source == "pose"
    assert our_live_threads() == []
