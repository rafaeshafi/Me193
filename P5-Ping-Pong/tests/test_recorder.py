"""Recorder: a live session written as JSON lines, so a bad run becomes an offline fixture.

Three streams (every IMU sample, every pose, every game event) plus a session.json with the
seed, level, calibration and units: enough to replay the session through the real code and write
a failing test for whatever went wrong.  Recording must never break the game.
"""

import json
import threading

import pytest

from pingpong import recorder
from pingpong.clock import FakeClock
from pingpong.events import GateResult, ImuSample, PaddlePose, Verdict

S = 1_000_000_000


def make(tmp_path, **kw):
    clock = FakeClock(start_ns=10 * S)
    rec = recorder.Recorder(tmp_path / "run", {"seed": 7, "level": 2}, clock=clock, **kw)
    return rec, clock


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sample(i):
    return ImuSample(t_ns=10 * S + i * 15_000_000, g=(i, -i, 2 * i), a=(0, 0, 1000))


def test_the_session_description_is_on_disk_straight_away(tmp_path):
    make(tmp_path)
    meta = json.loads((tmp_path / "run" / "session.json").read_text())
    assert meta["seed"] == 7 and meta["level"] == 2 and meta["version"] == recorder.VERSION


def test_streams_become_json_lines_after_a_flush_and_not_before(tmp_path):
    rec, _ = make(tmp_path)
    for i in range(3):
        rec.imu(sample(i))
    rec.pose(PaddlePose(t_scene_ns=10 * S, u=0.5, v=-0.2, conf=0.9, hand="right"))
    rec.event("serve", 10 * S, {"ball_id": 1, "aim_ab": (0.5, 0.5)})
    assert (tmp_path / "run" / "imu.jsonl").read_text() == ""
    rec.flush()
    imu = lines(tmp_path / "run" / "imu.jsonl")
    assert [row["g"] for row in imu] == [[0, 0, 0], [1, -1, 2], [2, -2, 4]]
    assert lines(tmp_path / "run" / "pose.jsonl")[0] == {"t": 10 * S, "u": 0.5, "v": -0.2, "c": 0.9, "h": "right"}
    assert lines(tmp_path / "run" / "events.jsonl")[0] == {"t": 10 * S, "k": "serve",
                                                           "d": {"ball_id": 1, "aim_ab": [0.5, 0.5]}}


def test_tick_flushes_on_the_interval_so_a_crash_loses_at_most_a_second(tmp_path):
    rec, clock = make(tmp_path, flush_interval_s=1.0)
    rec.imu(sample(0))
    rec.tick(clock.now_ns())
    assert (tmp_path / "run" / "imu.jsonl").read_text() == ""
    clock.advance_s(1.1)
    rec.tick(clock.now_ns())
    assert len(lines(tmp_path / "run" / "imu.jsonl")) == 1


def test_game_events_are_flattened_to_plain_json_including_their_dataclasses(tmp_path):
    from pingpong.events import GameEvent

    rec, _ = make(tmp_path)
    verdict = Verdict("HIT", 0.8, -0.02, 0.1, (GateResult("J1", True, "timing -20 ms"), GateResult("J2", False, "far")))
    rec.game_events([GameEvent("verdict", 11 * S, {"verdict": verdict}), GameEvent("miss", 12 * S, {"ball_id": 3})])
    rec.flush()
    events = lines(tmp_path / "run" / "events.jsonl")
    assert events[0]["k"] == "verdict" and events[0]["d"]["verdict"]["kind"] == "HIT"
    assert events[0]["d"]["verdict"]["gates"][1] == {"name": "J2", "passed": False, "note": "far"}
    assert events[1] == {"t": 12 * S, "k": "miss", "d": {"ball_id": 3}}


def test_everything_survives_a_load_round_trip(tmp_path):
    rec, _ = make(tmp_path)
    originals = [sample(i) for i in range(50)]
    for s in originals:
        rec.imu(s)
    pose = PaddlePose(t_scene_ns=10 * S + 5, u=-0.4, v=0.3, conf=0.7, hand="left")
    rec.pose(pose)
    rec.event("tag", 11 * S, {"role": "START", "value": 0})
    rec.close()
    loaded = recorder.load(tmp_path / "run")
    assert loaded.meta["seed"] == 7
    assert loaded.imu == originals and loaded.poses == [pose]
    assert loaded.events == [{"t": 11 * S, "k": "tag", "d": {"role": "START", "value": 0}}]


def test_the_imu_thread_and_the_flushing_thread_never_lose_or_reorder_samples(tmp_path):
    rec, _ = make(tmp_path)
    stop = threading.Event()

    def producer():
        for i in range(5000):
            rec.imu(sample(i))
        stop.set()

    thread = threading.Thread(target=producer)
    thread.start()
    while not stop.is_set():
        rec.flush()
    thread.join()
    rec.close()
    rows = lines(tmp_path / "run" / "imu.jsonl")
    assert [row["g"][0] for row in rows] == list(range(5000))


def test_a_write_failure_disables_recording_and_never_reaches_the_game(tmp_path):
    messages = []
    rec, _ = make(tmp_path, log=messages.append)
    rec.imu(sample(0))
    rec._files["imu"].close()                         # the disk "disappears" under it
    rec.flush()                                       # must not raise
    rec.imu(sample(1))
    rec.flush()
    rec.close()
    assert len(messages) == 1 and "recording" in messages[0].lower()      # reported once, not every flush


def test_a_directory_that_cannot_be_created_is_an_error_at_construction(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        recorder.Recorder(blocker / "run", {}, clock=FakeClock())


def test_close_is_idempotent_and_flushes_what_is_left(tmp_path):
    rec, _ = make(tmp_path)
    rec.imu(sample(0))
    rec.close()
    rec.close()
    assert len(lines(tmp_path / "run" / "imu.jsonl")) == 1


def test_numpy_numbers_and_tuples_do_not_break_the_json(tmp_path):
    import numpy as np

    rec, _ = make(tmp_path)
    rec.event("hit", 1, {"kmh": np.float64(23.5), "gates": (1, 2), "n": np.int64(4)})
    rec.close()
    assert lines(tmp_path / "run" / "events.jsonl")[0]["d"] == {"kmh": 23.5, "gates": [1, 2], "n": 4}


def test_session_directories_are_named_by_time_and_player(tmp_path):
    name = recorder.session_name("Rafae Shafi", wall=(2026, 10, 7, 9, 5, 3))
    assert name == "20261007-090503-rafae-shafi"
    assert recorder.default_root().name == "recordings"


def test_an_existing_session_is_never_overwritten(tmp_path):
    make(tmp_path)
    with pytest.raises(FileExistsError):
        make(tmp_path)
