"""The camera as the swing sensor: the whole live pipeline when the hub's IMU is too slow to use.

A scripted player whose hand really swings (the camera sees the stroke, the hub's own gyro is ignored) plays
through the real vision worker, PoseGyro, swing detector, judge and game.
"""

import json

import pytest

import config
from pingpong import fakerig, recorder, replay

OFFICIAL = config.SCORE_TOPIC


def camera_rig(**kw):
    return fakerig.FakeRig(swing_source="pose", **kw)


def payloads(rig):
    return [p["payload"] for p in rig.client.published if p["topic"] == OFFICIAL]


def test_a_scripted_player_returns_ten_balls_with_the_camera_as_the_only_swing_sensor():
    rig = camera_rig(level=1, hz=8.0)                                   # a hub far too slow to detect a swing
    rig.run(until=lambda: rig.game.tracker.streak >= 10, max_s=120)
    assert payloads(rig) == [f"{n}.0" for n in range(1, 11)]
    assert rig.rig.swing_source == "pose" and rig.rig.n_impacts >= 10


def test_club_level_is_playable_from_the_camera_too():
    rig = camera_rig(level=2)
    rig.run(until=lambda: rig.game.tracker.streak >= 6, max_s=120)
    assert rig.game.tracker.streak >= 6


def test_the_hub_gyro_is_ignored_in_camera_mode_and_the_hub_queue_does_not_pile_up(tmp_path):
    rig = camera_rig(record_dir=tmp_path / "s")
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    rig.close()
    assert rig.rig.hub.imu.empty() and rig.rig.hub.n_samples > 100        # the hub kept talking; nobody keeps it
    rows = [json.loads(line) for line in (tmp_path / "s" / "imu.jsonl").read_text().splitlines()]
    assert rows and all(r.get("s") == "pose" for r in rows)               # the recording holds what the detector saw


def test_the_hub_going_silent_does_not_pause_a_camera_game():
    rig = camera_rig(stale_ms=300.0)
    rig.run(until=lambda: rig.game.phase == "RALLY")
    now = rig.now_s()
    rig.hub_blackouts.append((now + 0.2, now + 2.0))
    rig.run(until=lambda: rig.game.tracker.streak >= 4, max_s=60)
    assert "hub" not in rig.pause_reasons_seen


def test_losing_the_camera_pauses_a_camera_game_and_the_rally_carries_on():
    rig = camera_rig()
    rig.run(until=lambda: rig.game.phase == "RALLY")
    now = rig.now_s()
    rig.pose_blackouts.append((now + 0.2, now + 1.8))
    rig.run(until=lambda: rig.game.tracker.streak >= 5, max_s=60)
    assert "pose" in rig.pause_reasons_seen and rig.game.tracker.streak >= 5


def test_haptic_pulses_do_not_blank_the_camera_swing_detector():
    assert camera_rig().rig.actuator.on_blank is None                    # the camera does not feel the motors
    assert fakerig.FakeRig().rig.actuator.on_blank is not None            # the hub's gyro does


def test_waving_the_hand_about_locks_the_paddle_so_the_ball_arriving_meanwhile_is_not_a_hit(tmp_path):
    rig = camera_rig(record_dir=tmp_path / "wave")
    rig.run(until=lambda: rig.game.phase == "COUNTDOWN")
    rig.wave_windows.append((rig.now_s(), rig.now_s() + 8.0))            # a 5 Hz wave of the hand
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=30)
    rig.close()
    assert rig.game.tracker.record == 0 and payloads(rig) == []
    kinds = [e["k"] for e in recorder.load(tmp_path / "wave").events]
    assert "shake_lock" in kinds                                         # the FFT monitor saw it on the camera's samples


def record(tmp_path, name="orig", until=None, **kw):
    rig = camera_rig(record_dir=tmp_path / name, **kw)
    rig.run(until=until or (lambda: rig.game.tracker.streak >= 5), max_s=120)
    rig.close()
    return recorder.load(tmp_path / name)


def test_a_camera_session_says_so_in_its_recording_and_replays_to_the_same_decisions(tmp_path):
    original = record(tmp_path)
    assert original.meta["calibration"]["swing"]["source"] == "pose"
    assert original.meta["units"]["gyro_per_dps"] == 10.0
    result = replay.replay(original, record_dir=tmp_path / "again")
    again = recorder.load(tmp_path / "again")
    decide = lambda s: [(e["k"], e["t"]) for e in s.events if e["k"] in ("serve", "hit", "miss", "fault")]  # noqa: E731
    assert decide(again) == decide(original) and result.hits == 5


def test_a_camera_replay_with_a_stricter_threshold_turns_hits_into_non_events(tmp_path):
    original = record(tmp_path, until=None)
    strict = replay.replay(original, overrides={"swing": {"t_pk": 1500.0}})
    assert strict.hits == 0 and replay.replay(original).hits == 5


def test_the_session_report_works_on_a_camera_session(tmp_path):
    from pingpong import sessionreport

    summary = sessionreport.summarize(record(tmp_path))
    assert summary["hits"] == 5 and summary["swing_source"] == "pose"
    text = sessionreport.format_report(summary)
    assert "camera" in text.lower() and "Hub:" not in text                # the 'hub' rate would be the camera's
