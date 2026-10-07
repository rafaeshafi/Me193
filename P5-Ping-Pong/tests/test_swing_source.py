"""Choosing the swing sensor: the hub's gyro or the camera (and playing with no hub at all).

--swing-source auto looks at what the hub bench measured: a hub below 25 Hz is too slow to see a 150 ms swing, so the
camera's hand speed takes over (the hub stays for haptics).  --no-hub is the camera-only bring-up mode.
"""

import pytest

import play
from pingpong import live, profile
from pingpong.calibration import SwingCalibration
from pingpong.paddle import ReachBox
from pingpong.sources_fake import FakeEnv


def args_for(*extra):
    return play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-record", "--no-store", *extra])


def saved(source, lo, hi):
    return profile.Calibration(swing=SwingCalibration((1.0, 0.0, 0.0), lo, hi, source=source),
                               box=ReachBox(-1.5, 1.5, -0.9, 0.7), shoulder_w=0.2)


def test_the_hub_is_the_default_swing_sensor(tmp_path):
    rig = live.build_live(args_for("--player", "rafae"), FakeEnv(), player_root=tmp_path)
    assert rig.swing_source == "imu" and rig.pose_gyro is None


@pytest.mark.parametrize("rate, expected", [(18.0, "pose"), (24.9, "pose"), (25.0, "imu"), (66.0, "imu"), (None, "imu")])
def test_auto_picks_the_camera_only_when_the_measured_hub_rate_is_below_25_hz(tmp_path, monkeypatch, rate, expected):
    monkeypatch.setattr(live.config, "HUB_RATE_HZ", rate)
    messages = []
    rig = live.build_live(args_for("--player", "rafae"), FakeEnv(), player_root=tmp_path, log=messages.append)
    assert rig.swing_source == expected
    says = any("camera" in m and rate is not None and f"{rate:.0f}" in m for m in messages)
    assert says == (expected == "pose")                                                      # and it says so


def test_an_explicit_swing_source_beats_the_measured_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(live.config, "HUB_RATE_HZ", 18.0)
    forced = live.build_live(args_for("--player", "rafae", "--swing-source", "imu"), FakeEnv(), player_root=tmp_path)
    assert forced.swing_source == "imu"
    monkeypatch.setattr(live.config, "HUB_RATE_HZ", 66.0)
    camera = live.build_live(args_for("--player", "rafae", "--swing-source", "pose"), FakeEnv(), player_root=tmp_path)
    assert camera.swing_source == "pose" and camera.pose_gyro is not None


def test_the_camera_rig_loads_the_camera_calibration_and_never_the_hub_one(tmp_path):
    profile.save("rafae", saved("imu", 300.0, 1100.0), root=tmp_path)
    profile.save("rafae", saved("pose", 240.0, 760.0), root=tmp_path)
    rig = live.build_live(args_for("--player", "rafae", "--swing-source", "pose"), FakeEnv(), player_root=tmp_path)
    assert rig.calibration.swing.source == "pose" and rig.calibration.swing.omega_lo == 240.0
    assert rig.session.game.omega_hi == 760.0
    hub = live.build_live(args_for("--player", "rafae", "--swing-source", "imu"), FakeEnv(), player_root=tmp_path)
    assert hub.calibration.swing.omega_lo == 300.0


def test_a_player_with_only_a_hub_calibration_is_told_how_to_make_a_camera_one(tmp_path):
    profile.save("rafae", saved("imu", 300.0, 1100.0), root=tmp_path)
    rig = live.build_live(args_for("--player", "rafae", "--swing-source", "pose"), FakeEnv(), player_root=tmp_path)
    assert rig.calibration.calibrated is False and rig.calibration.swing.source == "pose"
    assert "--swing-source pose" in rig.session.hud_state().message
    hub = live.build_live(args_for("--player", "newbie"), FakeEnv(), player_root=tmp_path)
    assert "--swing-source" not in hub.session.hud_state().message         # the hub path's hint is unchanged


def test_spin_is_off_with_the_camera_because_it_needs_the_hubs_accelerometer(tmp_path):
    from pingpong import fakerig, spin

    X, y = fakerig.spin_dataset(10, seed=4)
    model, report = spin.train(X, y)
    spin.save("rafae", model, report, root=tmp_path)
    messages = []
    camera = live.build_live(args_for("--player", "rafae", "--swing-source", "pose"), FakeEnv(), player_root=tmp_path,
                             log=messages.append)
    assert camera.session.game.spin_probs_fn is None and any("spin" in m.lower() for m in messages)
    hub = live.build_live(args_for("--player", "rafae", "--swing-source", "imu"), FakeEnv(), player_root=tmp_path)
    assert hub.session.game.spin_probs_fn is not None


def test_the_hub_is_still_connected_for_haptics_when_the_camera_is_the_swing_sensor(tmp_path):
    env = FakeEnv()
    rig = live.build_live(args_for("--player", "rafae", "--swing-source", "pose"), env, player_root=tmp_path)
    assert [c[0] for c in env.hub_device.calls][:1] == ["connect"] and rig.actuator is not None
    assert rig.hub_status() == "ok"


# --- no hub at all --------------------------------------------------------------------------------------------------
def test_no_hub_runs_on_the_camera_alone_without_a_card_or_a_connection(tmp_path):
    env = FakeEnv()
    rig = live.build_live(play.parse_args(["--no-hub", "--no-record", "--no-store", "--player", "rafae"]), env,
                          player_root=tmp_path)
    assert rig.swing_source == "pose" and env.hub_device.calls == []
    assert rig.hub_status() == "off" and rig.actuator is None
    assert rig.session.hud_state().hub_status == "off"


def test_no_hub_contradicts_an_explicit_hub_swing_sensor():
    with pytest.raises(live.LiveSetupError, match="no hub"):
        live.build_live(play.parse_args(["--no-hub", "--swing-source", "imu"]), FakeEnv())


def test_a_camera_game_without_a_hub_plays_a_rally_and_closes_cleanly(tmp_path):
    from pingpong import fakerig

    rig = fakerig.FakeRig(swing_source="pose", no_hub=True)
    rig.run(until=lambda: rig.game.tracker.streak >= 6, max_s=120)
    assert rig.rig.hub_status() == "off" and rig.game.paused is False
    rig.close()
    assert rig.client.log[-1] == ("loop_stop",)
