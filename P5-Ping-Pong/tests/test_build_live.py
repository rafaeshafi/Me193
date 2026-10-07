"""build_live: the command line + an environment (real or fake) -> a LiveRig, with failures that
leave nothing open (a leaked hub connection would hide the hub from the next run for ~24 s)."""

import pytest

import play
from pingpong import live, profile
from pingpong.calibration import SwingCalibration
from pingpong.paddle import ReachBox
from pingpong.sources_fake import FakeEnv


def live_args(*extra):
    return play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-record", *extra])


def test_build_live_needs_a_hub_card_and_says_how_to_find_it():
    with pytest.raises(live.LiveSetupError, match="scan_hubs"):
        live.build_live(play.parse_args([]), FakeEnv())


def test_build_live_refuses_a_malformed_card_serial():
    with pytest.raises(live.LiveSetupError, match="serial"):
        live.build_live(play.parse_args(["--card-color", "red", "--card-serial", "12345"]), FakeEnv())


def test_a_hub_that_is_not_found_is_a_clear_error():
    with pytest.raises(live.LiveSetupError, match="not found"):
        live.build_live(live_args(), FakeEnv(hub_found=False))


def test_a_camera_that_will_not_open_is_a_clear_error_and_the_hub_is_let_go_again():
    env = FakeEnv(camera="closed")
    with pytest.raises(live.LiveSetupError, match="camera"):
        live.build_live(live_args(), env)
    names = [c[0] for c in env.hub_device.calls]
    assert names[-2:] == ["motor_stop", "disconnect"]                 # no ghost connection for the next run


def test_a_guest_never_publishes_and_a_named_player_does(tmp_path):
    env = FakeEnv()
    guest = live.build_live(live_args("--player", "guest"), env, player_root=tmp_path)
    assert guest.mqtt_client is None and guest.session.hud_state().mqtt_status == "off"
    env = FakeEnv()
    named = live.build_live(live_args("--player", "rafae"), env, player_root=tmp_path)
    assert named.mqtt_client is env.mqtt_client
    assert any(c["call"] == "connect_async" for c in env.mqtt_client.config_calls)
    assert named.session.hud_state().mqtt_status == "offline"        # until the broker answers


def test_no_publish_silences_even_a_named_player(tmp_path):
    rig = live.build_live(live_args("--player", "rafae", "--no-publish"), FakeEnv(), player_root=tmp_path)
    assert rig.mqtt_client is None


def test_an_unknown_player_gets_the_default_calibration_and_the_lobby_says_so(tmp_path):
    rig = live.build_live(live_args("--player", "newbie"), FakeEnv(), player_root=tmp_path)
    assert rig.calibration.calibrated is False
    assert "calibrat" in rig.session.hud_state().message.lower()


def test_a_saved_calibration_is_loaded_for_the_player(tmp_path):
    saved = profile.Calibration(swing=SwingCalibration((0.0, 1.0, 0.0), 280.0, 1100.0),
                                box=ReachBox(-1.5, 1.5, -0.9, 0.7), shoulder_w=0.2, hand="left")
    profile.save("rafae", saved, root=tmp_path)
    rig = live.build_live(live_args("--player", "rafae"), FakeEnv(), player_root=tmp_path)
    assert rig.calibration.calibrated and rig.calibration.hand == "left"
    assert rig.session.game.judge.box == saved.box and rig.session.game.omega_hi == 1100.0
    assert rig.session.hud_state().message == ""


def test_the_hub_is_connected_with_the_configured_notify_interval_and_card(tmp_path):
    env = FakeEnv()
    live.build_live(live_args("--player", "rafae"), env, player_root=tmp_path)
    connect = [c for c in env.hub_device.calls if c[0] == "connect"][0][1]
    assert connect["device_notification_delay"] == live.config.NOTIFY_MS
    assert connect["card_serial"] == "1131"


def test_a_live_session_is_recorded_under_the_player_unless_told_not_to(tmp_path):
    args = play.parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "rafae"])
    rig = live.build_live(args, FakeEnv(), player_root=tmp_path / "players", record_root=tmp_path / "rec")
    rig.close()
    [session] = list((tmp_path / "rec").iterdir())
    assert session.name.endswith("-rafae") and (session / "session.json").exists()
    off = live.build_live(live_args("--player", "rafae"), FakeEnv(), player_root=tmp_path / "players",
                          record_root=tmp_path / "rec2")
    assert not (tmp_path / "rec2").exists() and off.recorder is None


def test_a_recording_that_cannot_be_started_is_reported_and_the_game_still_runs(tmp_path):
    blocker = tmp_path / "rec"
    blocker.write_text("not a directory")
    messages = []
    args = play.parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "rafae"])
    rig = live.build_live(args, FakeEnv(), player_root=tmp_path / "players", record_root=blocker, log=messages.append)
    assert rig.recorder is None and any("recording" in m.lower() for m in messages)
