"""Knowing what the broker holds: the lobby says so, and --resume keeps a retained best as the place to start.

The locked rule is "score = best streak THIS run", so a plain run starts from zero and its first hit publishes 1.0
over whatever the broker held.  That is exactly how a rehearsal after the graded take would wipe the graded score,
so the lobby (and the console) say what the broker holds before anyone swings; --resume is the opt-in that keeps it.
"""

import pytest

import config
import play
from pingpong import live, profile
from pingpong.calibration import SwingCalibration
from pingpong.paddle import ReachBox
from pingpong.sources_fake import FakeEnv
from pingpong.tilt import TiltCalibration

OFFICIAL = config.SCORE_TOPIC


def args_for(*extra):
    return play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-record", "--no-store", *extra])


def calibrated(tmp_path):
    profile.save("rafae", profile.Calibration(swing=SwingCalibration((1.0, 0.0, 0.0), 300.0, 1100.0),
                                              box=ReachBox(-1.5, 1.5, -0.9, 0.7), shoulder_w=0.2,
                                              tilt=TiltCalibration((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))),
                 root=tmp_path)


def build(tmp_path, *extra, messages=None):
    calibrated(tmp_path)
    env = FakeEnv()
    rig = live.build_live(args_for("--player", "rafae", *extra), env, player_root=tmp_path,
                          log=(messages.append if messages is not None else print))
    if env.mqtt_client is not None:
        env.mqtt_client.simulate_connect()
    return rig, env.mqtt_client


def test_the_lobby_and_the_console_say_what_the_broker_holds_and_how_to_keep_it(tmp_path):
    messages = []
    rig, client = build(tmp_path, messages=messages)
    assert rig.session.hud_state().message == ""                          # nothing known yet
    client.deliver(OFFICIAL, "18.0", retain=True)
    rig.pump()
    notice = rig.session.hud_state().message
    assert "18" in notice and "--resume" in notice
    assert any("18" in m and "--resume" in m for m in messages)
    assert rig.session.game.tracker.record == 0                           # a plain run still starts from zero


def test_the_broker_holding_nothing_worth_keeping_is_not_announced(tmp_path):
    messages = []
    rig, client = build(tmp_path, messages=messages)
    client.deliver(OFFICIAL, "0.0", retain=True)
    rig.pump()
    assert rig.session.hud_state().message == "" and not any("broker holds" in m for m in messages)   # ("thresholds" has "holds" in it)


def test_with_resume_the_record_starts_from_the_retained_best_and_the_lobby_says_so(tmp_path):
    rig, client = build(tmp_path, "--resume")
    client.deliver(OFFICIAL, "18.0", retain=True)
    rig.pump()
    assert rig.session.game.tracker.record == 18 and rig.session.hud_state().record == 18
    assert "resum" in rig.session.hud_state().message.lower() and "18" in rig.session.hud_state().message


def test_with_resume_a_short_rally_publishes_nothing_until_the_old_best_is_beaten(tmp_path):
    rig, client = build(tmp_path, "--resume")
    client.deliver(OFFICIAL, "3.0", retain=True)
    game = rig.session.game
    for hit_id in range(1, 4):
        game.tracker.on_valid_hit(hit_id)
        game.publisher.update(game.tracker.value())
    assert [p["payload"] for p in client.published if p["topic"] == OFFICIAL] == []
    game.tracker.on_valid_hit(4)
    game.publisher.update(game.tracker.value())
    assert [p["payload"] for p in client.published if p["topic"] == OFFICIAL] == ["4.0"]


def test_a_guest_or_a_no_publish_session_never_listens_and_so_says_nothing(tmp_path):
    rig, client = build(tmp_path, "--no-publish")
    assert client is None and rig.session.hud_state().message == ""
    rig.pump()
    assert rig.session.hud_state().message == ""


def test_resume_is_a_play_flag():
    assert play.parse_args(["--resume"]).resume is True and play.parse_args([]).resume is False
