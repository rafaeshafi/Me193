"""Session: events -> feedback + HUD state, tags, and the scripted fake player."""

import config
import pytest

from pingpong import app, levels
from pingpong.events import PaddlePose, TagEvent
from pingpong.sources_fake import FakeMqttClient

S = 1_000_000_000


class Recorder:
    def __init__(self):
        self.names = []

    def submit(self, name, fire_at_ns=None):
        self.names.append(name)
        return True


def test_a_scripted_fake_rally_reaches_ten_hits_and_publishes_one_to_ten():
    client = FakeMqttClient()
    result = app.run_scripted(10, level=1, client=client)
    payloads = [p["payload"] for p in client.published if p["topic"] == config.SCORE_TOPIC]
    assert payloads == [f"{n}.0" for n in range(1, 11)]
    assert result["streak"] == 10 and result["record"] == 10
    assert result["sim_seconds"] < 60


def test_the_scripted_run_never_touches_the_official_topic_when_the_source_is_fake():
    client = FakeMqttClient()
    app.run_scripted(5, level=1, client=client, source="fake")
    assert [p for p in client.published if p["topic"] == config.SCORE_TOPIC] == []
    demo = [p["payload"] for p in client.published if p["topic"] == config.DEMO_SCORE_TOPIC]
    assert demo == [f"{n}.0" for n in range(1, 6)]


def test_start_and_level_tags_drive_the_game():
    session = app.make_session(level=1)
    session.on_tag(TagEvent(role="LEVEL", value=3, t_ns=0))
    assert session.game.level.name == "Pro"
    session.on_tag(TagEvent(role="START", value=0, t_ns=0))
    assert session.game.phase == "COUNTDOWN"
    session.on_tag(TagEvent(role="LEVEL", value=1, t_ns=0))        # mid-game: ignored
    assert session.game.level.name == "Pro"


def test_the_countdown_counts_3_2_1_on_the_hud():
    session = app.make_session()
    session.on_start()
    digits = []
    for _ in range(3):
        digits.append(session.hud_state().countdown)
        session.clock.advance_s(1.0)
    assert digits == [3, 2, 1]


def test_hits_trigger_haptic_cues_and_a_record_banner_that_expires():
    rec = Recorder()
    session = app.make_session(actuator=rec)
    app.play_until_hits(session, 1)
    assert "record" in rec.names and any(n.startswith("hit_") for n in rec.names)
    assert session.hud_state().message == "NEW RECORD"
    session.clock.advance_s(2.0)
    assert session.hud_state().message == ""


def test_the_hud_state_shows_the_ball_and_the_last_shot():
    session = app.make_session()
    app.play_until_hits(session, 2)
    st = session.hud_state()
    assert st.streak == 2 and st.record == 2 and st.last_kmh and st.last_label
    session.tick()
    assert st.mqtt_status in ("ok", "off", "offline")


def test_the_ball_is_reported_while_it_is_in_flight():
    session = app.make_session()
    session.on_start()
    session.clock.advance_s(3.0)
    session.tick()
    session.clock.advance_s(0.3)
    st = session.hud_state()
    assert st.ball is not None and 0.0 <= st.ball[1] <= 1.0 and st.arrival_ab is not None


def test_the_xray_keeps_the_gates_of_the_last_judged_swing():
    session = app.make_session()
    app.play_until_hits(session, 1)
    names = [g.name for g in session.hud_state().gates]
    assert names == ["J1", "J2", "J3", "J4", "J5", "J6"]


def test_game_over_leaves_the_record_and_the_end_screen_data():
    session = app.make_session()
    app.play_until_hits(session, 3)
    session.clock.advance_s(5.0)                     # no more swings: a miss ends Survival
    session.tick()
    st = session.hud_state()
    assert st.phase == "MATCH_OVER" and st.record == 3


def test_the_hud_says_why_the_game_is_paused_and_clears_when_resumed():
    session = app.make_session()
    session.on_start()
    session.clock.advance_s(3.0)
    session.tick()
    session.game.set_pause("hub", True, session.clock.now_ns())
    msg = session.hud_state().message
    assert msg.startswith("PAUSED") and "hub" in msg
    session.game.set_pause("hub", False, session.clock.now_ns())
    assert not session.hud_state().message.startswith("PAUSED")
