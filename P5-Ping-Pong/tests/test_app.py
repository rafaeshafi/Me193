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
    assert any(n.startswith("hit_") for n in rec.names) and "record" not in rec.names    # nothing to beat yet
    session.clock.advance_s(5.0)
    session.tick()                                                  # a miss ends the game
    app.play_until_hits(session, 2)                                 # a second game: 2 passes the best of 1
    assert "record" in rec.names and session.hud_state().message == "NEW RECORD"
    session.clock.advance_s(2.0)
    assert session.hud_state().message == ""


def test_the_hud_state_shows_the_ball_and_the_last_shot():
    session = app.make_session()
    app.play_until_hits(session, 2)
    st = session.hud_state()
    assert st.streak == 2 and st.record == 2 and st.last_kmh and st.last_label
    session.tick()
    assert st.mqtt_status in ("ok", "off", "offline")


def test_the_hud_state_carries_the_shape_of_the_hand_plane_and_the_levels_hit_zone():
    session = app.make_session(level=1)
    st, box = session.hud_state(), session.game.judge.box
    assert st.box_sw == pytest.approx((box.u_max - box.u_min, box.v_max - box.v_min))
    assert st.radius_sw == levels.LEVELS[1].radius_sw and st.reach == levels.LEVELS[1].reach == 0.6
    session.game.set_level(levels.LEVELS[3])
    assert session.hud_state().radius_sw == 0.35 and session.hud_state().reach == 1.0


def test_a_soft_notice_is_only_a_hint_and_gives_way_to_a_real_one():
    session = app.make_session()
    session.set_notice("PADDLE STAYS UPRIGHT: run ./pp calibrate_swing", soft=True)
    assert session.hud_state().message.startswith("PADDLE") and session.has_notice() is False   # a real notice may still come
    session.set_notice("BROKER HOLDS 18")
    assert session.hud_state().message == "BROKER HOLDS 18" and session.has_notice() is True
    session.game.start(session.clock.now_ns())
    assert session.hud_state().message == ""                                     # and nothing shows once the game is on


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


def test_status_sources_can_be_bound_after_the_session_exists():
    session = app.make_session()
    assert (session.hud_state().hub_status, session.hud_state().mqtt_status) == ("ok", "off")
    session.bind_status(hub=lambda: "stale", mqtt=lambda: "offline")
    st = session.hud_state()
    assert (st.hub_status, st.mqtt_status) == ("stale", "offline")


def test_a_notice_is_shown_in_the_lobby_only_and_never_hides_a_pause_message():
    session = app.make_session()
    session.set_notice("UNCALIBRATED: run ./pp calibrate_swing")
    assert session.hud_state().message.startswith("UNCALIBRATED")
    session.on_start()
    assert session.hud_state().message == ""
    session.clock.advance_s(3.0)
    session.tick()
    session.game.set_pause("hub", True, session.clock.now_ns())
    assert session.hud_state().message.startswith("PAUSED")


def _play_a_game(session, hits=3):
    app.play_until_hits(session, hits)
    session.clock.advance_s(6.0)
    session.tick()                                                  # the miss ends the game


def test_a_finished_survival_game_is_summarised_for_the_store():
    summaries = []
    session = app.make_session()
    session.on_game_over = summaries.append
    _play_a_game(session, hits=3)
    [s] = summaries
    assert (s["mode"], s["level"], s["streak"], s["hits"], s["misses"]) == ("survival", "Rookie", 3, 3, 1)
    assert s["max_kmh"] > 0 and 5.0 < s["duration_s"] < 60.0 and s["winner"] is None
    summaries.clear()
    _play_a_game(session, hits=2)                                    # the next game starts from zero
    assert summaries[0]["hits"] == 2 and summaries[0]["streak"] == 2


def test_a_finished_match_is_summarised_with_its_winner_and_points():
    summaries = []
    session = app.make_session(mode="match", target=1)
    session.on_game_over = summaries.append
    app.play_until_hits(session, 1)
    for _ in range(40):                                              # let the CPU miss or the player miss until it ends
        if session.game.phase == "MATCH_OVER":
            break
        session.clock.advance_s(1.0)
        session.tick()
    assert summaries and summaries[0]["mode"] == "match" and summaries[0]["winner"] in ("player", "cpu")
    assert summaries[0]["player_points"] + summaries[0]["cpu_points"] >= 1


def test_the_end_screen_shows_the_leaderboard_the_session_was_given():
    session = app.make_session()
    session.leaderboard_fn = lambda: (("maya", 15), ("rafae", 12))
    _play_a_game(session, hits=1)
    assert session.game.phase == "MATCH_OVER" and session.hud_state().leaderboard == (("maya", 15), ("rafae", 12))
    session.on_start()
    assert session.hud_state().leaderboard == ()                     # only shown on the end screen


TOP = {"flat": 0.05, "top": 0.90, "back": 0.05}
BACK = {"flat": 0.05, "top": 0.05, "back": 0.90}


def _first_hit(session):
    events = []
    original = session.on_swing
    session.on_swing = lambda swing: events.extend(original(swing)) or events
    app.play_until_hits(session, 1)
    return next(e for e in events if e.kind == "hit")


def test_a_spin_model_turns_a_top_swing_into_topspin_and_a_back_swing_into_backspin():
    top = _first_hit(app.make_session(spin_probs_fn=lambda feat: TOP))
    back = _first_hit(app.make_session(spin_probs_fn=lambda feat: BACK))
    assert top.data["topspin"] > 0.3 and back.data["topspin"] < -0.3


def test_without_a_spin_model_every_ball_is_flat_and_the_hud_shows_no_spin_word():
    session = app.make_session()
    hit = _first_hit(session)
    assert hit.data["topspin"] == 0.0 and session.hud_state().spin_text == ""


def test_the_hud_names_the_spin_of_the_last_shot():
    session = app.make_session(spin_probs_fn=lambda feat: TOP)
    _first_hit(session)
    assert "TOPSPIN" in session.hud_state().spin_text


def test_the_features_of_the_swing_reach_the_spin_model():
    seen = []
    session = app.make_session(spin_probs_fn=lambda feat: seen.append(tuple(feat)) or TOP)
    app.play_until_hits(session, 1, feat=tuple(range(12)))
    assert seen[-1] == tuple(range(12))
