"""watch_score (durable evidence of what the broker held) and republish_best (re-send the best value once)."""

import re

import pytest

import config
from pingpong import store
from pingpong.sources_fake import FakeMqttClient
from tools import republish_best, watch_score


def scripted_watch(tmp_path, messages, *, secs=3.0, topic="ME193/Rogers/#"):
    client = FakeMqttClient()
    clock = {"t": 0.0}
    pending = list(messages)                                    # (at_second, topic, payload, retain)

    def sleep(seconds):
        clock["t"] += seconds
        while pending and pending[0][0] <= clock["t"]:
            _, t, payload, retain = pending.pop(0)
            client.deliver(t, payload, retain=retain)

    log = tmp_path / "watch.log"
    out = []
    stats = watch_score.run(client, topic=topic, secs=secs, log_path=log, out=out.append, sleep=sleep,
                            now=lambda: clock["t"], stamp=lambda: "2026-10-07T01:00:00.000")
    return stats, client, log, out


def test_it_subscribes_to_the_topic_once_connected_and_hangs_up_at_the_end(tmp_path):
    stats, client, _, _ = scripted_watch(tmp_path, [])
    client.simulate_connect()
    assert ("ME193/Rogers/#", 1) in client.subscriptions
    assert client.log[-2:] == [("disconnect",), ("loop_stop",)] or ("loop_stop",) in client.log


def test_every_message_is_logged_like_mosquitto_sub_with_its_retain_flag_and_qos(tmp_path):
    msgs = [(0.5, config.SCORE_TOPIC, "1.0", True), (1.0, config.SCORE_TOPIC, "2.0", False),
            (1.5, "ME193/Rogers/Someone", "9.0", False)]
    stats, _, log, _ = scripted_watch(tmp_path, msgs)
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0] == f"2026-10-07T01:00:00.000 {config.SCORE_TOPIC} [1.0] retain=1 qos=1"
    assert lines[2].split()[1] == "ME193/Rogers/Someone" and "retain=0" in lines[2]


def test_the_summary_tracks_the_official_topic_separately_from_classmates(tmp_path):
    msgs = [(0.2, config.SCORE_TOPIC, "3.0", False), (0.4, config.SCORE_TOPIC, "7.0", False),
            (0.6, config.SCORE_TOPIC, "5.0", False), (0.8, "ME193/Rogers/Maya", "40.0", False)]
    stats, _, _, out = scripted_watch(tmp_path, msgs)
    assert stats["official_last"] == 5.0 and stats["official_max"] == 7.0 and stats["official_count"] == 3
    assert stats["topics"]["ME193/Rogers/Maya"] == 1
    text = " ".join(out)
    assert "last 5.0" in text and "max 7.0" in text


def test_a_payload_that_is_not_a_float_on_the_official_topic_is_flagged_not_crashed_on(tmp_path):
    msgs = [(0.2, config.SCORE_TOPIC, "hello", False), (0.4, config.SCORE_TOPIC, "4.0", False)]
    stats, _, log, out = scripted_watch(tmp_path, msgs)
    assert stats["official_last"] == 4.0 and stats["official_bad"] == 1
    assert "NOT A FLOAT" in log.read_text() and any("not a float" in line.lower() for line in out)


def test_the_log_is_appended_to_so_several_runs_build_one_record(tmp_path):
    scripted_watch(tmp_path, [(0.2, config.SCORE_TOPIC, "1.0", False)])
    scripted_watch(tmp_path, [(0.2, config.SCORE_TOPIC, "2.0", False)])
    assert len((tmp_path / "watch.log").read_text().splitlines()) == 2


def test_the_watch_selftest_is_green():
    assert watch_score.main(["--selftest"]) == 0


# --- republish_best ---------------------------------------------------------------------------------------------
def test_without_yes_it_only_says_what_it_would_send():
    client, out = FakeMqttClient(), []
    assert republish_best.run(client, value=21, yes=False, out=out.append) == 0
    assert client.published == [] and any("21.0" in line and "--yes" in line for line in out)


def test_with_yes_it_sends_exactly_one_retained_qos1_float_to_the_official_topic():
    client, out = FakeMqttClient(), []
    assert republish_best.run(client, value=21, yes=True, out=out.append) == 0
    [msg] = client.published
    assert msg == {"topic": config.SCORE_TOPIC, "payload": "21.0", "qos": 1, "retain": True}
    assert re.fullmatch(r"\d+\.0", msg["payload"])


@pytest.mark.parametrize("bad", [-1, float("nan"), "abc"])
def test_a_value_that_is_not_a_non_negative_number_is_refused(bad):
    client = FakeMqttClient()
    assert republish_best.run(client, value=bad, yes=True, out=lambda *_: None) == 2
    assert client.published == []


def test_the_best_value_can_come_from_the_leaderboard_database(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    for streak in (4, 17, 9):
        db.record_game("rafae", dict(mode="survival", level="Club", target=7, streak=streak, record=streak, hits=streak,
                                     misses=1, faults=0, max_kmh=20.0, duration_s=30.0, source="live"))
    assert republish_best.best_from_store(tmp_path / "pp.db", "Rafae") == 17
    assert republish_best.best_from_store(tmp_path / "pp.db", "nobody") is None
    assert republish_best.best_from_store(tmp_path / "missing.db", "rafae") is None


def test_the_cli_needs_a_value_or_a_player_with_games_and_publishes_nothing_by_default(tmp_path, capsys):
    assert republish_best.main(["--db", str(tmp_path / "none.db"), "--player", "rafae"]) == 1
    assert "no best value" in capsys.readouterr().out.lower()


def test_the_republish_selftest_is_green():
    assert republish_best.main(["--selftest"]) == 0
