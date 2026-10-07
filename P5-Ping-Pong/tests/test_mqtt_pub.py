"""ScorePublisher: what goes to ME193/Rogers/RafaeShafi and when."""

import re

import pytest

import config
from pingpong.mqtt_pub import ScorePublisher
from pingpong.sources_fake import FakeMqttClient

OFFICIAL = config.SCORE_TOPIC


def make(scope="record_session", **kw):
    client = FakeMqttClient()
    pub = ScorePublisher(client, topic=OFFICIAL, scope=scope, **kw)
    return pub, client


def payloads(client, topic=OFFICIAL):
    return [p["payload"] for p in client.published if p["topic"] == topic]


def test_nothing_is_published_at_startup_not_even_a_zero():
    pub, client = make()
    pub.on_connect()
    assert client.published == [] or all(p["topic"] != OFFICIAL for p in client.published)


def test_first_rally_ticks_up_live_as_floats_qos1_retained():
    pub, client = make()
    for n in (1, 2, 3):
        pub.update(n)
    assert payloads(client) == ["1.0", "2.0", "3.0"]
    for p in client.published:
        if p["topic"] == OFFICIAL:
            assert p["qos"] == 1 and p["retain"] is True
            assert re.fullmatch(r"\d+\.0", p["payload"])


def test_record_scope_publishes_increases_only_and_holds_after_a_miss():
    pub, client = make()
    for n in (1, 2, 3, 3, 2, 3, 4):       # the running value never drops below the record
        pub.update(n)
    assert payloads(client) == ["1.0", "2.0", "3.0", "4.0"]


def test_live_streak_scope_publishes_every_change_including_the_reset():
    pub, client = make(scope="live_streak")
    for n in (1, 2, 0, 1):
        pub.update(n)
    assert payloads(client) == ["1.0", "2.0", "0.0", "1.0"]


def test_reconnect_republishes_the_current_value_only_if_above_zero():
    pub, client = make()
    pub.on_connect()
    assert payloads(client) == []
    pub.update(5)
    client.published.clear()
    pub.on_connect()
    assert payloads(client) == ["5.0"]


def test_session_end_publishes_the_final_value_if_it_was_not_already_sent():
    pub, client = make(scope="live_streak")
    pub.update(4)
    pub.update(0)
    client.published.clear()
    pub.close()
    assert payloads(client) == []          # 0.0 was already the last value sent
    pub2, client2 = make(scope="live_streak")
    pub2.update(7)
    client2.published.clear()
    pub2.close()
    assert payloads(client2) == []


def test_no_publish_flag_keeps_the_broker_untouched():
    pub, client = make(no_publish=True)
    pub.on_connect()
    pub.update(9)
    pub.close()
    assert client.published == []


@pytest.mark.parametrize("source", ["fake", "sim", "replay", "demo"])
def test_non_live_sources_never_reach_the_official_topic(source):
    pub, client = make(source=source)
    pub.update(10)
    assert payloads(client, OFFICIAL) == []
    if source in ("fake", "sim", "demo"):
        assert payloads(client, config.DEMO_SCORE_TOPIC) == ["10.0"]


def test_new_record_publishes_an_explicit_zero():
    pub, client = make()
    pub.update(8)
    client.published.clear()
    pub.publish_new_record_zero()
    assert payloads(client) == ["0.0"]
    pub.update(1)
    assert payloads(client)[-1] == "1.0"


def test_resume_seeds_the_session_record_from_a_retained_value():
    pub, client = make()
    pub.resume_from("30.0")
    pub.update(5)
    assert payloads(client) == []           # 5 < 30: nothing to say
    pub.update(31)
    assert payloads(client) == ["31.0"]


def test_resume_ignores_garbage_retained_values():
    pub, client = make()
    pub.resume_from("not a number")
    pub.update(1)
    assert payloads(client) == ["1.0"]


def test_payload_is_always_formatted_as_a_float_never_an_int():
    pub, client = make()
    pub.update(23)
    assert payloads(client) == ["23.0"]
    with pytest.raises(ValueError):
        pub.update(-1)
    with pytest.raises(ValueError):
        pub.update(float("nan"))


def test_the_status_topic_is_outside_the_score_topic_and_not_a_will_on_it():
    pub, client = make()
    pub.on_connect()
    status = [p for p in client.published if p["topic"] == config.STATUS_TOPIC]
    assert status and status[0]["payload"] == "online" and status[0]["retain"] is True
    assert OFFICIAL not in {c.get("will_topic") for c in client.config_calls}


def test_attach_sets_the_will_on_the_status_topic_only_and_boots_offline():
    from pingpong import mqtt_link

    pub, client = make()
    mqtt_link.attach(client, pub)
    calls = {c["call"]: c for c in client.config_calls}
    assert calls["will_set"]["will_topic"] == config.STATUS_TOPIC
    assert calls["will_set"]["payload"] == "offline" and calls["will_set"]["retain"] is True
    # test.mosquitto.org drops a connection attempt now and then (3 of 4 in one probe): retry briskly, never wait 30 s
    assert (calls["reconnect_delay_set"]["min"], calls["reconnect_delay_set"]["max"]) == (1, 5)
    assert "connect_async" in calls and "loop_start" in calls       # boots without a broker


def test_a_broker_reconnect_republishes_the_best_value():
    from pingpong import mqtt_link

    pub, client = make()
    mqtt_link.attach(client, pub)
    pub.update(6)
    client.published.clear()
    client.simulate_connect()
    assert payloads(client) == ["6.0"]


def test_status_fn_reports_the_real_connection_state_not_a_constant():
    from pingpong import mqtt_link

    client = FakeMqttClient()
    status = mqtt_link.status_fn(client)
    assert status() == "offline"
    client.simulate_connect()
    assert status() == "ok"


def test_shutdown_publishes_offline_then_disconnects_then_stops_the_loop():
    from pingpong import mqtt_link

    client = FakeMqttClient()
    pub = ScorePublisher(client, topic=OFFICIAL, scope="live_streak")
    mqtt_link.attach(client, pub)
    client.simulate_connect()
    pub.update(4)
    client.log.clear()
    mqtt_link.shutdown(client, pub)
    kinds = [e[0] for e in client.log]
    assert kinds[-2:] == ["disconnect", "loop_stop"]
    offline = [e for e in client.log if e[0] == "publish" and e[1] == config.STATUS_TOPIC]
    assert offline and offline[0][2] == "offline"
    assert kinds.index("publish") < kinds.index("disconnect")          # flushed before hanging up
    assert OFFICIAL not in [e[1] for e in client.log if e[0] == "publish"]   # no score reset on exit


def test_shutdown_survives_a_client_that_fails_each_step():
    from pingpong import mqtt_link

    class Broken(FakeMqttClient):
        def publish(self, *a, **k):
            raise RuntimeError("boom")

        def disconnect(self):
            raise RuntimeError("boom")

        def loop_stop(self):
            self.stopped = True

    client = Broken()
    pub = ScorePublisher(client, topic=OFFICIAL)
    mqtt_link.shutdown(client, pub)               # must not raise
    assert client.stopped is True


# --- knowing what the broker already holds (so a rehearsal cannot silently overwrite the graded score) ---------------------
def linked(**kw):
    from pingpong import mqtt_link

    pub, client = make(**kw)
    mqtt_link.attach(client, pub)
    client.simulate_connect()
    return pub, client


def test_a_live_publisher_listens_to_its_own_topic_and_remembers_what_the_broker_holds():
    pub, client = linked()
    assert (OFFICIAL, 1) in client.subscriptions
    assert pub.retained is None
    client.deliver(OFFICIAL, "18.0", retain=True)
    assert pub.retained == 18


def test_the_echo_of_our_own_publish_is_not_mistaken_for_the_retained_value():
    pub, client = linked()
    client.deliver(OFFICIAL, "19.0", retain=False)                       # a live message, not the stored one
    assert pub.retained is None


@pytest.mark.parametrize("kw", [{"source": "fake"}, {"source": "sim"}, {"no_publish": True}])
def test_a_publisher_that_never_touches_the_official_topic_does_not_even_listen_to_it(kw):
    pub, client = linked(**kw)
    assert client.subscriptions == []


@pytest.mark.parametrize("text", ["hello", "", "nan", "-3.0"])
def test_a_garbage_or_cleared_retained_value_is_ignored(text):
    pub, client = linked()
    client.deliver(OFFICIAL, "7.0", retain=True)
    client.deliver(OFFICIAL, text, retain=True)
    assert pub.retained is None


def test_without_resume_a_new_run_starts_from_zero_whatever_the_broker_holds():
    pub, client = linked()
    client.deliver(OFFICIAL, "18.0", retain=True)
    pub.update(1)
    assert payloads(client) == ["1.0"]                                   # "best streak this run": the locked decision


def test_with_resume_the_retained_best_is_the_floor_and_the_game_is_told():
    seeded = []
    pub, client = linked(resume=True)
    pub.on_resume = seeded.append
    client.deliver(OFFICIAL, "18.0", retain=True)
    pub.update(5)
    assert payloads(client) == [] and seeded == [18]
    pub.update(19)
    assert payloads(client) == ["19.0"]


def test_a_resubscribe_after_a_reconnect_hears_the_retained_value_again_without_harm():
    pub, client = linked(resume=True)
    client.deliver(OFFICIAL, "18.0", retain=True)
    client.simulate_connect()
    client.deliver(OFFICIAL, "18.0", retain=True)
    assert client.subscriptions.count((OFFICIAL, 1)) == 2 and pub.retained == 18
