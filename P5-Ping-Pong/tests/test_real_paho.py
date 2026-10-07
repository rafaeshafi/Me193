"""The REAL paho client against a small local broker (tests/minibroker.py): no internet, but real sockets.

FakeMqttClient only stands in for paho's API.  Here the network code we wrote around paho 2.1 runs for real,
including the failure mode seen on the public test broker: it closes a connection attempt without an answer now
and then, sometimes several in a row.
"""

import time

import pytest

import config
from pingpong import mqtt_link, realenv
from pingpong.mqtt_pub import ScorePublisher
from pingpong.realenv import RealEnv
from tests.minibroker import MiniBroker

OFFICIAL = config.SCORE_TOPIC


@pytest.fixture
def broker(monkeypatch):
    made = []

    def make(**kw):
        b = MiniBroker(**kw).start()
        made.append(b)
        monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
        monkeypatch.setattr(config, "BROKER_PORT", b.port)
        return b

    yield make
    for b in made:
        b.stop()


def wait_for(condition, seconds=8.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def live_link(**kw):
    client = mqtt_link.make_paho_client()
    pub = ScorePublisher(client, source="live", **kw)
    mqtt_link.attach(client, pub)
    return client, pub


def stop(client):
    client.loop_stop()
    client.disconnect()


# --- env_check's round trip ---------------------------------------------------------------------------------------------
def test_the_round_trip_measures_a_broker_that_answers(broker):
    broker()
    rtt = RealEnv().mqtt_roundtrip(config.SELFTEST_TOPIC, timeout_s=5.0)
    assert rtt is not None and rtt < 1000.0


def test_the_round_trip_does_not_report_a_failure_because_the_broker_dropped_the_first_attempts(broker, monkeypatch):
    monkeypatch.setattr(realenv, "RETRY_MIN_S", 0.05)         # fast here; the real delays are checked just below
    monkeypatch.setattr(realenv, "RETRY_MAX_S", 0.1)
    b = broker(drop_first=3)                                  # the public broker does this now and then
    rtt = RealEnv().mqtt_roundtrip(config.SELFTEST_TOPIC, timeout_s=6.0)
    assert rtt is not None and b.connections >= 4


def test_the_real_retry_delays_outlast_five_dropped_attempts_within_the_default_timeout():
    delays = [min(realenv.RETRY_MAX_S, realenv.RETRY_MIN_S * 2 ** k) for k in range(5)]
    assert sum(delays) + 1.0 < 10.0                           # RealEnv's default timeout, with a second to answer


def test_a_broker_that_is_not_there_is_reported_as_no_echo_not_as_a_crash(monkeypatch):
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", 9)           # nothing listens on the discard port
    assert RealEnv().mqtt_roundtrip(config.SELFTEST_TOPIC, timeout_s=0.6) is None


def test_the_official_check_publishes_a_retained_zero_sees_it_and_clears_it_again(broker, monkeypatch):
    monkeypatch.setattr(realenv, "RETRY_MIN_S", 0.05)
    b = broker(drop_first=2)                                # and it gets there although two attempts were dropped
    assert RealEnv().official_roundtrip(timeout_s=5.0) is not None
    assert (OFFICIAL, b"0.0", True) in b.log and OFFICIAL not in b.retained
    assert b.log[-1] == (OFFICIAL, b"", True)               # the last thing it did was to clear the topic


# --- the game's own connection --------------------------------------------------------------------------------------------
def test_the_score_goes_out_qos1_retained_and_a_late_joiner_reads_it(broker):
    b = broker()
    client, pub = live_link()
    assert wait_for(lambda: client.is_connected())
    pub.update(1)
    pub.update(2)
    assert wait_for(lambda: b.retained.get(OFFICIAL) == b"2.0")
    assert wait_for(lambda: b.retained.get(config.STATUS_TOPIC) == b"online")
    assert [p for t, p, r in b.log if t == OFFICIAL] == [b"1.0", b"2.0"]
    stop(client)


def test_a_clean_shutdown_says_offline_and_a_dead_connection_makes_the_broker_say_it(broker):
    b = broker()
    client, pub = live_link()
    assert wait_for(lambda: client.is_connected())
    mqtt_link.shutdown(client, pub)
    assert wait_for(lambda: b.retained.get(config.STATUS_TOPIC) == b"offline")
    b.retained.clear()
    crashing, _ = live_link()
    assert wait_for(lambda: crashing.is_connected() and b.retained.get(config.STATUS_TOPIC) == b"online")
    crashing.loop_stop()
    crashing._sock.close()                                  # the process dies without saying goodbye
    assert wait_for(lambda: b.retained.get(config.STATUS_TOPIC) == b"offline")


def test_the_score_survives_the_broker_being_unreachable_when_it_is_published(broker, monkeypatch):
    first = broker()
    port = first.port
    first.stop()                                            # nothing answers on this port for a while
    client, pub = live_link()
    client.reconnect_delay_set(0.1, 0.3)                    # keep the test short; the game's own is 1-5 s
    pub.update(1)
    pub.update(2)
    pub.update(3)
    time.sleep(0.5)
    second = MiniBroker(port=port).start()
    try:
        assert wait_for(lambda: second.retained.get(OFFICIAL) == b"3.0", seconds=10.0)
    finally:
        stop(client)
        second.stop()


def test_the_game_hears_what_the_broker_holds_and_resume_keeps_it_as_the_best_to_beat(broker):
    b = broker()
    b.retained[OFFICIAL] = b"18.0"
    seeded = []
    client, pub = live_link(resume=True)
    pub.on_resume = seeded.append
    assert wait_for(lambda: pub.retained == 18) and seeded == [18]
    pub.update(5)
    time.sleep(0.3)
    assert b.retained[OFFICIAL] == b"18.0"                   # a rehearsal below the best leaves it alone
    pub.update(19)
    assert wait_for(lambda: b.retained[OFFICIAL] == b"19.0")
    stop(client)


def test_a_plain_run_hears_the_value_too_but_publishes_over_it_as_agreed(broker):
    b = broker()
    b.retained[OFFICIAL] = b"18.0"
    client, pub = live_link()
    assert wait_for(lambda: pub.retained == 18)
    pub.update(1)
    assert wait_for(lambda: b.retained[OFFICIAL] == b"1.0")
    stop(client)


def test_a_fake_session_never_even_subscribes_to_the_official_topic(broker):
    b = broker()
    client = mqtt_link.make_paho_client()
    pub = ScorePublisher(client, source="fake")
    mqtt_link.attach(client, pub)
    assert wait_for(lambda: client.is_connected())
    time.sleep(0.2)
    assert pub.retained is None and OFFICIAL not in [t for t, _, _ in b.log]
    stop(client)
