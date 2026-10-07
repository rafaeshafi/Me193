import json

import pytest

import config


def test_official_topic_is_exactly_the_assigned_one():
    assert config.SCORE_TOPIC == "ME193/Rogers/RafaeShafi"


def test_extras_live_outside_the_ME193_tree():
    # ME193/# or ME193/Rogers/# wildcard listeners must never see our extras.
    for topic in (config.STATUS_TOPIC, config.DEMO_SCORE_TOPIC, config.SELFTEST_TOPIC):
        assert not topic.startswith("ME193/")
        assert topic.startswith(config.EXTRAS_PREFIX + "/")


def test_broker_defaults_to_the_class_broker():
    assert (config.BROKER_HOST, config.BROKER_PORT) == ("test.mosquitto.org", 1883)


def test_parse_broker_accepts_host_and_optional_port():
    assert config.parse_broker("test.mosquitto.org") == ("test.mosquitto.org", 1883)
    assert config.parse_broker("localhost:18831") == ("localhost", 18831)
    with pytest.raises(ValueError):
        config.parse_broker("localhost:notaport")


def test_record_scope_accepts_only_the_three_documented_values():
    for ok in ("record_session", "record_alltime", "live_streak"):
        assert config.parse_record_scope(ok) == ok
    with pytest.raises(ValueError):
        config.parse_record_scope("best_ever")


def test_merge_overrides_only_touches_known_keys():
    merged = config.merge_overrides({"NOTIFY_MS": 15, "STALE_MS": 300.0}, {"NOTIFY_MS": 20})
    assert merged == {"NOTIFY_MS": 20, "STALE_MS": 300.0}
    with pytest.raises(ValueError, match="NOTIFY_MSS"):
        config.merge_overrides({"NOTIFY_MS": 15}, {"NOTIFY_MSS": 20})


def test_load_local_reads_bench_results_and_ignores_a_missing_file(tmp_path):
    path = tmp_path / "config_local.json"
    assert config.load_local(path) == {}
    path.write_text(json.dumps({"NOTIFY_MS": 20, "GYRO_PER_DPS": 9.7}))
    assert config.load_local(path) == {"NOTIFY_MS": 20, "GYRO_PER_DPS": 9.7}


def test_write_local_merges_and_keeps_earlier_results(tmp_path):
    path = tmp_path / "config_local.json"
    config.write_local({"HUB_RATE_HZ": 52.1}, path)
    config.write_local({"GYRO_PER_DPS": 9.7}, path)
    assert json.loads(path.read_text()) == {"HUB_RATE_HZ": 52.1, "GYRO_PER_DPS": 9.7}


def test_write_local_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError):
        config.write_local({"NOT_A_SETTING": 1}, tmp_path / "config_local.json")


def test_card_serial_override_must_be_a_string_to_keep_leading_zeros(tmp_path):
    path = tmp_path / "config_local.json"
    path.write_text(json.dumps({"CARD_SERIAL": 997}))
    with pytest.raises(ValueError, match="CARD_SERIAL"):
        config.load_local(path)


def test_card_serial_is_a_string_when_configured():
    # Serials have leading zeros ("0997"); an int would silently lose them.
    assert config.CARD_SERIAL is None or isinstance(config.CARD_SERIAL, str)
