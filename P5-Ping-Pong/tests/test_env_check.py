"""env_check: camera + BLE + MQTT + hub IMU rate, driven here by a fake environment."""

import json

import pytest

from pingpong.sources_fake import FakeEnv
from tools import env_check

CARD = {"card_color": "green", "card_serial": "0997"}


def run(env, tmp_path, **kw):
    args = dict(CARD, secs=4.0, skip=(), camera_index=0, config_path=tmp_path / "config_local.json",
                report_path=tmp_path / "env_check.json", prompt=None, out=lambda *_: None)
    args.update(kw)
    return env_check.run_checks(env, **args)


def by_name(results):
    return {r.name: r for r in results}


def test_healthy_run_passes_everything_and_reports_the_hub_rate(tmp_path):
    results, code = run(FakeEnv(hz=66.0), tmp_path)
    r = by_name(results)
    assert code == 0
    assert {r[n].status for n in ("imports", "camera", "hub_connect", "mqtt", "hub_rate")} == {"PASS"}
    assert "66" in r["hub_rate"].detail
    assert "GO" in r["hub_rate"].detail


def test_slow_hub_is_a_no_go_with_the_pose_fallback_advice(tmp_path):
    results, code = run(FakeEnv(hz=20.0), tmp_path)
    r = by_name(results)["hub_rate"]
    assert r.status == "FAIL" and code == 1
    assert "NO-GO" in r.detail and "pose" in r.detail.lower()


def test_marginal_hub_rate_warns_but_does_not_fail(tmp_path):
    results, code = run(FakeEnv(hz=33.0), tmp_path)
    assert by_name(results)["hub_rate"].status == "WARN"
    assert code == 0


def test_hub_not_found_fails_with_a_scan_hint_and_skips_the_rate_test(tmp_path):
    results, code = run(FakeEnv(hub_found=False), tmp_path)
    r = by_name(results)
    assert r["hub_connect"].status == "FAIL" and "scan_hubs" in r["hub_connect"].detail
    assert r["hub_rate"].status == "SKIP"
    assert code == 1


def test_black_camera_frames_point_at_the_camera_permission(tmp_path):
    results, code = run(FakeEnv(camera="black"), tmp_path)
    r = by_name(results)["camera"]
    assert r.status == "FAIL" and "permission" in r.detail.lower()


def test_camera_that_will_not_open_fails(tmp_path):
    results, _ = run(FakeEnv(camera="closed"), tmp_path)
    assert by_name(results)["camera"].status == "FAIL"


def test_mqtt_without_an_echo_fails(tmp_path):
    results, code = run(FakeEnv(mqtt_ok=False), tmp_path)
    assert by_name(results)["mqtt"].status == "FAIL" and code == 1


def test_mqtt_reports_the_round_trip_time(tmp_path):
    results, _ = run(FakeEnv(mqtt_rtt_ms=118.0), tmp_path)
    assert "118" in by_name(results)["mqtt"].detail


def test_missing_card_configuration_is_a_clear_failure_not_a_crash(tmp_path):
    results, code = run(FakeEnv(), tmp_path, card_color=None, card_serial=None)
    r = by_name(results)["hub_connect"]
    assert r.status == "FAIL" and "scan_hubs" in r.detail
    assert code == 1


def test_skip_list_skips_the_named_checks(tmp_path):
    results, code = run(FakeEnv(), tmp_path, skip=("camera", "mqtt"))
    r = by_name(results)
    assert r["camera"].status == "SKIP" and r["mqtt"].status == "SKIP"
    assert code == 0


def test_measured_rate_is_written_to_config_local_and_the_report_saved(tmp_path):
    run(FakeEnv(hz=60.0), tmp_path)
    local = json.loads((tmp_path / "config_local.json").read_text())
    assert local["HUB_RATE_HZ"] == pytest.approx(60.0, abs=1.5)
    assert local["HUB_WORST_GAP_MS"] > 0
    assert local["STALE_MS"] >= 300.0
    # the card that actually connected is remembered, so no tool ever scans for "any" hub
    assert local["CARD_COLOR"] == "green" and local["CARD_SERIAL"] == "0997"
    assert local["NOTIFY_MS"] == 15
    report = json.loads((tmp_path / "env_check.json").read_text())
    assert {r["name"] for r in report["results"]} >= {"camera", "hub_rate", "mqtt"}


def test_hub_is_always_closed_even_when_a_later_check_fails(tmp_path):
    env = FakeEnv(mqtt_ok=False)
    run(env, tmp_path)
    assert [c[0] for c in env.hub_device.calls][-1] == "disconnect"


def test_beep_is_sent_once_non_blocking_on_connect(tmp_path):
    env = FakeEnv()
    run(env, tmp_path)
    beeps = [c for c in env.hub_device.calls if c[0] == "beep"]
    assert len(beeps) == 1 and beeps[0][1]["blocking"] is False


def test_selftest_entry_point_passes():
    assert env_check.main(["--selftest"]) == 0


# --- which camera is which --------------------------------------------------------------------------------------------------
def test_the_camera_check_lists_the_other_cameras_that_open_so_a_phone_at_index_0_can_be_told_apart(tmp_path):
    env = FakeEnv()
    env.cameras = {0: "ok", 1: "ok", 2: "closed"}
    results, _ = run(env, tmp_path)
    detail = by_name(results)["camera"].detail
    assert "index 1" in detail and "index 2" not in detail and "--camera-index" in detail


def test_a_lone_camera_is_not_accompanied_by_a_list_of_nothing(tmp_path):
    env = FakeEnv()
    env.cameras = {0: "ok"}
    results, _ = run(env, tmp_path)
    assert "other camera" not in by_name(results)["camera"].detail


def test_a_camera_index_that_works_and_is_not_the_configured_one_is_remembered(tmp_path):
    import json

    env = FakeEnv()
    env.cameras = {0: "closed", 1: "ok"}
    results, _ = run(env, tmp_path, camera_index=1)
    assert by_name(results)["camera"].status == "PASS"
    assert json.loads((tmp_path / "config_local.json").read_text())["CAMERA_INDEX"] == 1


def test_a_failing_camera_index_is_never_saved(tmp_path):
    import json

    env = FakeEnv()
    env.cameras = {0: "ok", 3: "closed"}
    run(env, tmp_path, camera_index=3)
    saved = tmp_path / "config_local.json"
    assert "CAMERA_INDEX" not in (json.loads(saved.read_text()) if saved.exists() else {})
