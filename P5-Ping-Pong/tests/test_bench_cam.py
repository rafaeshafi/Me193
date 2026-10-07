"""tools/bench_cam.py on a fake environment with a known answer: the camera runs 100 ms behind the IMU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong import fakerig
from pingpong.sources_fake import FakeEnv, FakeLandmarker, FakeTagDetector
from tools import bench_cam

S = 1_000_000_000
GPD = 10.0


u_of, speed_of = fakerig.wave_u, fakerig.wave_speed


class Bench:
    """An environment, the answers the tool prompts get, and where its files go."""

    def __init__(self, tmp_path, *, tau=0.10, person=True, tag_rate=1.0, imu_follows_hand=True, hz=66.0):
        self.env = FakeEnv(hz=hz)
        self.held, self.prompts, self.asked = {"id": None}, [], []
        self.answers = {"hud": True, "official": False}
        t0 = self.env.clock.now_ns()
        rel = lambda t_ns: (t_ns - t0) / S                                         # noqa: E731
        rng = np.random.default_rng(5)
        self.env.scenario = lambda now: (0, 0, 1000, round((80.0 * speed_of(rel(now)) if imu_follows_hand
                                                            else abs(rng.normal(0, 200))) * GPD), 0, 0)
        self.env.make_landmarker = lambda: FakeLandmarker(
            lambda t_ns: (u_of(rel(t_ns)), 0.1 * u_of(rel(t_ns))) if person else None, self.env.clock, lag_s=tau)
        seen = {"n": 0}

        def ids(t_ns):
            n = seen["n"] = seen["n"] + 1
            thinned_in = int((n + 1) * tag_rate) > int(n * tag_rate)         # shows exactly tag_rate of the looks
            return {self.held["id"]} if self.held["id"] is not None and thinned_in else set()

        self.env.make_tag_detector = lambda: FakeTagDetector(ids, self.env.clock)
        self.config_path, self.report_path = tmp_path / "config_local.json", tmp_path / "bench_cam.json"
        self.out = []

    def prompt(self, text):
        self.prompts.append(text)
        self.held["id"] = int(text.split("card ")[1].split()[0]) if "Hold card" in text else None

    def ask(self, text):
        self.asked.append(text)
        return self.answers["official" if "OFFICIAL" in text else "hud"]

    def run(self, **kw):
        defaults = dict(card={}, secs={"fps": 3.0, "tag": 2.0, "wave": 12.0}, only=("fps", "tags", "wave", "mqtt", "hud"),
                        camera_index=0, official=False, config_path=self.config_path, report_path=self.report_path,
                        prompt=self.prompt, ask=self.ask, out=self.out.append, ready_s=0.0, show=lambda frame: None)
        defaults.update(kw)
        return bench_cam.run(self.env, **defaults)

    def results(self):
        return {r["name"]: r for r in json.loads(self.report_path.read_text())["results"]}


def test_everything_passes_on_a_healthy_rig_and_the_lag_is_written_to_the_config(tmp_path):
    bench = Bench(tmp_path)
    results, code = bench.run()
    assert code == 0, bench.out
    got = bench.results()
    assert got["camera_lag"]["status"] == "PASS" and got["camera_fps"]["status"] == "PASS"
    assert json.loads(bench.config_path.read_text())["CAMERA_LAG_S"] == pytest.approx(0.10, abs=0.015)
    assert all(got[f"tag_{i}"]["status"] == "PASS" for i in range(4))


def test_every_card_is_asked_for_by_number_and_measured_separately(tmp_path):
    bench = Bench(tmp_path)
    bench.run(only=("tags",), ids=(0, 2))
    assert [p for p in bench.prompts if "Hold card" in p] and set(bench.results()) == {"tag_0", "tag_2"}


def test_a_card_the_camera_only_half_sees_fails_with_advice(tmp_path):
    bench = Bench(tmp_path, tag_rate=0.5)
    _, code = bench.run(only=("tags",))
    got = bench.results()
    assert code == 1 and got["tag_1"]["status"] == "FAIL" and "bigger" in got["tag_1"]["detail"].lower()


def test_nobody_in_frame_is_reported_as_the_likely_cause_of_a_slow_or_empty_pose(tmp_path):
    bench = Bench(tmp_path, person=False)
    _, code = bench.run(only=("fps",))
    detail = bench.results()["camera_fps"]["detail"]
    assert code == 1 and "no person" in detail.lower() and "front-light" in detail.lower()


def test_an_untrustworthy_lag_is_reported_but_never_written(tmp_path):
    bench = Bench(tmp_path, imu_follows_hand=False)                  # the gyro has nothing to do with the hand
    _, code = bench.run(only=("wave",))
    assert bench.results()["camera_lag"]["status"] in ("WARN", "FAIL")
    assert not bench.config_path.exists() or "CAMERA_LAG_S" not in json.loads(bench.config_path.read_text())


def test_the_hub_rate_under_camera_and_pose_load_is_judged_like_the_plan_says(tmp_path):
    good_dir = tmp_path / "a"
    good_dir.mkdir()
    good = Bench(good_dir, hz=66.0)
    good.run(only=("wave",))
    assert good.results()["hub_rate_under_load"]["status"] == "PASS"
    slow_dir = tmp_path / "b"
    slow_dir.mkdir()
    slow = Bench(slow_dir, hz=20.0)
    _, code = slow.run(only=("wave",))
    assert slow.results()["hub_rate_under_load"]["status"] == "FAIL" and code == 1


def test_the_scratch_topic_is_always_checked_and_the_official_topic_only_on_request_and_consent(tmp_path):
    bench = Bench(tmp_path)
    bench.run(only=("mqtt",))
    assert bench.env.official_calls == 0 and "mqtt" in bench.results()
    bench2_dir = tmp_path / "x"
    bench2_dir.mkdir()
    bench2 = Bench(bench2_dir)
    bench2.answers["official"] = False
    bench2.run(only=("mqtt",), official=True)
    assert bench2.env.official_calls == 0 and any("OFFICIAL" in q for q in bench2.asked)    # asked, declined
    bench3_dir = tmp_path / "y"
    bench3_dir.mkdir()
    bench3 = Bench(bench3_dir)
    bench3.answers["official"] = True
    bench3.run(only=("mqtt",), official=True)
    assert bench3.env.official_calls == 1 and bench3.results()["mqtt_official"]["status"] == "PASS"


def test_an_unreadable_hud_is_a_fail_with_the_fixes_the_plan_names(tmp_path):
    bench = Bench(tmp_path)
    bench.answers["hud"] = False
    _, code = bench.run(only=("hud",))
    assert code == 1 and "external display" in bench.results()["hud_legibility"]["detail"].lower()


def test_a_camera_that_will_not_open_stops_the_bench_with_the_fix(tmp_path):
    bench = Bench(tmp_path)
    bench.env.camera = "closed"
    _, code = bench.run(only=("fps",))
    assert code == 1 and "camera" in " ".join(bench.out).lower() and "permission" in " ".join(bench.out).lower()


def test_the_camera_index_is_saved_only_when_the_camera_works(tmp_path):
    bench = Bench(tmp_path)
    bench.run(only=("fps",), camera_index=1)
    assert json.loads(bench.config_path.read_text())["CAMERA_INDEX"] == 1


def test_the_camera_worker_is_handed_over_once_the_camera_runs_so_a_window_can_follow_it(tmp_path):
    bench, seen = Bench(tmp_path), []
    bench.run(only=("fps",), on_camera=seen.append)
    assert len(seen) == 1 and callable(seen[0].latest_frame)
    closed, got = Bench(tmp_path), []
    closed.env.camera = "closed"
    closed.run(only=("fps",), on_camera=got.append)
    Bench(tmp_path).run(only=("mqtt",), on_camera=got.append)
    assert got == []                                               # no camera running: nothing to show


def test_no_window_exists_until_the_hud_step_draws_something(monkeypatch):
    # an OpenCV window nobody draws into keeps the Python icon bouncing in the Dock for the whole bench, which
    # looks like Python failing to start while it is only waiting at an Enter prompt
    calls = []
    for name in ("namedWindow", "imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(bench_cam.cv2, name, lambda *a, _n=name, **k: calls.append(_n) or -1)
    monkeypatch.setattr("pingpong.hostcheck.require_host", lambda *a: None)
    monkeypatch.setattr(bench_cam.signal, "signal", lambda *a: None)
    before = []

    def fake_run(env, *, show, **kw):
        before.extend(calls)
        show("frame")
        return [], 0

    monkeypatch.setattr(bench_cam, "run", fake_run)
    assert bench_cam.main(["--only", "hud"]) == 0
    assert before == [] and calls[:2] == ["imshow", "waitKey"]


def test_main_wires_the_live_window_to_the_camera_to_every_wait_and_to_the_hud_still(monkeypatch):
    calls = []
    for name in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(bench_cam.cv2, name, lambda *a, _n=name, **k: calls.append(_n) or -1)
    monkeypatch.setattr("pingpong.hostcheck.require_host", lambda *a: None)
    monkeypatch.setattr(bench_cam.signal, "signal", lambda *a: None)
    vision = SimpleNamespace(latest_frame=lambda: np.zeros((360, 640, 3), np.uint8), last_landmarks=None)
    seen = {}

    def fake_run(env, *, show, on_camera, **kw):
        assert calls == []                                    # nothing on screen before the camera runs
        on_camera(vision)
        env.sleep(0.12)                                       # a wait inside a step: the window must keep drawing
        seen["live"] = calls.count("imshow")
        show("the HUD")                                       # the legibility step's picture
        env.sleep(0.12)
        seen["after"] = calls.count("imshow")
        return [], 0

    monkeypatch.setattr(bench_cam, "run", fake_run)
    assert bench_cam.main(["--only", "fps"]) == 0
    assert seen["live"] >= 2 and seen["after"] == seen["live"] + 1       # live frames, then only the HUD still


def test_the_selftest_is_green():
    assert bench_cam.main(["--selftest"]) == 0
