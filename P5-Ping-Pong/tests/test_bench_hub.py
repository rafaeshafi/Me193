"""bench_hub: measures hub rate/units/clipping and records the swing fixtures."""

import json

import pytest

from pingpong import fixtures
from pingpong.events import ImuSample
from pingpong.sources_fake import FakeEnv
from tools import bench_hub

T0 = 1_000_000_000
FACES = [(980, 0, 0), (-980, 0, 0), (0, 980, 0), (0, -980, 0), (0, 0, 980), (0, 0, -980)]


def stream(hz, secs, fn):
    """ImuSample list from fn(t_s) -> (ax, ay, az, gx, gy, gz)."""
    out = []
    for i in range(int(hz * secs) + 1):
        t = i / hz
        ax, ay, az, gx, gy, gz = fn(t)
        out.append(ImuSample(t_ns=T0 + int(t * 1e9), g=(gx, gy, gz), a=(ax, ay, az)))
    return out


def good_takes(peak=32767, turn_rate=900.0):
    return {
        "rate": [stream(66, 10, lambda t: (0, 0, 1000, 0, 0, 0))],
        "faces": [stream(60, 1.5, lambda t, v=v: (*v, 0, 0, 0)) for v in FACES],
        "turns": [stream(60, 6.0, lambda t: (0, 0, 1000, 0, 0, turn_rate if 1.0 <= t < 5.0 else 0.0))
                  for _ in range(3)],
        "max": [stream(60, 2.5, lambda t: (0, 0, 1000, peak if 1.0 <= t < 1.2 else 0, 0, 0))
                for _ in range(10)],
    }


def test_analyse_measures_rate_accel_gyro_scale_and_clipping():
    m = bench_hub.analyse(good_takes())
    assert m["rate"]["hz"] == pytest.approx(66.0, abs=0.5)
    assert m["accel"]["accel_per_g"] == pytest.approx(980.0, abs=1.0) and m["accel"]["ok"]
    assert m["gyro"]["gyro_per_dps"] == pytest.approx(10.0, rel=0.03) and m["gyro"]["ok"]
    assert m["clip"]["plateau"] is True and m["clip"]["fs_raw"] == 32767


def test_no_clipping_is_reported_when_the_peaks_are_below_full_scale_and_vary():
    takes = good_takes()
    takes["max"] = [stream(60, 2.5, lambda t, k=k: (0, 0, 1000, (12000 + 500 * k) if 1.0 <= t < 1.02 else 0, 0, 0))
                    for k in range(10)]
    m = bench_hub.analyse(takes)
    assert m["clip"]["plateau"] is False and m["clip"]["fs_raw"] is None


def test_inconsistent_turns_are_not_trusted():
    takes = good_takes()
    rates = (600.0, 900.0, 1200.0)
    takes["turns"] = [stream(60, 6.0, lambda t, r=r: (0, 0, 1000, 0, 0, r if 1.0 <= t < 5.0 else 0.0))
                      for r in rates]
    m = bench_hub.analyse(takes)
    assert m["gyro"]["ok"] is False


def test_apply_writes_only_trustworthy_numbers(tmp_path):
    takes = good_takes()
    takes["turns"] = [stream(60, 6.0, lambda t, r=r: (0, 0, 1000, 0, 0, r if 1.0 <= t < 5.0 else 0.0))
                      for r in (600.0, 900.0, 1200.0)]
    path = tmp_path / "config_local.json"
    bench_hub.apply(bench_hub.analyse(takes), path)
    written = json.loads(path.read_text())
    assert "GYRO_PER_DPS" not in written                      # inconsistent turns
    assert written["ACCEL_PER_G"] == pytest.approx(980.0, abs=1.0)
    assert written["HUB_FS_RAW"] == 32767
    assert written["HUB_RATE_HZ"] == pytest.approx(66.0, abs=0.5)


def test_run_steps_prompts_for_every_take_and_saves_fixtures(tmp_path):
    env = FakeEnv(hz=60.0)
    link = env.make_hub(15, {"card_color": 5, "card_serial": "0997"})
    link.connect()
    steps = [bench_hub.Step("soft", 2, 1.0, "soft swing"), bench_hub.Step("waving", 1, 2.0, "wave")]
    seen = []

    def prompt(step, index):
        seen.append((step.label, index))
        env.scenario = lambda now: (0, 0, 1000, 5 * index, 0, 0)

    takes = bench_hub.run_steps(env, link, steps, prompt, out=lambda *_: None, fixtures_dir=tmp_path)
    assert seen == [("soft", 0), ("soft", 1), ("waving", 0)]
    assert [len(t) for t in takes["soft"]] == [pytest.approx(60, abs=2)] * 2
    assert len(takes["waving"][0]) == pytest.approx(120, abs=3)
    on_disk = fixtures.load_all(tmp_path)
    assert sorted(on_disk) == ["soft", "waving"] and len(on_disk["soft"]) == 2


def test_selftest_passes():
    assert bench_hub.main(["--selftest"]) == 0


# --- time to get into position -----------------------------------------------------------------------------------------------
def test_a_new_step_gives_time_to_get_into_position_and_later_takes_only_a_breath():
    env = FakeEnv()
    link = env.make_hub(15, {"card_color": 5, "card_serial": "0997"})
    link.connect()
    said = []
    prompt = bench_hub.make_prompt(env, link, set(), said.append)
    rate, faces = bench_hub.STEPS[0], bench_hub.STEPS[1]
    waits = []
    for step, index in ((rate, 0), (faces, 0), (faces, 1)):
        t0 = env.clock.now_ns()
        prompt(step, index)
        waits.append((env.clock.now_ns() - t0) / 1e9)
    assert waits[0] >= 6.0                                   # the play position is 1.8 m from where the command was typed
    assert 3.0 <= waits[1] < 6.0                             # a different kind of take: pick the hub up, get ready
    assert waits[2] == pytest.approx(1.0, abs=0.01)          # the next take of the same step: just a breath
    assert any("rate" in line for line in said) and any("get into position" in line for line in said)
    assert [c[0] for c in link.dev.calls].count("beep") == 3  # one GO beep per take
