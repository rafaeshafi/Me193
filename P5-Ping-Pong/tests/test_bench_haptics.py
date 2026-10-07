"""tools/bench_haptics.py: measure the hub's ring-down after a motor pulse and ask what was felt."""

import json
import math

import pytest

from pingpong.haptics import PATTERNS
from pingpong.sources_fake import FakeEnv
from tools import bench_haptics

S = 1_000_000_000


class Rig:
    """A fake hub whose IMU rings for `ring_s` after each motor pulse, and a player who rates what they felt."""

    def __init__(self, tmp_path, *, ring_s=0.10, amp=800.0, rating=4.0, hub_found=True):
        self.env = FakeEnv(hz=66.0, hub_found=hub_found)
        self.pulses = []                                       # (write time ns, ms)
        self.ring_s, self.amp, self.rating = ring_s, amp, rating
        dev = self.env.hub_device
        original = dev.motor_run_for_time

        def motor_run_for_time(time_ms, **kw):
            original(time_ms, **kw)
            self.pulses.append((self.env.clock.now_ns(), time_ms))

        dev.motor_run_for_time = motor_run_for_time
        self.env.scenario = self.imu
        self.config_path, self.report_path = tmp_path / "config_local.json", tmp_path / "bench_haptics.json"
        self.out, self.asked = [], []

    def imu(self, now):
        shake, flip = 0.0, 1 if (now // 15_000_000) % 2 else -1
        for t_write, ms in self.pulses:
            end = t_write + ms * 1_000_000
            if t_write <= now < end:
                shake = self.amp
            elif end <= now < end + self.ring_s * S:
                shake = max(shake, self.amp * math.exp(-(now - end) / S / (self.ring_s / 3)))
        return (0, 0, 1000 + round(flip * shake / 4), round(flip * shake), 0, 0)

    def rate(self, recipe):
        self.asked.append(recipe)
        return self.rating

    def run(self, **kw):
        args = dict(card={}, recipes=("hit_perfect", "hit_early", "fault"), repeats=2, config_path=self.config_path,
                    report_path=self.report_path, prompt=lambda text: None, rate=self.rate, out=self.out.append)
        args.update(kw)
        return bench_haptics.run(self.env, **args)

    def results(self):
        return {r["name"]: r for r in json.loads(self.report_path.read_text())["results"]}


def test_the_ring_down_is_measured_and_written_as_the_blanking_time(tmp_path):
    rig = Rig(tmp_path, ring_s=0.10)
    _, code = rig.run()
    assert code == 0
    blank = json.loads(rig.config_path.read_text())["BLANK_AFTER_PULSE_S"]
    assert 0.08 < blank < 0.22                                  # a ring of ~0.10 s plus the margin


def test_a_longer_ring_gives_a_longer_blanking_time(tmp_path):
    short_dir, long_dir = tmp_path / "s", tmp_path / "l"
    short_dir.mkdir()
    long_dir.mkdir()
    values = []
    for directory, ring in ((short_dir, 0.10), (long_dir, 0.30)):
        rig = Rig(directory, ring_s=ring)
        rig.run()
        values.append(json.loads(rig.config_path.read_text())["BLANK_AFTER_PULSE_S"])
    assert values[1] > values[0] + 0.1


def test_every_recipe_is_pulsed_rated_and_reported(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    assert rig.asked == ["hit_perfect", "hit_early", "fault"]
    got = rig.results()
    assert set(got) >= {"haptic_hit_perfect", "haptic_hit_early", "haptic_fault", "haptic_blanking"}
    assert all(got[f"haptic_{n}"]["status"] == "PASS" for n in ("hit_perfect", "hit_early", "fault"))


def test_when_nothing_is_felt_the_game_still_works_on_beep_and_light_and_says_how_to_fix_it(tmp_path):
    rig = Rig(tmp_path, rating=1.0)
    _, code = rig.run()
    detail = rig.results()["haptic_summary"]["detail"].lower()
    assert code == 0 and rig.results()["haptic_summary"]["status"] == "WARN"
    assert "beep" in detail and ("coin" in detail or "mass" in detail) and "forearm" in detail


def test_a_pulse_the_imu_cannot_feel_never_changes_the_blanking_time(tmp_path):
    rig = Rig(tmp_path, amp=0.0)
    rig.run()
    assert not rig.config_path.exists() or "BLANK_AFTER_PULSE_S" not in json.loads(rig.config_path.read_text())
    assert rig.results()["haptic_blanking"]["status"] == "WARN"


def test_a_missing_hub_is_a_clear_failure(tmp_path):
    rig = Rig(tmp_path, hub_found=False)
    results, code = rig.run()
    assert code == 1 and "not found" in " ".join(rig.out)


def test_the_motors_are_stopped_and_the_hub_released_at_the_end(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    assert [c[0] for c in rig.env.hub_device.calls][-2:] == ["motor_stop", "disconnect"]


def test_only_recipes_with_motor_pulses_make_sense_to_bench():
    with pytest.raises(ValueError, match="motor"):
        bench_haptics.motor_recipes(("hit_good",))
    assert bench_haptics.motor_recipes(("hit_perfect", "record")) == ("hit_perfect", "record")
    assert all(any(s.kind == "motors" for s in PATTERNS[n].steps) for n in bench_haptics.DEFAULT_RECIPES)


def test_the_selftest_is_green():
    assert bench_haptics.main(["--selftest"]) == 0
