"""Haptic patterns + ActuatorCore: rate/duty limits, priority, blank windows, safety."""

import legoeducation as le
import pytest

from pingpong import haptics
from pingpong.clock import FakeClock
from pingpong.sources_fake import FakeDoubleMotor

MS = 1_000_000
S = 1_000_000_000


class TimedDevice(FakeDoubleMotor):
    """Fake device that also timestamps every WRITE (batch, beep, light) on the fake clock."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.writes = []          # (t_ns, kind)
        self.motor_pulses = []    # (t_ns, ms)

    def motor_run_for_time(self, time_ms, **kw):
        super().motor_run_for_time(time_ms, **kw)
        self.motor_pulses.append((self.clock.now_ns(), time_ms))

    def end_batch(self, blocking=True):
        super().end_batch(blocking=blocking)
        self.writes.append((self.clock.now_ns(), "batch"))

    def beep(self, **kw):
        super().beep(**kw)
        self.writes.append((self.clock.now_ns(), "beep"))

    def light_color(self, color, **kw):
        super().light_color(color, **kw)
        self.writes.append((self.clock.now_ns(), "light"))


def make(**kw):
    clock = FakeClock(start_ns=10 * S)
    dev = TimedDevice(clock)
    dev.connect()
    blanks = []
    core = haptics.ActuatorCore(dev, clock=clock, on_blank=lambda a, b: blanks.append((a, b)), **kw)
    return core, dev, clock, blanks


def drain(core, clock, limit=10_000):
    """Run the scheduler until nothing is left, advancing the fake clock to each wake-up."""
    for _ in range(limit):
        nxt = core.process(clock.now_ns())
        if nxt is None:
            return
        clock.advance_s(max(0.001, (nxt - clock.now_ns()) / 1e9))
    raise AssertionError("scheduler did not go idle")


def names(dev):
    return [c[0] for c in dev.calls]


def test_perfect_hit_is_a_thump_a_high_beep_and_a_white_light_all_non_blocking():
    core, dev, clock, _ = make()
    core.submit("hit_perfect")
    drain(core, clock)
    pulses = [c for c in dev.calls if c[0] == "motor_run_for_time"]
    assert {p[1]["time_ms"] for p in pulses} == {60} and {p[1]["speed"] for p in pulses} == {100}
    beep = [c for c in dev.calls if c[0] == "beep"][0][1]
    assert beep["frequency"] == 1760 and beep["blocking"] is False
    light = [c for c in dev.calls if c[0] == "light_color"][0][1]
    assert light["color"] == le.LEGO_COLOR_WHITE and light["blocking"] is False
    assert all(c[1].get("blocking") is not True for c in dev.calls if c[0] in ("beep", "light_color"))


def test_both_motors_start_together_in_one_batched_write_and_antiphase_by_default():
    core, dev, clock, _ = make()
    core.submit("hit_perfect")
    drain(core, clock)
    seq = names(dev)
    i = seq.index("begin_batch")
    assert seq[i:i + 4] == ["begin_batch", "motor_run_for_time", "motor_run_for_time", "end_batch"]
    left, right = [c[1] for c in dev.calls if c[0] == "motor_run_for_time"][:2]
    assert {left["motor"], right["motor"]} == {le.MOTOR_LEFT, le.MOTOR_RIGHT}
    assert left["direction"] != right["direction"]               # antiphase cancels housing torque


def test_a_good_hit_has_no_motor_cue_only_beep_and_light():
    core, dev, clock, _ = make()
    core.submit("hit_good")
    drain(core, clock)
    assert "motor_run_for_time" not in names(dev)
    assert "beep" in names(dev) and "light_color" in names(dev)


def test_early_is_one_tick_and_late_is_two_ticks_not_left_versus_right():
    for name, ticks in (("hit_early", 1), ("hit_late", 2)):
        core, dev, clock, _ = make()
        core.submit(name)
        drain(core, clock)
        batches = names(dev).count("begin_batch")
        assert batches == ticks, name


def test_never_more_than_ten_writes_in_any_second_even_for_the_longest_pattern():
    core, dev, clock, _ = make()
    core.submit("record")
    drain(core, clock)
    core.submit("fault")
    drain(core, clock)
    times = sorted(t for t, _ in dev.writes)
    for i, t in enumerate(times):
        assert sum(1 for u in times[i:] if u - t < S) <= haptics.MAX_WRITES_PER_S


def test_a_higher_priority_pattern_replaces_a_pending_lower_one_and_equal_ones_are_dropped():
    core, dev, clock, _ = make()
    assert core.submit("hit_good") is True
    assert core.submit("hit_early") is False                    # same priority inside 100 ms: dropped
    assert core.submit("record") is True                        # higher priority wins
    drain(core, clock)
    freqs = [c[1]["frequency"] for c in dev.calls if c[0] == "beep"]
    assert 1320 not in freqs and 700 not in freqs               # the replaced/dropped beeps never sounded


def test_motor_on_time_never_exceeds_25_percent_of_any_two_seconds():
    core, dev, clock, _ = make()
    for _ in range(12):
        core.submit("fault")
        drain(core, clock)
        clock.advance_s(0.5)
    pulses = [(t, ms) for t, ms in dev.motor_pulses]
    # each batch holds two motor commands but both run at once: count one pulse per batch
    per_batch = pulses[::2]
    for t0, _ in per_batch:
        window = sum(ms for t, ms in per_batch if t0 <= t < t0 + 2 * S)
        assert window <= 0.25 * 2000 + 1e-6


def test_over_the_duty_cap_the_motors_stay_quiet_but_the_beep_and_light_still_fire():
    core, dev, clock, _ = make()
    for _ in range(12):
        core.submit("fault")
        drain(core, clock)
        clock.advance_s(0.2)
    assert names(dev).count("beep") == 12 and names(dev).count("light_color") == 12
    assert names(dev).count("begin_batch") < 12


def test_blank_windows_are_built_from_the_actual_write_time():
    core, dev, clock, blanks = make()
    core.submit("hit_perfect")
    drain(core, clock)
    assert len(blanks) == 1
    start, end = blanks[0]
    t_write, _ = dev.writes[[w[1] for w in dev.writes].index("batch")]
    assert start == t_write - 10 * MS
    assert end == t_write + 60 * MS + round(haptics.config.BLANK_AFTER_PULSE_S * S)


def test_no_motor_mutes_the_motors_but_keeps_beep_and_light():
    core, dev, clock, blanks = make(no_motor=True)
    core.submit("hit_perfect")
    drain(core, clock)
    assert "motor_run_for_time" not in names(dev) and blanks == []
    assert "beep" in names(dev) and "light_color" in names(dev)


def test_disarm_stops_the_motors_immediately_and_mutes_everything_after():
    core, dev, clock, _ = make()
    core.disarm()
    assert dev.calls[-1][0] == "motor_stop" and dev.calls[-1][1]["motor"] == le.MOTOR_BOTH
    core.submit("hit_perfect")
    drain(core, clock)
    assert "motor_run_for_time" not in names(dev) and "beep" in names(dev)
    core.arm()
    core.submit("record")
    clock.advance_s(1.0)
    drain(core, clock)
    assert "motor_run_for_time" in names(dev)


def test_a_command_is_not_executed_before_its_fire_time():
    core, dev, clock, _ = make()
    fire = clock.now_ns() + 300 * MS
    core.submit("hit_perfect", fire_at_ns=fire)
    assert core.process(clock.now_ns()) is not None
    assert [c for c in dev.calls if c[0] in ("beep", "begin_batch")] == []
    drain(core, clock)
    assert min(t for t, _ in dev.writes) >= fire


def test_every_pattern_respects_the_pulse_length_and_gap_invariants():
    for name, pat in haptics.PATTERNS.items():
        pulses = [s for s in pat.steps if s.kind == "motors"]
        assert all(s.ms <= 400 for s in pulses), name
        starts = sorted((s.at_ms, s.ms) for s in pulses)
        for (t0, ms0), (t1, _) in zip(starts, starts[1:]):
            assert t1 - (t0 + ms0) >= 80 or t1 - t0 >= 80, name


def test_unknown_pattern_names_are_an_error():
    core, *_ = make()
    with pytest.raises(KeyError):
        core.submit("hit_fantastic")


def test_close_cancels_a_left_open_batch_and_forgets_pending_work():
    core, dev, clock, _ = make()
    core.submit("record")
    dev.fail_on = {"end_batch"}
    try:
        core.process(clock.now_ns())
    except RuntimeError:
        pass
    dev.fail_on = set()
    core.close()
    assert "cancel_batch" in names(dev)
    assert core.process(clock.now_ns()) is None


def test_the_actuator_thread_plays_patterns_and_stops_cleanly():
    import time

    dev = FakeDoubleMotor()
    dev.connect()
    core = haptics.ActuatorCore(dev)                          # real clock
    actuator = haptics.Actuator(core, log=lambda *_: None).start()
    actuator.submit("hit_good")
    deadline = time.monotonic() + 2.0
    while ("beep" not in names(dev) or "light_color" not in names(dev)) and time.monotonic() < deadline:
        time.sleep(0.01)
    actuator.stop()
    assert "beep" in names(dev) and "light_color" in names(dev)
    assert not actuator._thread.is_alive()
    assert actuator.submit("hit_good") is False               # closed cores refuse new work


def test_a_failing_write_does_not_kill_the_actuator_thread():
    import time

    dev = FakeDoubleMotor()
    dev.connect()
    dev.fail_on = {"beep"}
    errors = []
    actuator = haptics.Actuator(haptics.ActuatorCore(dev), log=errors.append).start()
    actuator.submit("hit_good")
    time.sleep(0.2)
    dev.fail_on = set()
    actuator.submit("record")
    deadline = time.monotonic() + 3.0
    while "light_color" not in [c[0] for c in dev.calls][2:] and time.monotonic() < deadline:
        time.sleep(0.01)
    actuator.stop()
    assert errors and any("fake failure" in str(e) for e in errors)
    assert actuator._thread.is_alive() is False


def test_submit_never_waits_for_a_slow_device_write_that_is_already_in_progress():
    # A BLE write blocks its caller (blocking=False included); the game thread submits
    # cues from its 60 Hz loop and must never stall behind the actuator thread's write.
    import threading
    import time

    class SlowDevice(FakeDoubleMotor):
        def __init__(self):
            super().__init__()
            self.in_write = threading.Event()

        def beep(self, **kw):
            self.in_write.set()
            time.sleep(0.3)
            super().beep(**kw)

    dev = SlowDevice()
    dev.connect()
    core = haptics.ActuatorCore(dev)
    core.submit("hit_good")
    worker = threading.Thread(target=lambda: core.process(core.clock.now_ns()))
    worker.start()
    assert dev.in_write.wait(2.0)
    t0 = time.monotonic()
    core.submit("record")
    waited = time.monotonic() - t0
    worker.join()
    assert waited < 0.05, f"submit() waited {waited * 1000:.0f} ms behind a device write"


def test_disarm_does_not_hold_the_scheduler_lock_while_it_writes_to_the_hub():
    import threading
    import time

    class SlowStop(FakeDoubleMotor):
        def motor_stop(self, **kw):
            time.sleep(0.3)
            super().motor_stop(**kw)

    dev = SlowStop()
    dev.connect()
    core = haptics.ActuatorCore(dev)
    threading.Thread(target=core.disarm).start()
    time.sleep(0.05)
    t0 = time.monotonic()
    core.submit("hit_good")
    assert time.monotonic() - t0 < 0.05
