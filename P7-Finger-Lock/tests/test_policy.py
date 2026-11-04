import pytest

from lock import config
from lock.policy import DENIED, IDLE, LISTENING, LOCKOUT, UNLOCKED, DigitReader, Lock

CFG = config.validate({"code": [2, 5, 3], "hold_frames": 3, "gap_frames": 2, "min_confidence": 0.8,
                       "entry_timeout_s": 10, "unlock_s": 5, "deny_s": 2, "max_attempts": 3, "lockout_s": 30})


def probs(cls, p=0.95):
    v = [(1 - p) / 5] * 6
    v[cls] = p
    return v


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(events=None):
    clock = Clock()
    lock = Lock(CFG, now=clock, on_event=(lambda k, i: events.append((k, i))) if events is not None else None)
    return lock, clock


def show(lock, cls, n, p=0.95):
    out = []
    for _ in range(n):
        out.append(lock.observe(probs(cls, p)))
    return [d for d in out if d is not None]


def test_reader_accepts_after_hold_and_needs_release():
    r = DigitReader(3, 2, 0.8)
    assert [r.observe(probs(2)) for _ in range(3)] == [None, None, 2]
    assert r.observe(probs(2)) is None and r.observe(probs(2)) is None  # held: not counted again
    assert r.observe(probs(0)) is None and r.observe(probs(0)) is None  # released after gap_frames
    assert [r.observe(probs(5)) for _ in range(3)] == [None, None, 5]


def test_reader_low_confidence_resets_run():
    r = DigitReader(3, 2, 0.8)
    r.observe(probs(2)), r.observe(probs(2))
    r.observe(probs(2, 0.5))
    assert r.run == 0 and r.observe(probs(2)) is None


def test_reader_ignores_zero_and_progress():
    r = DigitReader(4, 2, 0.8)
    assert r.observe(probs(0)) is None and r.progress == 0.0
    r.observe(probs(3)), r.observe(probs(3))
    assert r.progress == 0.5


def test_button_starts_listening_and_leds():
    lock, _ = make()
    assert lock.state == IDLE and lock.leds() == (False, False)
    assert lock.press_button()
    assert lock.state == LISTENING and lock.leds() == (True, True)


def test_nothing_read_while_idle():
    lock, _ = make()
    assert show(lock, 2, 10) == []


def test_correct_code_unlocks_then_idles():
    events = []
    lock, clock = make(events)
    lock.press_button()
    for d in (2, 5, 3):
        assert show(lock, d, 3) == [d]
        show(lock, 0, 2)
    assert lock.state == UNLOCKED and lock.leds() == (False, True) and lock.unlocks == 1
    assert ("unlock", {"entered": [2, 5, 3]}) in events
    clock.t = 5.1
    lock.tick()
    assert lock.state == IDLE


def test_wrong_code_denied_then_idle():
    lock, clock = make()
    lock.press_button()
    for d in (2, 5, 4):
        show(lock, d, 3), show(lock, 0, 2)
    assert lock.state == DENIED and lock.leds() == (True, False) and lock.attempts == 1
    clock.t = 2.5
    lock.tick()
    assert lock.state == IDLE


def test_three_failures_lock_out_and_recover():
    events = []
    lock, clock = make(events)
    for i in range(3):
        clock.t += 100
        assert lock.press_button()
        for d in (1, 2, 3):
            show(lock, d, 3), show(lock, 0, 2)
        clock.t += 2.5
        lock.tick()
    assert lock.state == LOCKOUT
    assert any(k == "intruder" for k, _ in events)
    assert not lock.press_button()
    clock.t += 31
    lock.tick()
    assert lock.state == IDLE and lock.attempts == 0 and lock.press_button()


def test_success_resets_attempt_counter():
    lock, clock = make()
    lock.press_button()
    for d in (1, 2, 3):
        show(lock, d, 3), show(lock, 0, 2)
    clock.t += 3
    lock.tick()
    lock.press_button()
    for d in (2, 5, 3):
        show(lock, d, 3), show(lock, 0, 2)
    assert lock.state == UNLOCKED and lock.attempts == 0


def test_entry_timeout_returns_to_idle():
    events = []
    lock, clock = make(events)
    lock.press_button()
    show(lock, 2, 3)
    clock.t = 10.5
    lock.tick()
    assert lock.state == IDLE and ("timeout", {"entered": [2]}) in events


def test_held_hand_is_one_digit():
    lock, _ = make()
    lock.press_button()
    assert show(lock, 2, 30) == [2]
    assert lock.entered == [2]


def test_snapshot_fields():
    lock, _ = make()
    lock.press_button()
    show(lock, 2, 1)
    s = lock.snapshot()
    assert s["state"] == LISTENING and s["entered"] == [] and s["candidate"] == 2 and 0 < s["progress"] < 1
    assert s["remaining"] == 10 and len(s["probs"]) == 6


def test_lockout_blinks_red():
    lock, clock = make()
    lock.state = LOCKOUT
    lock.since = 0
    clock.t = 0.0
    assert lock.leds()[0] is True
    clock.t = 0.3
    assert lock.leds()[0] is False
