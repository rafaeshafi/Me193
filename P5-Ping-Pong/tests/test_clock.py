import pytest

from pingpong.clock import Clock, FakeClock


def test_fake_clock_starts_where_told_and_advances_in_seconds():
    clock = FakeClock(start_ns=5_000_000_000)
    assert clock.now_ns() == 5_000_000_000
    clock.advance_s(0.25)
    assert clock.now_ns() == 5_250_000_000


def test_fake_clock_refuses_to_go_backwards():
    clock = FakeClock()
    with pytest.raises(ValueError):
        clock.advance_s(-0.001)


def test_real_clock_is_monotonic_nanoseconds():
    clock = Clock()
    a = clock.now_ns()
    b = clock.now_ns()
    assert isinstance(a, int) and b >= a
