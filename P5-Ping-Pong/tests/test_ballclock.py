"""BallClock: the moment of the game at which the picture shows the ball (view, unless the ball is waiting to hear what happens next)."""

import pytest

from pingpong.ballclock import STALL_NS, BallClock

MS = 1_000_000


def test_with_nothing_to_wait_for_the_ball_is_drawn_at_the_pictures_own_time():
    clock = BallClock()
    assert [clock.tick(t * MS) for t in (100, 117, 133, 150)] == [t * MS for t in (100, 117, 133, 150)]


def test_a_pin_holds_the_ball_back_from_that_moment_and_it_waits_there_as_long_as_the_pin_stays():
    clock = BallClock()
    assert clock.tick(100 * MS, pin_ns=130 * MS) == 100 * MS
    assert clock.tick(120 * MS, pin_ns=130 * MS) == 120 * MS
    assert clock.tick(140 * MS, pin_ns=130 * MS) == 130 * MS
    assert clock.tick(200 * MS, pin_ns=130 * MS) == 130 * MS


def test_when_the_pin_goes_the_ball_catches_up_a_little_quicker_than_the_clock_and_never_jumps():
    clock = BallClock(catch_up=0.5)
    clock.tick(100 * MS, pin_ns=100 * MS)
    clock.tick(200 * MS, pin_ns=100 * MS)                           # waited 100 ms
    shown = [clock.tick(t * MS) for t in range(210, 460, 10)]
    steps = [b - a for a, b in zip([100 * MS] + shown, shown)]
    assert all(step <= 15 * MS + 1 for step in steps)               # 10 ms of picture time moves the ball 15 ms along its flight
    assert shown[-1] == 450 * MS                                    # and after about 220 ms (a 110 ms wait, gaining 5 ms a picture) it is back
    assert all(a <= b for a, b in zip(shown, shown[1:]))


def test_a_pin_that_moves_back_does_not_take_the_ball_back_with_it():
    clock = BallClock()
    clock.tick(100 * MS, pin_ns=150 * MS)
    clock.tick(200 * MS, pin_ns=150 * MS)
    assert clock.tick(210 * MS, pin_ns=140 * MS) == 150 * MS


def test_a_new_flight_starts_at_its_beginning_however_far_behind_the_picture_is():
    clock = BallClock()
    clock.tick(100 * MS, pin_ns=100 * MS)
    assert clock.tick(250 * MS, pin_ns=100 * MS, floor_ns=180 * MS) == 180 * MS
    assert clock.tick(260 * MS) == 195 * MS                          # and then gains on the picture
    assert clock.tick(300 * MS, floor_ns=400 * MS) == 300 * MS       # (a flight that has not begun yet is not shown early)


def test_a_stall_or_time_running_backwards_forgets_the_wait():
    clock = BallClock()
    clock.tick(100 * MS, pin_ns=100 * MS)
    clock.tick(200 * MS, pin_ns=100 * MS)
    assert clock.tick(200 * MS + STALL_NS + 1) == 200 * MS + STALL_NS + 1
    clock.tick(500 * MS, pin_ns=450 * MS)
    assert clock.tick(100 * MS) == 100 * MS


def test_reset_starts_afresh():
    clock = BallClock()
    clock.tick(100 * MS, pin_ns=50 * MS)
    clock.reset()
    assert clock.tick(110 * MS) == 110 * MS


def test_asking_twice_at_the_same_time_gives_the_same_answer():
    clock = BallClock()
    clock.tick(100 * MS, pin_ns=100 * MS)
    a = clock.tick(150 * MS, pin_ns=100 * MS)
    assert clock.tick(150 * MS, pin_ns=100 * MS) == a == 100 * MS
