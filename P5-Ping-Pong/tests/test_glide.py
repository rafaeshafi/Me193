"""Glide: the hand as the screen draws it, smoothed between the camera's readings (display only)."""

import math

import numpy as np
import pytest

from pingpong.glide import Glide

MS = 1_000_000
DT = 1_000_000_000 // 60


def run(glide, targets, dt=DT, t0=0):
    return [glide.update(t0 + k * dt, target) for k, target in enumerate(targets)]


def test_with_no_smoothing_the_hand_is_drawn_where_it_is():
    glide = Glide(0.0)
    assert glide.update(0, (1.0, 2.0)) == (1.0, 2.0)
    assert glide.update(DT, (3.0, -1.0)) == (3.0, -1.0)


def test_a_hand_that_stays_put_is_drawn_there_and_a_hand_that_moved_gets_there():
    glide = Glide(0.030)
    out = run(glide, [(0.0, 0.0)] * 5 + [(1.0, 0.5)] * 60)
    assert out[4] == (0.0, 0.0)
    assert out[-1] == pytest.approx((1.0, 0.5), abs=1e-3)
    assert all(a[0] <= b[0] for a, b in zip(out, out[1:]))                  # no overshoot: it only ever approaches


def test_its_step_response_is_two_equal_one_pole_filters_of_half_the_time_constant_each():
    glide = Glide(0.030)                                                    # 15 ms each
    glide.update(0, (0.0, 0.0))
    t = 0
    out = []
    for _ in range(12):                                                     # 1 ms apart: 12 ms
        t += MS
        out.append(glide.update(t, (1.0, 1.0))[0])
    tau = 0.015
    exact = 1.0 - math.exp(-0.012 / tau) * (1.0 + 0.012 / tau)              # two poles in a row
    assert out[-1] == pytest.approx(exact, abs=0.002)
    coarse = Glide(0.030)                                                   # and the same hand when the pictures are 16.7 ms apart
    coarse.update(0, (0.0, 0.0))
    two = [coarse.update(DT * k, (1.0, 1.0))[0] for k in (1, 2)]
    assert two[1] == pytest.approx(1.0 - math.exp(-2 * DT / 1e9 / tau) * (1.0 + 2 * DT / 1e9 / tau), abs=0.002)


def test_sixty_or_a_hundred_and_twenty_pictures_a_second_draw_the_same_hand():
    def path(t_s):                                                          # a hand moving to and fro, as the readings (held) say
        return (math.sin(2 * math.pi * 1.2 * math.floor(t_s * 30) / 30), 0.0)

    shown = {}
    for rate in (60, 120):
        glide, dt = Glide(0.030), 1_000_000_000 // rate
        last = None
        for k in range(rate * 2):
            t = k * dt
            last = glide.update(t, path(t / 1e9))
            if k == rate // 2:                                                  # half a second in
                shown[rate] = last[0]
    assert shown[60] == pytest.approx(shown[120], abs=0.03)


def test_asking_again_for_the_same_moment_gives_the_same_answer():
    glide = Glide(0.030)
    glide.update(0, (0.0, 0.0))
    first = glide.update(DT, (1.0, 1.0))
    assert glide.update(DT, (5.0, 5.0)) == first


def test_a_hand_that_is_lost_is_not_drawn_and_comes_back_where_it_is_not_from_where_it_was():
    glide = Glide(0.030)
    run(glide, [(0.0, 0.0)] * 10)
    assert glide.update(10 * DT, None) is None
    assert glide.update(11 * DT, (2.0, 2.0)) == (2.0, 2.0)


def test_a_long_silence_or_a_jump_across_the_whole_box_does_not_sweep_the_paddle_across_the_screen():
    glide = Glide(0.030)
    run(glide, [(0.0, 0.0)] * 10)
    assert glide.update(10 * DT + 400 * MS, (0.5, 0.5)) == (0.5, 0.5)                 # 0.4 s without a call
    glide2 = Glide(0.030)
    run(glide2, [(0.0, 0.0)] * 10)
    assert glide2.update(10 * DT, (4.0, 0.0)) == (4.0, 0.0)                           # 4 shoulder widths in one picture: another body


def test_it_takes_most_of_the_stairs_out_of_a_hand_that_is_read_at_thirty_a_second_and_drawn_at_sixty():
    readings = lambda k: (2.0 * (k // 2) / 30.0, 0.0)                    # a hand moving at 2 shoulder widths a second, read 30 times a second
    held = [readings(k) for k in range(240)]
    glide = Glide(0.030)
    smooth = run(glide, held)

    def wobble(points):
        v = np.diff([p[0] for p in points]) * 60.0
        return float(np.sqrt(np.mean((v - np.convolve(v, np.ones(9) / 9, mode="same"))[5:-5] ** 2)))

    assert wobble(smooth) < 0.5 * wobble(held)
    lag = [(a[0] - b[0]) / 2.0 for a, b in zip(held[60:], smooth[60:])]     # how far behind the held reading it is drawn, in seconds
    assert 0.0 < np.mean(lag) < 0.06                                        # at 2 SW/s that is under 30 ms of delay
