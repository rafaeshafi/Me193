import math

import pytest

from pingpong.oneeuro import OneEuro, OneEuro2D


def run(f, signal, dt=1 / 30):
    return [f(x, i * dt) for i, x in enumerate(signal)]


def test_a_constant_signal_passes_through_unchanged():
    out = run(OneEuro(), [3.0] * 50)
    assert out[0] == 3.0 and all(abs(v - 3.0) < 1e-9 for v in out)


def test_noise_on_a_still_hand_is_strongly_smoothed():
    import random
    rng = random.Random(3)
    noisy = [1.0 + rng.gauss(0, 0.01) for _ in range(300)]       # ~ pose jitter on a still hand (shoulder widths)
    out = run(OneEuro(), noisy)
    raw_spread = max(noisy[60:]) - min(noisy[60:])
    assert max(out[60:]) - min(out[60:]) < 0.5 * raw_spread


def test_a_fast_move_has_much_less_lag_than_a_slow_one_would():
    step = [0.0] * 10 + [1.0] * 40
    fast = run(OneEuro(min_cutoff=1.2, beta=5.0), step)
    slow = run(OneEuro(min_cutoff=1.2, beta=0.0), step)
    settle = lambda s: next(i for i, v in enumerate(s) if i >= 10 and v > 0.9)   # noqa: E731
    assert settle(fast) < settle(slow)


def test_the_filter_never_overshoots_a_step_by_much():
    step = [0.0] * 10 + [1.0] * 60
    out = run(OneEuro(), step)
    assert max(out) <= 1.05


def test_time_going_backwards_is_rejected():
    f = OneEuro()
    f(1.0, 1.0)
    with pytest.raises(ValueError):
        f(2.0, 0.5)


def test_reset_forgets_history():
    f = OneEuro()
    run(f, [5.0] * 20)
    f.reset()
    assert f(1.0, 0.0) == 1.0


def test_two_d_filter_smooths_each_axis_independently():
    f = OneEuro2D()
    out = [f((1.0, math.sin(i / 5)), i / 30) for i in range(60)]
    assert out[-1][0] == pytest.approx(1.0)
    assert out[-1][1] != pytest.approx(math.sin(59 / 5), abs=1e-9)    # smoothed, not raw
