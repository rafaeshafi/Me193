"""Easing curves for the menus and the intro: pure functions of a progress 0..1."""

import pytest

from pingpong import anim


@pytest.mark.parametrize("curve", [anim.smoothstep, anim.ease_out_cubic, anim.ease_in_out_cubic, anim.ease_out_back,
                                   anim.pop])
def test_every_curve_starts_at_nothing_and_ends_at_everything(curve):
    assert curve(0.0) == pytest.approx(0.0, abs=1e-9)
    assert curve(1.0) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("curve", [anim.smoothstep, anim.ease_out_cubic, anim.ease_in_out_cubic, anim.ease_out_back, anim.pop])
def test_a_progress_outside_zero_to_one_is_held_at_the_ends(curve):
    assert curve(-3.0) == pytest.approx(curve(0.0))
    assert curve(7.0) == pytest.approx(curve(1.0))


@pytest.mark.parametrize("curve", [anim.smoothstep, anim.ease_out_cubic, anim.ease_in_out_cubic])
def test_the_plain_easings_never_go_backwards(curve):
    values = [curve(k / 100) for k in range(101)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_ease_out_back_overshoots_the_end_and_settles_on_it():
    values = [anim.ease_out_back(k / 100) for k in range(101)]
    assert max(values) > 1.05 and values[-1] == pytest.approx(1.0)


def test_pop_grows_past_full_size_and_comes_back_like_a_button_popping_in():
    values = [anim.pop(k / 100) for k in range(101)]
    peak = max(range(101), key=values.__getitem__)
    assert 1.05 < values[peak] < 1.35 and 40 <= peak <= 85


def test_lerp_runs_from_a_to_b():
    assert anim.lerp(10.0, 20.0, 0.0) == 10.0 and anim.lerp(10.0, 20.0, 1.0) == 20.0 and anim.lerp(10.0, 20.0, 0.25) == 12.5


def test_progress_turns_elapsed_time_into_a_clamped_fraction():
    assert anim.progress(1.0, start=0.5, length=2.0) == pytest.approx(0.25)
    assert anim.progress(0.0, start=0.5, length=2.0) == 0.0 and anim.progress(9.0, start=0.5, length=2.0) == 1.0
    assert anim.progress(1.0, start=0.0, length=0.0) == 1.0                   # a zero-length step is already over


def test_bob_swings_between_minus_and_plus_amplitude_with_the_period():
    assert anim.bob(0.0, period_s=2.0, amplitude=3.0) == pytest.approx(0.0, abs=1e-9)
    assert anim.bob(0.5, period_s=2.0, amplitude=3.0) == pytest.approx(3.0)
    assert anim.bob(1.5, period_s=2.0, amplitude=3.0) == pytest.approx(-3.0)
