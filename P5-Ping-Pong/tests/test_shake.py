"""ShakeMonitor (judge gate J6): an FFT of the last 1.5 s of gyro tells shaking from swinging.

A shake is a sustained, narrow-band oscillation of at least 1.2 Hz: the dominant FFT bin lies in the band,
holds a large share of the power, the motion is big enough to matter, and it has several full cycles.
Measured on the real hub: a person shaking it or waving it like a fan does so at 1.6-2.4 Hz (not the 3-8 Hz
first assumed), while real swings repeat at 0.4-0.8 Hz.  The cycle count matters: a swing plus its backswing also
looks periodic for a moment, and a false lock would cost the player their next hit.
"""

import math
import random

import pytest

from pingpong.events import ImuSample
from pingpong.shake import ShakeMonitor

S = 1_000_000_000
HZ = 66.0
T0 = 5 * S
GPD = 10.0


def sine(f, amp, axis=0, start=0.0, stop=1e9):
    def fn(t):
        v = [0.0, 0.0, 0.0]
        if start <= t <= stop:
            v[axis] = amp * math.sin(2 * math.pi * f * (t - start))
        return v
    return fn


def lobe(t, t0, dur, peak):
    return peak * math.sin(math.pi * (t - t0) / dur) if t0 <= t <= t0 + dur else 0.0


def stream(fn, seconds, hz=HZ, jitter_ms=0.0, gpd=GPD, seed=3):
    rng = random.Random(seed)
    out = []
    for i in range(int(seconds * hz) + 1):
        t = i / hz + (rng.uniform(-jitter_ms, jitter_ms) / 1000.0 if jitter_ms else 0.0)
        g = fn(max(t, 0.0))
        out.append(ImuSample(t_ns=T0 + int(t * 1e9), g=tuple(round(v * gpd) for v in g), a=(0, 0, 1000)))
    return sorted(out, key=lambda s: s.t_ns)


def locks(monitor, samples):
    """[(arrival_ns, lock_until_ns)] for every lock the monitor returns."""
    seen = []
    for s in samples:
        until = monitor.feed(s)
        if until is not None:
            seen.append((s.t_ns, until))
    return seen


def new(**kw):
    return ShakeMonitor(gyro_per_dps=kw.pop("gyro_per_dps", GPD), **kw)


@pytest.mark.parametrize("freq", [1.6, 2.0, 2.4, 3.0, 4.4, 5.0, 6.5, 8.0])
def test_sustained_shaking_from_1_6_to_8_hz_locks_the_paddle_within_about_a_second_and_a_half(freq):
    seen = locks(new(), stream(sine(freq, 300.0), 3.5))
    assert seen, f"{freq} Hz shaking was not caught"
    assert (seen[0][0] - T0) / S <= 1.8                            # caught while it is still going on
    assert seen[0][1] - seen[0][0] == pytest.approx(1.0 * S, rel=0.01)       # locked for one second


@pytest.mark.parametrize("freq", [0.5, 0.8, 1.0, 12.0, 20.0])
def test_swing_rate_repetition_and_fast_buzz_are_not_a_shake(freq):
    assert locks(new(), stream(sine(freq, 400.0), 4.0)) == []


def test_a_small_tremor_does_not_lock():
    assert locks(new(), stream(sine(5.0, 30.0), 4.0)) == []


def test_a_normal_swing_with_its_backswing_never_locks():
    def fn(t):
        return [lobe(t, 1.0, 0.20, -300.0) + lobe(t, 1.25, 0.15, 650.0) + lobe(t, 1.45, 0.20, -250.0), 0.0, 0.0]

    assert locks(new(), stream(fn, 4.0)) == []


def test_a_rally_of_swings_every_half_second_never_locks():
    def fn(t):
        return [sum(lobe(t, 0.3 + 0.5 * k, 0.15, 600.0) for k in range(20)), 0.0, 0.0]

    assert locks(new(), stream(fn, 10.0)) == []


def test_frantic_swinging_at_three_times_a_second_is_a_shake():
    def fn(t):
        return [sum(lobe(t, 0.3 + k / 3.3, 0.15, 600.0) for k in range(20)), 0.0, 0.0]

    assert locks(new(), stream(fn, 4.0))


def test_the_lock_extends_while_shaking_continues_and_the_monitor_goes_quiet_afterwards():
    seen = locks(new(), stream(sine(5.0, 300.0, stop=2.5), 6.0))
    assert len(seen) > 5
    untils = [u for _, u in seen]
    assert untils == sorted(untils)
    assert (seen[-1][0] - T0) / S < 4.0                              # nothing is returned long after it stopped


def test_shaking_on_any_axis_is_caught():
    for axis in (0, 1, 2):
        assert locks(new(), stream(sine(5.0, 300.0, axis=axis), 3.0)), axis


def test_irregular_arrival_times_are_resampled_before_the_fft():
    assert locks(new(), stream(sine(5.0, 300.0), 3.0, jitter_ms=5.0))


def test_the_result_does_not_depend_on_the_hub_unit_scale():
    for gpd in (10.0, 100.0, 0.5):
        assert locks(new(gyro_per_dps=gpd), stream(sine(5.0, 300.0), 3.0, gpd=gpd)), gpd
        assert locks(new(gyro_per_dps=gpd), stream(sine(5.0, 30.0), 3.0, gpd=gpd)) == [], gpd


def test_samples_inside_a_blank_window_are_ignored_so_haptic_vibration_cannot_lock():
    monitor = new()
    monitor.blank(T0, T0 + 5 * S)                                    # the whole stream is "our own pulse"
    assert locks(monitor, stream(sine(5.0, 300.0), 3.0)) == []


def test_a_stalled_stream_neither_crashes_nor_locks():
    monitor = new()
    first = stream(sine(5.0, 300.0), 0.4)
    later = [ImuSample(t_ns=s.t_ns + 3 * S, g=s.g, a=s.a) for s in stream(sine(5.0, 300.0), 0.4)]
    assert locks(monitor, first + later) == []


def test_it_needs_nearly_a_full_window_of_data_before_it_judges():
    assert locks(new(), stream(sine(5.0, 300.0), 0.5)) == []


def test_broadband_noise_is_not_a_shake_even_though_its_loudest_bin_often_lands_in_the_band():
    rng = random.Random(7)
    cache = {}

    def noise(t):
        key = round(t * 1000)
        if key not in cache:
            cache[key] = [rng.gauss(0, 200.0) for _ in range(3)]
        return cache[key]

    assert locks(new(), stream(noise, 20.0)) == []
    assert locks(new(min_peakedness=0.0), stream(noise, 20.0))        # the guard is what stops it
