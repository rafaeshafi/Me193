"""Tilt: the paddle on screen turns when the hub turns in the hand (side to side, like a doorknob).

Calibration finds the axis the player turns the hub about, which way is 'right' (the first move) and what 'upright'
is (the hub held still); the estimator then follows the hub's rotation about that axis with the gyro (fast) and
gravity (no drift).  Positive angles are clockwise as the player sees them.
"""

import math

import numpy as np
import pytest

from pingpong import tilt
from pingpong.events import ImuSample

GPD, APG, HZ = 0.988, 1017.0, 64.0
R_TRUE = np.array([0.8, 0.6, 0.0])                    # the axis the hub turns about, in hub coordinates
UP = np.array([0.0, 0.0, 1.0])                        # 'upright': what the accelerometer reads with the paddle upright


def rotate(v, axis, deg):
    """Rodrigues: v turned by deg about the unit axis (right-hand rule)."""
    a, k = math.radians(deg), np.asarray(axis, float) / np.linalg.norm(axis)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * (k @ v) * (1 - math.cos(a))


def stream(angles_deg, *, axis=R_TRUE, up=UP, hz=HZ, bias_dps=(0.0, 0.0, 0.0), t0_ns=5_000_000_000, noise=0.0, seed=1):
    """The raw samples of a hub that is turned about `axis` through the given angles (one per sample).

    The gyro reads the rate about the axis; gravity, seen from the hub, turns the OPPOSITE way."""
    rng = np.random.default_rng(seed)
    unit = np.asarray(axis, float) / np.linalg.norm(axis)
    out, prev = [], angles_deg[0]
    for i, deg in enumerate(angles_deg):
        rate = (deg - prev) * hz if i else 0.0
        prev = deg
        g = rate * unit + np.asarray(bias_dps) + rng.normal(0, noise, 3)
        a = rotate(up, unit, -deg) + rng.normal(0, noise / 1000.0, 3)
        out.append(ImuSample(t_ns=t0_ns + round(i * 1e9 / hz), g=tuple(np.round(g * GPD).astype(int)),
                             a=tuple(np.round(a * APG).astype(int))))
    return out


def bunched(samples, size=4):
    """The same samples, but arriving the way a real hub delivers them: groups sharing one arrival time."""
    out = []
    for i, s in enumerate(samples):
        last = min(len(samples) - 1, (i // size + 1) * size - 1)
        out.append(ImuSample(t_ns=samples[last].t_ns, g=s.g, a=s.a))
    return out


def sine(amp, cycles, seconds=4.0, first=1.0, hz=HZ):
    n = int(seconds * hz)
    return [first * amp * math.sin(2 * math.pi * cycles * i / n) for i in range(n)]


def fit(samples, **kw):
    return tilt.fit(samples, neutral=tuple(UP), gyro_per_dps=GPD, accel_per_g=APG, bias_dps=(0.0, 0.0, 0.0), **kw)


def angle_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


# --- calibration ------------------------------------------------------------------------------------------------------------
def test_the_axis_is_the_one_the_hub_was_turned_about():
    cal = fit(stream(sine(40.0, 2), noise=1.5))
    assert angle_between(cal.axis, R_TRUE) < 3.0
    assert cal.neutral == pytest.approx(tuple(UP))


def test_the_first_move_is_right_so_a_wiggle_that_starts_to_the_left_gives_the_opposite_axis():
    # 'right' is whatever the player does first: that turn becomes a positive angle either way
    cal = fit(stream(sine(40.0, 2, first=-1.0), noise=1.5))
    assert angle_between(cal.axis, -R_TRUE) < 3.0


def test_the_neutral_and_the_gyros_resting_offset_are_kept_with_the_axis():
    cal = tilt.fit(stream(sine(40.0, 2)), neutral=(0.0, 0.1, 5.0), gyro_per_dps=GPD, accel_per_g=APG,
                   bias_dps=(1.0, -2.0, 0.5))
    assert np.linalg.norm(cal.neutral) == pytest.approx(1.0) and cal.bias_dps == (1.0, -2.0, 0.5)


def test_too_little_turning_is_refused_and_says_to_turn_further():
    with pytest.raises(tilt.TiltError, match="further"):
        fit(stream(sine(6.0, 2), noise=1.0))


def test_a_turn_that_is_big_enough_to_start_but_too_small_says_to_turn_further_not_both_ways():
    with pytest.raises(tilt.TiltError, match="further"):
        fit(stream(sine(14.0, 2), noise=1.0))                      # +-14 degrees: both ways, but not far


def test_turning_one_way_only_is_refused_and_says_to_turn_both_ways():
    with pytest.raises(tilt.TiltError, match="both ways"):
        fit(stream([40.0 * math.sin(math.pi * i / 256) for i in range(256)]))        # 0 -> 40 -> 0, never to the left


def test_turning_about_no_particular_axis_is_refused():
    rng = np.random.default_rng(3)
    mess = [ImuSample(t_ns=5_000_000_000 + i * 15_000_000, g=tuple(rng.normal(0, 120, 3).round().astype(int)),
                      a=(0, 0, 1017)) for i in range(256)]
    with pytest.raises(tilt.TiltError, match="one axis"):
        fit(mess)


def test_a_calibration_survives_json():
    cal = fit(stream(sine(40.0, 2)))
    assert tilt.TiltCalibration.from_json(cal.to_json()) == cal


# --- following the hub ---------------------------------------------------------------------------------------------------------
def estimator(**kw):
    return tilt.TiltEstimator(tilt.TiltCalibration(tuple(R_TRUE / np.linalg.norm(R_TRUE)), tuple(UP), (0.0, 0.0, 0.0)),
                              GPD, APG, **kw)


def run(est, samples):
    return [est.feed(s) for s in samples]


def test_the_angle_follows_a_turn_about_the_axis_and_holds_it():
    ramp = [min(30.0, i * 0.8) for i in range(160)]                   # to 30 degrees, then held
    angles = run(estimator(), stream(ramp))
    assert angles[-1] == pytest.approx(30.0, abs=2.0)
    assert angles[20] == pytest.approx(ramp[20], abs=3.0)             # and it is following on the way, not lagging far


def test_samples_that_arrive_in_bursts_are_still_integrated_at_the_hubs_steady_rate():
    # the hub samples at a steady 64 Hz but BLE delivers them in bursts (gaps of 2 ms, then 60 ms): integrating over the
    # arrival gaps gave jumps of tens of degrees per sample on a real recording
    ramp = [min(30.0, i * 0.8) for i in range(160)]
    angles = run(estimator(), bunched(stream(ramp), size=6))
    assert angles[-1] == pytest.approx(30.0, abs=2.0)
    mid = angles[: len(ramp) // 2]
    assert max(abs(b - a) for a, b in zip(mid, mid[1:])) < 3.0                  # smooth on the way up too


def test_after_a_real_gap_with_lost_samples_the_time_that_passed_still_counts():
    est = estimator()
    first = stream([0.0] * 5)
    run(est, first)
    turning = stream([10.0 * i for i in range(1, 6)], t0_ns=first[-1].t_ns + 400_000_000)     # 0.4 s of nothing, then a fast turn
    angles = run(est, turning)
    assert angles[0] > 0.0 and np.isfinite(angles).all()                          # (the turn is not swallowed or blown up)


def test_turning_the_other_way_is_negative():
    angles = run(estimator(), stream([max(-25.0, -i * 0.8) for i in range(160)]))
    assert angles[-1] == pytest.approx(-25.0, abs=2.0)


def test_gravity_pulls_a_drifting_gyro_back_where_the_hub_really_is():
    # a gyro that reads 3 dps too much drifts 3 degrees a second; held still for 20 s the paddle must stay upright
    still = stream([0.0] * int(20 * HZ), bias_dps=tuple(3.0 * R_TRUE / np.linalg.norm(R_TRUE)))
    angles = run(estimator(), still)
    assert abs(angles[-1]) < 3.0


def test_a_turn_about_another_axis_does_not_move_the_paddle():
    pitch = np.array([-0.6, 0.8, 0.0])                                # perpendicular to both the axis and 'upright'
    angles = run(estimator(), stream(sine(40.0, 2), axis=pitch))
    assert max(abs(a) for a in angles) < 4.0


def test_while_the_hub_is_being_swung_only_the_gyro_is_trusted():
    still = stream([0.0] * 40)
    swung = [ImuSample(t_ns=s.t_ns, g=s.g, a=(2600, 0, 1500)) for s in still]       # 3 g, nowhere near upright
    assert abs(run(estimator(), swung)[-1]) < 1.0


def test_when_the_axis_points_at_the_ground_gravity_cannot_see_the_turn_and_the_angle_relaxes_to_zero():
    est = tilt.TiltEstimator(tilt.TiltCalibration((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)), GPD, APG)
    turn = stream([min(60.0, i * 2.0) for i in range(60)], axis=(0.0, 0.0, 1.0))
    held = stream([60.0] * int(40 * HZ), axis=(0.0, 0.0, 1.0), t0_ns=turn[-1].t_ns + 15_000_000)
    angles = run(est, turn + held)
    assert angles[59] > 40.0 and abs(angles[-1]) < 20.0


def test_the_angle_never_goes_past_eighty_degrees():
    angles = run(estimator(), stream([min(170.0, i * 2.0) for i in range(100)]))
    assert max(angles) <= tilt.MAX_DEG + 1e-9
