"""SwingDetector: signed forward-axis FSM, tested on synthetic IMU streams.

A magnitude-only detector fired on the BACKSWING 100% of the time in a reviewer's
simulation, so the first-class cases here are backswing-only (0 IMPACT) and
backswing-then-forward (IMPACT on the forward peak).
"""

import math
import random

import pytest

from pingpong.events import ImuSample
from pingpong.swing import SwingDetector, SwingParams

HZ = 66.0
T0 = 5_000_000_000
GPD = 10.0   # raw counts per deg/s in the default params


def pulse(t, start, dur, peak):
    """Half-sine lobe: `peak` dps at start + dur/2."""
    if start <= t <= start + dur:
        return peak * math.sin(math.pi * (t - start) / dur)
    return 0.0


def stream(duration_s, fn, hz=HZ, noise=0.0, seed=1, gpd=GPD, bias=(0.0, 0.0, 0.0), clip=None):
    rng = random.Random(seed)
    out = []
    for i in range(int(duration_s * hz) + 1):
        t = i / hz
        s = fn(t)
        g = [s * gpd + bias[0] * gpd + rng.gauss(0, noise * gpd),
             bias[1] * gpd + rng.gauss(0, noise * gpd),
             bias[2] * gpd + rng.gauss(0, noise * gpd)]
        if clip is not None:
            g = [max(-clip, min(clip, v)) for v in g]
        out.append(ImuSample(t_ns=T0 + int(t * 1e9), g=tuple(round(v) for v in g), a=(0, 0, 1000)))
    return out


def run(det, samples):
    """[(arrival_ns, event)] for every event the detector emits."""
    seen = []
    for s in samples:
        for e in det.feed(s):
            seen.append((s.t_ns, e))
    return seen


def impacts(seen):
    return [(t, e) for t, e in seen if e.kind == "IMPACT"]


def new(**kw):
    params = dict(u_fwd=(1.0, 0.0, 0.0), gyro_per_dps=GPD, t_pk=250.0, fs_raw=32767)
    params.update(kw)
    return SwingDetector(SwingParams(**params))


def test_clean_forward_swing_gives_one_impact_at_the_true_peak():
    det = new()
    seen = run(det, stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0)))
    hits = impacts(seen)
    assert len(hits) == 1
    arrival, ev = hits[0]
    true_peak_ns = T0 + int((0.6 + 0.075) * 1e9)
    assert abs(ev.t_ns - true_peak_ns) <= 8e6          # |t_peak - true peak| <= 8 ms
    assert ev.w_pk == pytest.approx(800.0, rel=0.10)
    assert 100 <= ev.dur_ms <= 250
    assert ev.n_reversals == 0 and ev.clipped is False
    assert len(ev.feat) == 12


def test_impact_is_reported_shortly_after_the_peak_never_before_it():
    det = new()
    arrival, ev = impacts(run(det, stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0))))[0]
    delay_ms = (arrival - ev.t_ns) / 1e6
    assert 0 < delay_ms <= 70                          # detected on the falling edge


def test_backswing_only_gives_no_impact():
    det = new()
    seen = run(det, stream(3.0, lambda t: pulse(t, 0.5, 0.20, -700.0)))
    assert impacts(seen) == []


def test_forward_swing_after_a_larger_backswing_impacts_on_the_forward_peak():
    det = new()
    fn = lambda t: pulse(t, 0.5, 0.20, -600.0) + pulse(t, 0.78, 0.15, 500.0)   # noqa: E731
    hits = impacts(run(det, stream(2.5, fn)))
    assert len(hits) == 1
    ev = hits[0][1]
    assert abs(ev.t_ns - (T0 + int((0.78 + 0.075) * 1e9))) <= 10e6
    assert ev.w_pk == pytest.approx(500.0, rel=0.12)
    assert ev.n_reversals == 1                         # the backswing lobe


def test_soft_swing_above_threshold_counts_and_a_weaker_one_does_not():
    assert len(impacts(run(new(), stream(2.0, lambda t: pulse(t, 0.6, 0.18, 320.0))))) == 1
    assert impacts(run(new(), stream(2.0, lambda t: pulse(t, 0.6, 0.18, 200.0)))) == []


def test_waving_at_3hz_stops_producing_impacts_after_the_first_lobes():
    det = new()
    samples = stream(10.0, lambda t: 600.0 * math.sin(2 * math.pi * 3.0 * t))
    seen = impacts(run(det, samples))
    late = [e for t, e in seen if (t - T0) / 1e9 > 0.9]
    assert late == []                                   # oscillation guard
    assert len(seen) <= 2


def test_rest_noise_for_30_seconds_gives_no_events():
    det = new()
    assert run(det, stream(30.0, lambda t: 0.0, noise=6.0)) == []


def test_single_sample_spike_is_not_a_swing():
    def fn(t):
        return 1500.0 if abs(t - 1.0) < 0.5 / HZ else 0.0

    assert impacts(run(new(), stream(2.0, fn))) == []


def test_samples_inside_a_blank_window_are_ignored():
    det = new()
    det.blank(T0 + int(0.4e9), T0 + int(1.2e9))
    assert run(det, stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0))) == []


def test_two_swings_inside_the_refractory_window_give_one_impact():
    det = new()
    fn = lambda t: pulse(t, 0.6, 0.12, 700.0) + pulse(t, 0.78, 0.12, 700.0)    # noqa: E731
    assert len(impacts(run(det, stream(2.0, fn)))) == 1


def test_thresholds_are_in_dps_so_a_10x_unit_change_changes_nothing():
    fn = lambda t: pulse(t, 0.5, 0.2, -400.0) + pulse(t, 0.78, 0.15, 800.0)    # noqa: E731
    a = impacts(run(new(gyro_per_dps=10.0), stream(2.5, fn, gpd=10.0)))
    b = impacts(run(new(gyro_per_dps=100.0), stream(2.5, fn, gpd=100.0)))
    assert len(a) == len(b) == 1
    assert abs(a[0][1].t_ns - b[0][1].t_ns) <= 100_000        # 0.1 ms: only count-rounding differs
    assert a[0][1].w_pk == pytest.approx(b[0][1].w_pk, rel=1e-3)


def test_a_constant_gyro_bias_is_learned_and_removed_at_rest():
    det = new()
    samples = stream(3.0, lambda t: pulse(t, 2.0, 0.15, 800.0), bias=(30.0, -20.0, 10.0))
    hits = impacts(run(det, samples))
    assert len(hits) == 1
    assert hits[0][1].w_pk == pytest.approx(800.0, rel=0.10)


def test_saturated_samples_flag_clipping():
    det = new(fs_raw=3000)
    samples = stream(2.0, lambda t: pulse(t, 0.6, 0.20, 1500.0), clip=3000)   # 1500 dps*10 > 3000 raw
    hits = impacts(run(det, samples))
    assert len(hits) == 1 and hits[0][1].clipped is True


def test_swing_start_is_announced_before_the_impact():
    det = new()
    seen = run(det, stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0)))
    kinds = [e.kind for _, e in seen]
    assert kinds[0] == "SWING_START" and kinds.count("IMPACT") == 1
    assert seen[0][0] < impacts(seen)[0][0]


def test_off_axis_rotation_is_not_a_forward_swing():
    # all the rotation is about y; the learned forward axis is x
    det = new()
    samples = []
    for i in range(int(2.0 * HZ) + 1):
        t = i / HZ
        samples.append(ImuSample(t_ns=T0 + int(t * 1e9), g=(0, round(pulse(t, 0.6, 0.15, 800.0) * GPD), 0),
                                 a=(0, 0, 1000)))
    assert impacts(run(det, samples)) == []


def test_forward_axis_is_normalised():
    det = new(u_fwd=(2.0, 0.0, 0.0))
    assert len(impacts(run(det, stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0))))) == 1


def test_the_recent_signed_rate_is_available_for_the_hud_trace():
    det = new()
    run(det, stream(2.0, lambda t: pulse(t, 0.6, 0.15, 600.0)))
    trace = det.trace(2.0)
    times = [t for t, _ in trace]
    assert times == sorted(times) and 120 <= len(trace) <= 135                 # ~2 s at 66 Hz
    assert max(v for _, v in trace) == pytest.approx(600.0, rel=0.05)           # signed rate in dps
    assert len(det.trace(0.5)) < len(trace) and new().trace(1.0) == []


def test_the_interpolated_peak_time_stays_between_the_peak_sample_and_its_neighbours_even_after_a_gap():
    # A Bluetooth hiccup right before the peak sample makes the parabola's vertex land after the NEXT sample;
    # a peak can never be later than the first sample that already shows it falling.
    stamps_values = [(0.600, 0.0), (0.615, 300.0), (0.735, 700.0), (0.750, 650.0), (0.765, 100.0), (0.780, 0.0),
                     (0.795, 0.0), (0.810, 0.0)]
    warm = [(i * 0.015, 0.0) for i in range(41)]                      # 0.6 s of rest for the warm-up
    samples = [ImuSample(t_ns=T0 + int(t * 1e9), g=(round(v * GPD), 0, 0), a=(0, 0, 1000))
               for t, v in warm[:-1] + stamps_values]
    seen = impacts(run(new(), samples))
    assert len(seen) == 1
    t_peak = seen[0][1].t_ns
    assert T0 + int(0.735 * 1e9) <= t_peak <= T0 + int(0.750 * 1e9)


def test_an_event_is_never_dated_after_the_sample_that_produced_it_on_an_irregular_stream():
    rng = random.Random(11)
    det, t = new(), T0
    for i in range(6000):
        t += rng.choice([15, 15, 16, 30, 100, 250]) * 1_000_000
        phase = (i // 20) % 4
        v = [0.0, 700.0, 300.0, -400.0][phase] * rng.uniform(0.5, 1.2)
        sample = ImuSample(t_ns=t, g=(round(v * GPD), 0, 0), a=(0, 0, 1000))
        for event in det.feed(sample):
            assert event.t_ns <= t, (event.kind, event.t_ns - t)


def test_fuzz_messy_streams_never_throw_and_every_event_is_finite_and_not_dated_in_the_future():
    from pingpong.shake import ShakeMonitor

    rng = random.Random(7)
    for _ in range(4):
        params = SwingParams(u_fwd=tuple(rng.gauss(0, 1) for _ in range(3)), gyro_per_dps=rng.choice([1.0, 10.0, 16.4, 100.0]),
                             t_pk=rng.uniform(80, 400))
        det, shake = SwingDetector(params), ShakeMonitor(gyro_per_dps=params.gyro_per_dps)
        t, g = T0, [0.0, 0.0, 0.0]
        for _ in range(6000):
            t += rng.choice([15, 15, 15, 16, 30, 100, 400]) * 1_000_000          # irregular arrival, BLE gaps
            roll = rng.random()
            if roll < 0.02:
                g = [rng.uniform(-3000, 3000) for _ in range(3)]                  # a spike
            elif roll < 0.10:
                g = [v + rng.gauss(0, 400) for v in g]                            # a burst of motion
            else:
                g = [0.9 * v + rng.gauss(0, 5) for v in g]
            raw = tuple(max(-32768, min(32767, round(v * params.gyro_per_dps))) for v in g)      # int16, clipping
            sample = ImuSample(t_ns=t, g=raw, a=tuple(rng.randint(-2000, 2000) for _ in range(3)))
            if rng.random() < 0.01:
                det.blank(t, t + 150_000_000)
                shake.blank(t, t + 150_000_000)
            for event in det.feed(sample):
                assert event.kind in ("SWING_START", "IMPACT") and event.t_ns <= t
                assert all(math.isfinite(x) for x in (event.w_pk, event.dur_ms, *event.feat))
            until = shake.feed(sample)
            assert until is None or until > t


def test_a_swing_that_arms_the_moment_the_cool_down_ends_is_still_measured_from_its_own_onset():
    # The cool-down after one impact is still running through the next swing's backswing, so the detector
    # leaves it only on a sample that is already rising into the forward stroke.  Measuring that stroke from
    # the last time the detector happened to be idle (before the FIRST swing) makes it look a second long
    # and throws a real swing away.
    samples = stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0) - pulse(t, 0.72, 0.26, 300.0))
    first = next(i for i, s in enumerate(samples) if (s.t_ns - T0) / 1e9 > 0.99)
    for i, rate in enumerate((-60.0, 80.0, 400.0, 650.0, 800.0, 600.0, 250.0, 60.0, 0.0)):
        samples[first + i] = ImuSample(t_ns=samples[first + i].t_ns, g=(round(rate * GPD), 0, 0), a=(0, 0, 1000))
    for i in range(first + 9, len(samples)):
        samples[i] = ImuSample(t_ns=samples[i].t_ns, g=(0, 0, 0), a=(0, 0, 1000))
    hits = impacts(run(new(), samples))
    assert len(hits) == 2 and hits[1][1].dur_ms < 300.0


def test_a_forward_stroke_right_after_the_backswing_is_not_swallowed_by_the_cool_down():
    # The cool-down after an impact ends once its time is up AND the motion has died down, but the next swing's
    # backswing flips straight into its forward stroke without a quiet sample in between: the stroke must still
    # arm.  (A forward rate below zero means the last stroke is over, so there is nothing left to cool.)
    samples = stream(2.0, lambda t: pulse(t, 0.6, 0.15, 800.0) - pulse(t, 0.80, 0.30, 300.0))
    first = next(i for i, s in enumerate(samples) if (s.t_ns - T0) / 1e9 > 1.04)          # the cool-down is over
    for i, rate in enumerate((-250.0, -120.0, 140.0, 500.0, 800.0, 600.0, 250.0, 60.0, 0.0)):
        samples[first + i] = ImuSample(t_ns=samples[first + i].t_ns, g=(round(rate * GPD), 0, 0), a=(0, 0, 1000))
    for i in range(first + 9, len(samples)):
        samples[i] = ImuSample(t_ns=samples[i].t_ns, g=(0, 0, 0), a=(0, 0, 1000))
    hits = impacts(run(new(), samples))
    assert len(hits) == 2 and hits[1][1].dur_ms < 300.0


def test_an_impact_carries_the_gyro_vector_at_its_peak_and_the_rotation_over_the_stroke():
    # the stroke about x (800 dps) with a twist about y (200 dps) at the same time: the game turns the twist into spin
    samples = []
    for i in range(int(2.0 * HZ) + 1):
        t = i / HZ
        gx, gy = pulse(t, 0.6, 0.30, 800.0), pulse(t, 0.6, 0.30, 200.0)
        samples.append(ImuSample(t_ns=T0 + int(t * 1e9), g=(round(gx * GPD), round(gy * GPD), 0), a=(0, 0, 1000)))
    found = impacts(run(new(), samples))
    assert len(found) == 1
    ev = found[0][1]
    assert ev.g_dps[0] == pytest.approx(800.0, rel=0.05) and ev.g_dps[1] == pytest.approx(200.0, rel=0.1) and abs(ev.g_dps[2]) < 5
    assert ev.net_rot_deg[0] > 40.0 and ev.net_rot_deg[1] > 10.0                      # degrees turned during the forward phase
    assert ev.net_rot_deg[1] / ev.net_rot_deg[0] == pytest.approx(0.25, rel=0.1)
    norm = math.sqrt(sum(c * c for c in ev.g_dps))
    assert ev.axis_unit == pytest.approx(tuple(c / norm for c in ev.g_dps), abs=1e-6)       # the same peak, as a direction
    assert ev.g_dps[0] <= ev.w_pk * 1.05 + 1.0                                                # the forward rate is its projection


def test_an_event_made_by_hand_has_no_rotation_to_report():
    from pingpong.events import SwingEvent

    e = SwingEvent(kind="IMPACT", t_ns=1, w_pk=500.0, dur_ms=150.0, n_reversals=0, axis_unit=(1, 0, 0),
                   net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1), clipped=False, feat=(0.0,) * 12)
    assert e.g_dps == (0.0, 0.0, 0.0) and e.net_rot_deg == (0.0, 0.0, 0.0)

