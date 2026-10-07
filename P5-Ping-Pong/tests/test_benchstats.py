"""Pure analysis used by env_check and the bench tools."""

import math

import pytest

from pingpong import benchstats

MS = 1_000_000


def test_rate_stats_of_a_steady_15ms_stream():
    stamps = [i * 15 * MS for i in range(201)]
    s = benchstats.rate_stats(stamps)
    assert s["n"] == 201
    assert s["hz"] == pytest.approx(66.67, abs=0.1)
    assert s["worst_gap_ms"] == pytest.approx(15.0)
    assert s["gaps_over_100ms"] == 0


def test_rate_stats_flags_a_dropout():
    stamps = [i * 20 * MS for i in range(50)]
    stamps += [stamps[-1] + 250 * MS + i * 20 * MS for i in range(50)]
    s = benchstats.rate_stats(stamps)
    assert s["worst_gap_ms"] == pytest.approx(250.0)
    assert s["gaps_over_100ms"] == 1
    assert s["p999_gap_ms"] >= 20.0


def test_rate_stats_handles_too_few_samples():
    assert benchstats.rate_stats([])["hz"] == 0.0
    assert benchstats.rate_stats([5 * MS])["n"] == 1


def test_rate_stats_histogram_counts_every_gap():
    stamps = [0, 10 * MS, 30 * MS, 130 * MS, 135 * MS]
    s = benchstats.rate_stats(stamps)
    assert sum(s["histogram"].values()) == 4


@pytest.mark.parametrize("hz, verdict", [(66.0, "GO"), (40.0, "GO"), (39.9, "WARN"),
                                         (25.0, "WARN"), (24.9, "NO-GO"), (0.0, "NO-GO")])
def test_rate_verdict_thresholds_match_the_plan(hz, verdict):
    assert benchstats.rate_verdict(hz) == verdict


def test_accel_scale_from_six_rest_faces():
    # Every face reads |a| = 1 g; the hub reports 980 counts per g here.
    faces = [(980, 0, 0), (-980, 0, 0), (0, 980, 0), (0, -980, 0), (0, 0, 980), (0, 0, -980)]
    res = benchstats.accel_scale_from_faces(faces)
    assert res["accel_per_g"] == pytest.approx(980.0)
    assert res["cv"] == pytest.approx(0.0, abs=1e-9)
    assert res["ok"] is True


def test_accel_scale_rejects_faces_that_disagree_by_more_than_three_percent():
    faces = [(1000, 0, 0), (-1000, 0, 0), (0, 1000, 0), (0, -1000, 0), (0, 0, 1100), (0, 0, -1100)]
    assert benchstats.accel_scale_from_faces(faces)["ok"] is False


def test_gyro_scale_from_a_full_turn():
    # 10 raw counts per deg/s, constant 90 deg/s for 4 s = one 360 degree turn.
    dt = 0.015
    n = int(4.0 / dt)
    samples = [(i * dt, 900.0) for i in range(n + 1)]
    scale = benchstats.gyro_scale_from_turn(samples, true_degrees=360.0)
    assert scale == pytest.approx(10.0, rel=0.02)


def test_xcorr_lag_recovers_a_known_delay():
    fs = 60.0
    n = 400
    base = [math.sin(2 * math.pi * 1.3 * i / fs) * math.exp(-((i - 200) / 60.0) ** 2) for i in range(n)]
    delay = 7  # samples = 0.1167 s
    shifted = [0.0] * delay + base[:-delay]
    lag = benchstats.xcorr_lag_s(base, shifted, fs, max_lag_s=0.5)
    assert lag == pytest.approx(delay / fs, abs=1.5 / fs)


def _wave(seconds=12.0):
    """An irregular hand wave (two sines) -> position u(t) and the speed the gyro would feel."""
    import numpy as np

    def u(t):
        return 0.8 * np.sin(2 * np.pi * 1.1 * t) + 0.5 * np.sin(2 * np.pi * 1.9 * t + 0.7)

    def speed(t):
        du = 0.8 * 2 * np.pi * 1.1 * np.cos(2 * np.pi * 1.1 * t) + 0.5 * 2 * np.pi * 1.9 * np.cos(2 * np.pi * 1.9 * t + 0.7)
        return np.abs(du)

    return u, speed


@pytest.mark.parametrize("tau", [0.04, 0.10, 0.20])
def test_the_camera_lag_is_recovered_from_a_wave_within_a_frame(tau):
    import numpy as np

    u, speed = _wave()
    imu_t = np.arange(0, 12.0, 1 / 66.0)
    pose_t = np.arange(0.0, 12.0, 1 / 30.0)
    gyro = np.stack([300.0 * speed(imu_t), np.zeros_like(imu_t), np.zeros_like(imu_t)], axis=1)
    pose_uv = np.stack([u(pose_t - tau), 0.1 * u(pose_t - tau)], axis=1)       # the camera shows the past
    lag, quality = benchstats.wave_lag_s((pose_t * 1e9).astype(int), pose_uv, (imu_t * 1e9).astype(int), gyro)
    assert lag == pytest.approx(tau, abs=0.015) and quality > 0.8


def test_unrelated_signals_give_a_low_quality_so_the_tool_will_not_trust_the_lag():
    import numpy as np

    rng = np.random.default_rng(3)
    imu_t = np.arange(0, 12.0, 1 / 66.0)
    pose_t = np.arange(0.0, 12.0, 1 / 30.0)
    gyro = np.abs(rng.normal(0, 200, (len(imu_t), 3)))
    pose_uv = rng.normal(0, 0.5, (len(pose_t), 2))
    _, quality = benchstats.wave_lag_s((pose_t * 1e9).astype(int), pose_uv, (imu_t * 1e9).astype(int), gyro)
    assert quality < 0.35


def test_too_little_data_is_an_error_not_a_made_up_number():
    import numpy as np

    with pytest.raises(ValueError, match="wave"):
        benchstats.wave_lag_s(np.arange(10) * 33_000_000, np.zeros((10, 2)), np.arange(20) * 15_000_000, np.zeros((20, 3)))


def _pulse_stream(t_pulse=2.0, pulse_ms=60, ring_s=0.10, amp=800.0, noise=3.0, hz=66.0, seconds=4.0, seed=1):
    """Raw (t_ns, gyro, accel) samples: quiet noise, then a shake that decays after the pulse ends."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(seconds * hz)):
        t = i / hz
        shake = 0.0
        if t_pulse <= t < t_pulse + pulse_ms / 1000:
            shake = amp
        elif t_pulse + pulse_ms / 1000 <= t < t_pulse + pulse_ms / 1000 + ring_s:
            shake = amp * math.exp(-(t - t_pulse - pulse_ms / 1000) / (ring_s / 3))
        sign = 1 if i % 2 else -1
        g = (round(rng.normal(0, noise) + sign * shake), round(rng.normal(0, noise)), round(rng.normal(0, noise)))
        a = (round(rng.normal(0, noise)), round(rng.normal(0, noise)), round(1000 + rng.normal(0, noise) + sign * shake / 4))
        out.append((int(t * 1e9), g, a))
    return out


def test_the_hub_ringing_after_a_motor_pulse_is_measured_from_its_own_imu():
    samples = _pulse_stream(ring_s=0.10)
    r = benchstats.pulse_response(samples, t_pulse_ns=2_000_000_000, pulse_ms=60)
    assert r["felt_by_imu"] is True and r["peak_gyro"] > 400.0
    assert 0.03 < r["ringdown_s"] < 0.18                        # it rings for roughly the 0.10 s we built in
    longer = benchstats.pulse_response(_pulse_stream(ring_s=0.30), 2_000_000_000, 60)
    assert longer["ringdown_s"] > r["ringdown_s"] + 0.1


def test_a_pulse_the_imu_cannot_feel_is_reported_as_such():
    samples = _pulse_stream(amp=0.0)
    r = benchstats.pulse_response(samples, t_pulse_ns=2_000_000_000, pulse_ms=60)
    assert r["felt_by_imu"] is False and r["ringdown_s"] == 0.0


def test_the_blanking_time_is_the_longest_ring_plus_a_margin_and_stays_in_sane_bounds():
    assert benchstats.blank_after_pulse_s([0.10, 0.14, 0.08]) == pytest.approx(0.16)
    assert benchstats.blank_after_pulse_s([0.0, 0.0]) == pytest.approx(0.05)         # never below the floor
    assert benchstats.blank_after_pulse_s([0.9]) == pytest.approx(0.5)               # never absurdly long
    assert benchstats.blank_after_pulse_s([]) is None
