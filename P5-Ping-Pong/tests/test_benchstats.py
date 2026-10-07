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
