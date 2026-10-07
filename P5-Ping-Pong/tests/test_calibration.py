import math
import random

import numpy as np
import pytest

from pingpong import calibration
from pingpong.events import ImuSample

GPD = 10.0
T0 = 1_000_000_000
HZ = 66.0


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def lobe(t, start, dur, peak):
    return peak * math.sin(math.pi * (t - start) / dur) if start <= t <= start + dur else 0.0


def take(axis, forward_dps, back_dps=0.0, noise=3.0, seed=0, off_axis=0.0):
    """One swing: a backswing lobe against `axis`, then a forward lobe along it (all in dps)."""
    rng = random.Random(seed)
    u = unit(axis)
    ortho1 = unit(np.cross(u, [0.3, 0.2, 0.9]))
    ortho2 = unit(np.cross(u, ortho1))
    wobble = off_axis * (rng.uniform(-1, 1) * ortho1 + rng.uniform(-1, 1) * ortho2)   # random per swing
    out = []
    for i in range(int(2.0 * HZ)):
        t = i / HZ
        s = lobe(t, 0.5, 0.2, -back_dps) + lobe(t, 0.78, 0.15, forward_dps)
        g = s * u + s * wobble + np.array([rng.gauss(0, noise) for _ in range(3)])
        out.append(ImuSample(t_ns=T0 + int(t * 1e9), g=tuple(int(round(c * GPD)) for c in g), a=(0, 0, 1000)))
    return out


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b)))))))


AXIS = (0.6, -0.3, 0.74)


def test_the_forward_axis_is_recovered_within_five_degrees():
    takes = [take(AXIS, 700 + 60 * k, 300, seed=k, off_axis=0.15) for k in range(8)]
    u, peaks = calibration.fit_forward_axis(takes, gyro_per_dps=GPD)
    assert angle_deg(u, AXIS) < 5.0
    assert len(peaks) == 8 and all(600 < p < 1300 for p in peaks)


def test_the_axis_sign_follows_the_forward_stroke_even_when_the_backswing_is_big():
    takes = [take(AXIS, 800, 500, seed=k) for k in range(6)]
    u, _ = calibration.fit_forward_axis(takes, gyro_per_dps=GPD)
    assert float(np.dot(u, unit(AXIS))) > 0.95


def test_fitting_needs_at_least_three_takes():
    with pytest.raises(ValueError):
        calibration.fit_forward_axis([take(AXIS, 700), take(AXIS, 750)], gyro_per_dps=GPD)


def test_soft_and_full_peaks_become_omega_lo_and_hi_by_median():
    cal = calibration.swing_strengths(soft=[310, 290, 305, 800, 300], full=[1000, 1100, 1050, 1020, 400])
    assert cal["omega_lo"] == pytest.approx(305)
    assert cal["omega_hi"] == pytest.approx(1020)
    assert cal["t_pk"] == pytest.approx(calibration.T_PK_FACTOR * 305)


def test_soft_and_full_must_be_clearly_separated():
    with pytest.raises(ValueError, match="separated"):
        calibration.swing_strengths(soft=[500] * 5, full=[520] * 5)


def test_a_profile_round_trips_through_json_and_validates_its_axis():
    cal = calibration.SwingCalibration(u_fwd=(0.6, -0.3, 0.74), omega_lo=300.0, omega_hi=1100.0)
    back = calibration.SwingCalibration.from_json(cal.to_json())
    assert back.omega_lo == 300.0 and back.omega_hi == 1100.0
    assert back.t_pk == pytest.approx(calibration.T_PK_FACTOR * 300.0)
    assert math.sqrt(sum(c * c for c in back.u_fwd)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        calibration.SwingCalibration(u_fwd=(0.0, 0.0, 0.0), omega_lo=300.0, omega_hi=1100.0)


def test_detector_params_come_from_the_calibration():
    cal = calibration.SwingCalibration(u_fwd=(1.0, 0.0, 0.0), omega_lo=400.0, omega_hi=1200.0)
    params = cal.swing_params(gyro_per_dps=GPD, accel_per_g=980.0, fs_raw=30000)
    assert params.t_pk == pytest.approx(calibration.T_PK_FACTOR * 400.0) and params.gyro_per_dps == GPD \
        and params.fs_raw == 30000


def test_a_camera_calibration_drops_the_hub_spike_gate_and_asks_for_longer_swings():
    hub = calibration.SwingCalibration((1.0, 0.0, 0.0), 300.0, 1000.0).swing_params(GPD, 1000.0, 32767)
    cam = calibration.SwingCalibration((1.0, 0.0, 0.0), 300.0, 1000.0, source="pose").swing_params(GPD, 1000.0, 32767)
    assert (hub.spike_ratio, hub.min_dur_ms, hub.refractory_s) == (0.4, 60.0, 0.30)
    assert (cam.spike_ratio, cam.min_dur_ms, cam.refractory_s) == (0.0, 100.0, 0.15)
    assert hub.t_pk == pytest.approx(calibration.T_PK_FACTOR * 300.0)
    assert cam.t_pk == pytest.approx(calibration.POSE_T_PK_FACTOR * 300.0)          # the camera keeps its scripted-hand factor
    assert (cam.max_dur_ms, hub.max_dur_ms) == (400.0, 2000.0)           # real hub swings last up to ~1.4 s


def test_a_swing_source_other_than_imu_or_pose_is_refused():
    with pytest.raises(ValueError, match="source"):
        calibration.SwingCalibration((1.0, 0.0, 0.0), 300.0, 1000.0, source="lidar")


def test_the_source_survives_json_and_an_old_file_without_one_is_an_imu_calibration():
    cam = calibration.SwingCalibration((0.0, 1.0, 0.0), 250.0, 800.0, source="pose")
    assert calibration.SwingCalibration.from_json(cam.to_json()) == cam
    old = '{"u_fwd": [0.0, 1.0, 0.0], "omega_lo": 250.0, "omega_hi": 800.0}'
    assert calibration.SwingCalibration.from_json(old).source == "imu"


# --- strengths measured on the forward rate (real swings: the backswing is often as big as the stroke) -------------------
def lobes_take(gx_dps, axis_idx=0, off=None, hz=64.0):
    out = []
    for i, v in enumerate(gx_dps):
        g = [0, 0, 0]
        g[axis_idx] = round(v * GPD)
        if off:
            g = [g[k] + round(off[k] * GPD) for k in range(3)]
        out.append(ImuSample(t_ns=T0 + int(i * 1e9 / hz), g=tuple(g), a=(0, 0, 1000)))
    return out


def test_a_swings_strength_is_its_forward_peak_even_when_the_backswing_is_bigger():
    # real soft swings: the backswing is about as big as the stroke, so the largest MAGNITUDE can be the wrong lobe
    take = lobes_take([0] * 12 + [-200, -450, -600, -450, -200, 0, 150, 300, 400, 300, 150, 0])
    assert calibration.forward_peaks([take], (1.0, 0.0, 0.0), GPD, bias=(0.0, 0.0, 0.0)) == [pytest.approx(400.0)]


def test_only_the_part_of_the_motion_along_the_axis_counts_as_forward():
    take = lobes_take([0] * 12 + [200, 400, 200, 0], axis_idx=1)           # a big rotation about ANOTHER axis
    assert calibration.forward_peaks([take], (1.0, 0.0, 0.0), GPD, bias=(0.0, 0.0, 0.0)) == [pytest.approx(0.0, abs=1e-6)]


def test_a_constant_gyro_offset_is_not_part_of_the_strength():
    take = lobes_take([0] * 12 + [200, 400, 200, 0], off=(50.0, 0.0, 0.0))
    assert calibration.forward_peaks([take], (1.0, 0.0, 0.0), GPD, bias=(50.0, 0.0, 0.0)) == [pytest.approx(400.0, abs=1.0)]


def test_the_rest_offset_is_always_given_because_real_takes_do_not_start_at_rest():
    # a person is already into the backswing at the GO beep: the take's first samples are not the hub's rest offset
    take = lobes_take([300, 250, 200, 100, 0, -300, -600, -300, 0, 150, 300, 400, 300, 150, 0])
    with pytest.raises(TypeError):
        calibration.forward_peaks([take], (1.0, 0.0, 0.0), GPD)
    assert calibration.forward_peaks([take], (1.0, 0.0, 0.0), GPD, bias=(0.0, 0.0, 0.0)) == [pytest.approx(400.0)]


def test_the_weakest_counting_swing_is_a_fraction_of_the_soft_strength_set_from_real_play():
    # 0.7 dropped half the intended swings of two real games (tests/test_play_swings.py)
    assert calibration.T_PK_FACTOR == 0.42
    assert calibration.SwingCalibration((1.0, 0.0, 0.0), 440.0, 1150.0).t_pk == pytest.approx(0.42 * 440.0)
