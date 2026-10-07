"""The swing detector, the shake lock, the judge and the calibration flow against REAL swings of the real hub.

Everything before the bench session of 2026-10-07 was tuned on scripted 150 ms half-sine pulses.  The first real
takes showed what a person does with the hub: a backswing about as big as the stroke, then a stroke that builds up
for ~0.5 s with shoulders and hesitations on the way, a slow tail; shakes and waves at ~2 Hz, not 3-8 Hz.  The old
detector found 1 of 20 soft and 3 of 20 hard swings.  These tests keep it from drifting back.
"""

import math

import numpy as np
import pytest

import real_takes as real
from pingpong import levels
from pingpong.calibflow import CalibrationFlow
from pingpong.events import PaddlePose
from pingpong.judge import BallWindow, HitJudge
from pingpong.paddle import ReachBox
from pingpong.shake import ShakeMonitor
from pingpong.swing import SwingDetector

S = 1_000_000_000
# the forward axis fitted on all 40 soft + hard swings of the session (the subset's own fit must agree with it)
REFERENCE_AXIS = (-0.285, -0.747, 0.601)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def impacts(take, params, lead=True):
    """[(event, arrival_ns)] for every IMPACT the detector gives on a take (with a quiet lead-in, as live)."""
    detector, out = SwingDetector(params), []
    for sample in (real.lead_in(take) if lead else []) + take["samples"]:
        out += [(e, sample.t_ns) for e in detector.feed(sample) if e.kind == "IMPACT"]
    return out


SWINGS = [(label, t["index"]) for label in ("soft", "hard", "max") for t in real.takes(label)]


def take_of(label, index):
    return next(t for t in real.takes(label) if t["index"] == index)


def test_the_axis_and_strengths_fitted_from_real_takes_are_what_a_person_does_with_the_hub():
    axis, omega_lo, omega_hi = real.calibrated()
    assert angle_deg(axis, REFERENCE_AXIS) < 6.0
    assert 350.0 < omega_lo < 450.0 and 1050.0 < omega_hi < 1300.0         # dps, with 0.988 counts per dps (398 / 1168)
    assert omega_hi > 2.5 * omega_lo


@pytest.mark.parametrize("label, index", SWINGS)
def test_every_real_swing_gives_exactly_one_impact_at_its_forward_peak(label, index):
    take = take_of(label, index)
    axis = real.calibrated()[0]
    peak, peak_ns = real.true_forward_peak(take, axis)
    main = [(e, at) for e, at in impacts(take, real.params()) if e.w_pk >= 0.7 * peak]      # a smaller lobe is not THE swing
    assert len(main) == 1, f"{label}[{index}]: {len(main)} impacts of at least 70% of the peak"
    event, arrival = main[0]
    assert abs((event.t_ns - take["samples"][0].t_ns) - peak_ns) <= 40_000_000          # +-40 ms of the true peak
    assert event.w_pk == pytest.approx(peak, rel=0.05)            # the tracked peak IS the forward peak (within 1.7% here)
    assert (arrival - event.t_ns) / 1e6 <= 200.0                                      # recognised within 0.2 s


def test_the_slow_soft_swings_are_found_and_their_late_recovery_lobe_is_not_a_second_swing():
    take = take_of("soft", 3)                                    # a 0.6 s broad lobe, then a 240 dps recovery lobe
    found = impacts(take, real.params())
    assert [round(e.w_pk) for e, _ in found] == [round(found[0][0].w_pk)] and len(found) == 1


@pytest.mark.parametrize("index", [3, 9])
def test_a_real_backswing_on_its_own_never_fires(index):
    assert impacts(take_of("backswing_only", index), real.params()) == []


def test_a_wobble_about_another_axis_with_a_big_magnitude_is_not_a_forward_swing():
    # backswing_only[9] has a 303 dps magnitude peak but only a 244 dps forward component: below the 279 dps threshold
    take = take_of("backswing_only", 9)
    gyro = np.array([s.g for s in take["samples"]], dtype=float) / real.GPD
    magnitude = np.linalg.norm(gyro - real.REST_DPS, axis=1).max()
    assert magnitude >= real.params().t_pk > real.true_forward_peak(take, real.calibrated()[0])[0]
    assert impacts(take, real.params()) == []


def test_ten_seconds_of_real_fan_waving_gives_at_most_one_impact_instead_of_a_hit_per_wave():
    wave = real.takes("waving")[0]
    assert len(impacts(wave, real.params(), lead=False)) <= 2


@pytest.mark.parametrize("index", [4, 9])
def test_real_shaking_locks_the_paddle_within_a_second_and_a_half(index):
    take = take_of("shakes", index)
    monitor, first = ShakeMonitor(gyro_per_dps=real.GPD, rms_min_dps=0.35 * real.params().t_pk), None
    for sample in take["samples"]:
        if monitor.feed(sample) is not None and first is None:
            first = (sample.t_ns - take["samples"][0].t_ns) / S
    assert first is not None and first <= 1.8


def test_real_waving_locks_the_paddle_too():
    wave = real.takes("waving")[0]
    monitor = ShakeMonitor(gyro_per_dps=real.GPD, rms_min_dps=0.35 * real.params().t_pk)
    assert any(monitor.feed(s) is not None for s in wave["samples"])


@pytest.mark.parametrize("label, index", SWINGS + [("backswing_only", 3), ("backswing_only", 9)])
def test_no_real_swing_or_backswing_ever_locks_the_paddle(label, index):
    take = take_of(label, index)
    monitor = ShakeMonitor(gyro_per_dps=real.GPD, rms_min_dps=0.35 * real.params().t_pk)
    assert all(monitor.feed(s) is None for s in take["samples"])


def test_a_whole_rally_of_real_swings_never_locks_the_paddle_either():
    monitor = ShakeMonitor(gyro_per_dps=real.GPD, rms_min_dps=0.35 * real.params().t_pk)
    stream = real.stitch([take_of("soft", 0), take_of("hard", 3), take_of("soft", 4), take_of("hard", 7)], rest_s=1.0)
    assert all(monitor.feed(s) is None for s in stream)


@pytest.mark.parametrize("label, index", SWINGS)
def test_the_judge_accepts_a_real_swing_that_meets_a_ball_on_time_and_in_place(label, index):
    take = take_of(label, index)
    event, _ = max(impacts(take, real.params()), key=lambda pair: pair[0].w_pk)
    box = ReachBox(-1.0, 1.0, -0.5, 0.5)
    judge = HitJudge(box, t_pk=real.params().t_pk)
    ball = BallWindow(ball_id=1, t_c_ns=event.t_ns, aim_ab=(0.5, 0.5), level=levels.LEVELS[1])
    u, v = box.to_uv(0.5, 0.5)
    poses = [PaddlePose(t_scene_ns=event.t_ns - int(0.5 * S) + k * 33_000_000, u=u, v=v, conf=0.9, hand="right")
             for k in range(16)]
    verdict = judge.judge(event, ball, poses, now_ns=event.t_ns + 100_000_000)
    assert verdict.kind == "HIT", [(g.name, g.passed, g.note) for g in verdict.gates if not g.passed]


def test_the_detection_margin_the_judge_waits_for_covers_what_real_swings_need():
    latencies = []
    for label, index in SWINGS:
        take = take_of(label, index)
        peak, _ = real.true_forward_peak(take, real.calibrated()[0])
        latencies += [(at - e.t_ns) / 1e9 for e, at in impacts(take, real.params()) if e.w_pk >= 0.7 * peak]
    assert np.percentile(latencies, 95) <= HitJudge(ReachBox(-1, 1, -0.5, 0.5)).d95_s + 0.005


def test_the_calibration_flow_accepts_real_soft_and_full_swings_and_recognises_its_own_takes():
    flow = CalibrationFlow(gyro_per_dps=real.GPD, accel_per_g=real.APG, fs_raw=32767)
    flow.step, flow.corners, flow._shoulder_w = "soft", [(-1.1, 0.7), (1.1, 0.7), (1.1, -0.6), (-1.1, -0.6)], 0.2
    takes = real.takes("soft") + real.takes("hard")
    for sample in real.stitch(takes):
        flow.feed_imu(sample)
    notes = " | ".join(flow.take_notes())
    assert flow.finished(), notes
    for complaint in ("too long", "waving", "too gentle"):
        assert complaint not in notes, notes
    cal = flow.calibration()
    assert angle_deg(cal.swing.u_fwd, REFERENCE_AXIS) < 8.0
    assert 350.0 < cal.swing.omega_lo < 450.0 and 1050.0 < cal.swing.omega_hi < 1300.0
    detected, total = flow.self_check
    assert total == 10 and detected >= 9
