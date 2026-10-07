"""The swings a person really makes while playing are softer than the 'soft' swings of the calibration.

Two live games on 2026-10-07 (Rookie rally, recorded with the calibration below) lost half of their missed balls to one
cause: a real swing of 210-300 dps right at the ball that the weakest-swing threshold (0.7 x the calibration's soft
strength = 321 dps) dropped.  tests/data/real_play_swings.jsonl holds six of those swings, cut from the recorded hub data."""

import dataclasses

import pytest

import real_takes as real
from pingpong import calibration, fixtures
from pingpong.swing import SwingDetector

S = 1_000_000_000
TAKES = fixtures.load_takes(real.DATA.parent / "real_play_swings.jsonl")
# the calibration those games were played with (data/players/rafae, saved 17:35) and the hub's measured units
CAL = calibration.SwingCalibration((-0.3686453255516036, -0.7747006984585711, 0.5137503788385507),
                                   457.89116420728544, 1194.9351296656782)
UNITS = (0.988, 1017.73, 32767)
OLD_THRESHOLD = 0.7 * CAL.omega_lo


def params(**kw):
    return dataclasses.replace(CAL.swing_params(*UNITS), **kw)


def found_at_the_peak(take, p):
    """The IMPACT the detector gives at the take's recorded swing, or None."""
    detector, peak = SwingDetector(p), take["meta"]["peak_ns"]
    for sample in real.lead_in(take) + take["samples"]:
        for e in detector.feed(sample):
            if e.kind == "IMPACT" and abs((e.t_ns - take["samples"][0].t_ns) - peak) <= 60_000_000:
                return e
    return None


def test_the_fixture_is_six_real_swings_all_below_the_old_threshold():
    assert len(TAKES) == 6
    strengths = [t["meta"]["w_pk"] for t in TAKES]
    assert 200 <= min(strengths) and max(strengths) < OLD_THRESHOLD


@pytest.mark.parametrize("index", range(6))
def test_the_old_threshold_dropped_every_one_of_them(index):
    assert found_at_the_peak(TAKES[index], params(t_pk=OLD_THRESHOLD)) is None


@pytest.mark.parametrize("index", range(6))
def test_the_default_threshold_finds_every_one_of_them_at_its_own_peak(index):
    take = TAKES[index]
    event = found_at_the_peak(take, params())
    assert event is not None, f"a real {take['meta']['w_pk']:.0f} dps swing at a ball was not detected (threshold {params().t_pk:.0f})"
    assert event.w_pk == pytest.approx(take["meta"]["w_pk"], rel=0.06)


def test_the_weakest_counting_swing_is_under_half_of_the_soft_strength_and_the_camera_keeps_its_own_factor():
    assert calibration.T_PK_FACTOR == 0.42
    assert CAL.t_pk == pytest.approx(0.42 * CAL.omega_lo) and CAL.t_pk < 200.0       # 10 dps of margin under the weakest swing (210)
    camera = calibration.SwingCalibration((1.0, 0.0, 0.0), 300.0, 800.0, source="pose")
    assert camera.t_pk == pytest.approx(0.7 * 300.0)                  # tuned on scripted hands only: unchanged


def test_the_shake_lock_keeps_the_floor_it_was_tuned_with_whatever_the_swing_threshold():
    # 0.35 x the old threshold (0.7 x omega_lo): lowering the swing threshold must not make the lock twitchier
    assert CAL.shake_rms_dps == pytest.approx(0.35 * 0.7 * CAL.omega_lo)
    from pingpong import fakerig

    rig = fakerig.FakeRig()                                           # its calibration: omega_lo 300 dps
    assert rig.rig.imu.shake.rms_min == pytest.approx(0.35 * 0.7 * 300.0)
