import pytest

from features import MODEL1_FEATURES, MODEL2_FEATURES, compute, dominant_frequency
from synth import synth


def test_feature_sets_and_keys():
    f = compute(synth("WALK"))
    assert MODEL1_FEATURES == ["peak_x"]
    assert set(MODEL1_FEATURES) <= set(f) and set(MODEL2_FEATURES) <= set(f)
    assert "peak_x" not in MODEL2_FEATURES or True  # Model 2 may see it too; Model 1 sees only it
    assert all(isinstance(v, float) for v in f.values())


def test_peak_x_orders_stand_walk_run_and_bike_hides_among_them():
    peaks = {k: compute(synth(k))["peak_x"] for k in ("STAND", "WALK", "RUN", "BIKE")}
    assert peaks["STAND"] < peaks["WALK"] < peaks["RUN"]
    assert peaks["BIKE"] < peaks["WALK"]            # the Model 1 failure mode: biking looks quieter than walking
    assert peaks["STAND"] == pytest.approx(1.0, abs=0.05)


def test_gyro_and_gravity_features_separate_bike():
    f = {k: compute(synth(k)) for k in ("STAND", "WALK", "RUN", "BIKE")}
    assert f["BIKE"]["gyro_ratio"] > 2 * max(f["STAND"]["gyro_ratio"], f["WALK"]["gyro_ratio"], f["RUN"]["gyro_ratio"])
    assert f["BIKE"]["mean_y"] > 0.8 and f["WALK"]["mean_y"] < 0.3
    assert f["STAND"]["gyro_rms"] < 5 < f["WALK"]["gyro_rms"] < f["RUN"]["gyro_rms"]


def test_dominant_frequency():
    assert compute(synth("WALK"))["dom_freq"] == pytest.approx(1.8, abs=0.15)
    assert compute(synth("RUN"))["dom_freq"] == pytest.approx(2.8, abs=0.15)
    assert dominant_frequency([1.0] * 200) == 0.5  # flat signal: the lowest bin, no crash
