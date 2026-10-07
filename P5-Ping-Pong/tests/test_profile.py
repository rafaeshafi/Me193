"""Calibration (swing axis + strengths, reach box, shoulder width) saved per player."""

import json

import pytest

from pingpong import calibration, profile
from pingpong.calibration import SwingCalibration
from pingpong.paddle import ReachBox


def sample():
    return profile.Calibration(
        swing=SwingCalibration(u_fwd=(0.0, 3.0, 4.0), omega_lo=310.0, omega_hi=1180.0),
        box=ReachBox(-1.2, 1.1, -0.7, 0.6), shoulder_w=0.21, hand="left")


def same(a, b):
    """Equal up to float noise in the normalised axis (re-normalising a unit vector moves the last bit)."""
    assert a.swing.u_fwd == pytest.approx(b.swing.u_fwd, abs=1e-12)
    assert (a.swing.omega_lo, a.swing.omega_hi) == (b.swing.omega_lo, b.swing.omega_hi)
    assert (a.box, a.shoulder_w, a.hand, a.calibrated) == (b.box, b.shoulder_w, b.hand, b.calibrated)


def test_a_calibration_round_trips_through_json():
    original = sample()
    again = profile.Calibration.from_json(original.to_json())
    same(again, original)
    assert again.swing.u_fwd == pytest.approx((0.0, 0.6, 0.8))          # stored normalised
    assert again.calibrated is True


def test_the_paddle_tilt_calibration_is_saved_with_the_rest_and_an_older_file_without_it_still_loads():
    from pingpong.tilt import TiltCalibration

    tilt = TiltCalibration(axis=(0.6, 0.8, 0.0), neutral=(0.0, 0.0, 1.0), bias_dps=(1.0, -2.0, 0.5))
    with_tilt = profile.Calibration(swing=sample().swing, box=sample().box, shoulder_w=0.21, hand="left", tilt=tilt)
    assert profile.Calibration.from_json(with_tilt.to_json()).tilt == tilt
    old = json.loads(sample().to_json())
    assert "tilt" not in old and profile.Calibration.from_json(json.dumps(old)).tilt is None
    assert profile.Calibration.default().tilt is None


def test_the_default_is_flagged_uncalibrated_so_the_game_can_say_so():
    default = profile.Calibration.default()
    assert default.calibrated is False and default.shoulder_w is None
    assert default.swing.omega_hi > default.swing.omega_lo


def test_saving_and_loading_by_player_name(tmp_path):
    profile.save("Rafae", sample(), root=tmp_path)
    assert (tmp_path / "rafae" / "calibration.json").exists()
    same(profile.load("rafae", root=tmp_path), sample())
    same(profile.load("RAFAE", root=tmp_path), sample())                # names are case-insensitive


def test_an_unknown_player_has_no_calibration(tmp_path):
    assert profile.load("nobody", root=tmp_path) is None


def test_player_names_become_safe_folder_names(tmp_path):
    assert profile.slug("Rafae Shafi") == "rafae-shafi"
    assert profile.slug("../../etc/passwd") == "etc-passwd"
    with pytest.raises(ValueError):
        profile.slug("   ")
    with pytest.raises(ValueError):
        profile.slug("../..")


def test_a_corrupt_file_is_reported_not_silently_replaced_by_defaults(tmp_path):
    (tmp_path / "rafae").mkdir()
    (tmp_path / "rafae" / "calibration.json").write_text("{not json")
    with pytest.raises(ValueError, match="calibration"):
        profile.load("rafae", root=tmp_path)


def test_swing_params_come_from_the_calibration_and_the_measured_units():
    params = sample().swing_params(gyro_per_dps=9.7, accel_per_g=980.0, fs_raw=30000)
    assert params.gyro_per_dps == 9.7 and params.accel_per_g == 980.0 and params.fs_raw == 30000
    assert params.t_pk == pytest.approx(calibration.T_PK_FACTOR * 310.0)
    assert params.u_fwd == pytest.approx((0.0, 0.6, 0.8))


def test_the_file_is_plain_readable_json(tmp_path):
    profile.save("rafae", sample(), root=tmp_path)
    data = json.loads((tmp_path / "rafae" / "calibration.json").read_text())
    assert data["hand"] == "left" and data["box"]["u_min"] == -1.2 and data["version"] == 1


def camera_sample():
    return profile.Calibration(
        swing=SwingCalibration(u_fwd=(1.0, 0.0, 0.0), omega_lo=240.0, omega_hi=760.0, source="pose"),
        box=ReachBox(-1.0, 1.0, -0.6, 0.5), shoulder_w=0.2)


def test_the_default_for_the_camera_swing_source_is_a_camera_calibration_flagged_uncalibrated():
    default = profile.Calibration.default("pose")
    assert default.swing_source == "pose" and default.calibrated is False
    assert profile.Calibration.default().swing_source == "imu"


def test_a_camera_calibration_lives_in_its_own_file_and_never_replaces_the_hub_one(tmp_path):
    profile.save("rafae", sample(), root=tmp_path)
    profile.save("rafae", camera_sample(), root=tmp_path)
    assert (tmp_path / "rafae" / "calibration.json").exists()
    assert (tmp_path / "rafae" / "calibration-pose.json").exists()
    assert profile.load("rafae", root=tmp_path).swing.omega_lo == 310.0                   # the hub one, untouched
    assert profile.load("rafae", root=tmp_path, source="pose").swing.omega_lo == 240.0


def test_a_player_with_only_a_hub_calibration_has_no_camera_calibration_yet(tmp_path):
    profile.save("rafae", sample(), root=tmp_path)
    assert profile.load("rafae", root=tmp_path, source="pose") is None
