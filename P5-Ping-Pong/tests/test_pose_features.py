"""The bundled MediaPipe model really loads and runs offline (CPU), on a blank frame."""

import numpy as np
import pytest

from pingpong import pose_features


def test_the_model_file_is_bundled_in_the_project():
    assert pose_features.POSE_MODEL_PATH.exists()
    assert pose_features.POSE_MODEL_PATH.stat().st_size > 1_000_000


def test_a_missing_model_gives_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(pose_features, "POSE_MODEL_PATH", tmp_path / "nope.task")
    with pytest.raises(FileNotFoundError, match="pose model"):
        pose_features.make_landmarker()


def test_the_landmarker_runs_on_a_blank_frame_and_finds_nobody():
    landmarker = pose_features.make_landmarker()
    try:
        image = pose_features.to_mp_image(np.zeros((360, 640, 3), dtype=np.uint8))
        result = landmarker.detect_for_video(image, 0)
        assert result.pose_landmarks == []
    finally:
        landmarker.close()
