"""The bundled MediaPipe models really load and run offline (CPU), on a blank frame."""

from types import SimpleNamespace

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


def test_the_full_model_ships_with_mediapipe_and_runs_the_same_way_on_a_blank_frame():
    landmarker = pose_features.make_landmarker(model="full")
    try:
        image = pose_features.to_mp_image(np.zeros((360, 640, 3), dtype=np.uint8))
        assert landmarker.detect_for_video(image, 0).pose_landmarks == []
        assert landmarker.detect_for_video(image, 33).pose_landmarks == []         # a second frame: tracking state is fine
    finally:
        landmarker.close()


def test_a_model_this_game_does_not_know_is_refused_with_the_ones_it_does():
    with pytest.raises(ValueError, match="lite.*full"):
        pose_features.make_landmarker(model="heavy")


def test_the_legacy_full_model_answers_in_the_shape_the_vision_worker_reads():
    seen = []

    class Legacy:                                              # mediapipe.solutions.pose.Pose-shaped
        def process(self, rgb):
            seen.append(rgb)
            points = [SimpleNamespace(x=i / 100, y=0.5, z=0.0, visibility=0.9) for i in range(33)]
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))

        def close(self):
            seen.append("closed")

    landmarker = pose_features.LegacyPoseLandmarker(pose=Legacy())
    image = pose_features.to_mp_image(np.full((36, 64, 3), 7, dtype=np.uint8))
    result = landmarker.detect_for_video(image, 0)
    assert len(result.pose_landmarks) == 1 and len(result.pose_landmarks[0]) == 33
    assert result.pose_landmarks[0][11].x == pytest.approx(0.11) and result.pose_landmarks[0][11].visibility == 0.9
    assert seen[0].shape == (36, 64, 3) and seen[0].dtype == np.uint8
    landmarker.close()
    assert seen[-1] == "closed"


def test_the_legacy_model_finding_nobody_gives_the_empty_list_the_tasks_model_does():
    class Empty:
        def process(self, rgb):
            return SimpleNamespace(pose_landmarks=None)

    landmarker = pose_features.LegacyPoseLandmarker(pose=Empty())
    image = pose_features.to_mp_image(np.zeros((36, 64, 3), dtype=np.uint8))
    assert landmarker.detect_for_video(image, 0).pose_landmarks == []
