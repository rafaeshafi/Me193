"""MediaPipe plumbing copied from P1-Pose-Race/pose_features.py (model path edited to models/).

mediapipe is imported lazily so importing this module (and the test-suite) stays fast and
works where the model/camera are absent.  Pins that make it run on Apple silicon are in
requirements.txt (mediapipe 0.10.21 -> numpy<2 -> opencv 4.10.0.84).
"""

from pathlib import Path

POSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"


def make_landmarker(*, video_mode=True):
    """A MediaPipe PoseLandmarker over the bundled pre-trained model (one person)."""
    from mediapipe.tasks.python import BaseOptions, vision

    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing pose model: {POSE_MODEL_PATH} (copy it from P1-Pose-Race)")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_mp_image(bgr_frame):
    """OpenCV gives BGR; MediaPipe wants RGB in its own Image wrapper."""
    import cv2
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
