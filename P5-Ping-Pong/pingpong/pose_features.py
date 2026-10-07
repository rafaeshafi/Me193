"""MediaPipe plumbing copied from P1-Pose-Race/pose_features.py (model path edited to models/).

mediapipe is imported lazily so importing this module (and the test-suite) stays fast and
works where the model/camera are absent.  Pins that make it run on Apple silicon are in
requirements.txt (mediapipe 0.10.21 -> numpy<2 -> opencv 4.10.0.84).
"""

from pathlib import Path
from types import SimpleNamespace

POSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
LANDMARKERS = ("lite", "full")          # the pose models a player can run: the fast one, and the larger, steadier one


class LegacyPoseLandmarker:
    """The larger "full" pose model, through mediapipe's older solutions API (the model ships inside the package; the
    Tasks API wants it as a separate file), answering the way the Tasks landmarker does: detect_for_video() gives an
    object whose pose_landmarks is [] or a list with one body of 33 landmarks (x, y, visibility).

    Its own landmark smoothing is off: the game filters the hand itself, and the filter has to see the raw readings."""

    def __init__(self, pose=None, *, static_image_mode=False):
        if pose is None:
            import mediapipe as mp

            pose = mp.solutions.pose.Pose(static_image_mode=static_image_mode, model_complexity=1, smooth_landmarks=False,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._pose = pose

    def detect_for_video(self, image, ts_ms):
        found = self._pose.process(image.numpy_view()).pose_landmarks
        return SimpleNamespace(pose_landmarks=[list(found.landmark)] if found else [])

    def close(self):
        self._pose.close()


def make_landmarker(*, video_mode=True, model="lite"):
    """A MediaPipe pose landmarker over a pre-trained model (one person): "lite" is the model bundled in models/ (fast),
    "full" the larger one that ships with mediapipe (steadier, slower: ./pp train_pose measures which suits the player)."""
    if model not in LANDMARKERS:
        raise ValueError(f"unknown pose model {model!r}: choose one of {', '.join(LANDMARKERS)}")
    if model == "full":
        return LegacyPoseLandmarker(static_image_mode=not video_mode)
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
