"""Shared pose plumbing: the detector, the landmarks, and the feature vector.

collect_poses.py, train_poses.py and pose_car.py must agree *exactly* on what
a feature vector contains. If they drift apart, the classifier still loads and
still predicts — it just predicts nonsense, with no error to tell you why. So
there is one definition, here, and all three import it.
"""

import math
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

HERE = Path(__file__).parent
POSE_MODEL_PATH = HERE / "pose_landmarker_lite.task"   # Google's pre-trained pose model
DATA_PATH = HERE / "pose_data.csv"                     # examples you record
CLASSIFIER_PATH = HERE / "pose_model.joblib"           # the classifier you train

# --- Landmark indices, from MediaPipe's own numbering -----------------------
# These are ANATOMICAL: L_WRIST is the driver's left wrist however the image
# is mirrored.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

# The landmarks that make up a feature vector, in a fixed order. Upper body
# only — where your feet are has nothing to do with steering a car.
FEATURE_LANDMARKS = (
    ("nose", NOSE),
    ("l_shoulder", L_SHOULDER), ("r_shoulder", R_SHOULDER),
    ("l_elbow", L_ELBOW), ("r_elbow", R_ELBOW),
    ("l_wrist", L_WRIST), ("r_wrist", R_WRIST),
    ("l_hip", L_HIP), ("r_hip", R_HIP),
)
FEATURE_NAMES = tuple(f"{name}_{axis}"
                      for name, _ in FEATURE_LANDMARKS
                      for axis in ("x", "y"))

# Landmarks whose visibility we insist on. The arms carry the whole signal, so
# a guess there is worse than no reading. Nose and hips are allowed to be
# uncertain — they only set the frame of reference, and hips in particular are
# often cropped out when the driver stands close to the camera.
REQUIRED_LANDMARKS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)

MIN_VISIBILITY = 0.5      # below this a landmark is a guess, not a measurement
MIN_SHOULDER_FRAC = 0.08  # shoulders narrower than this fraction of the frame

# What each trained pose class tells the car to do, as (left%, right%).
CLASS_SPEEDS = {
    "forward": (70, 70),
    "back": (-70, -70),
    "left": (-35, 75),
    "right": (75, -35),
    "stop": (0, 0),
}
DEFAULT_CLASSES = ("forward", "back", "left", "right", "stop")


def make_landmarker(*, video_mode=True):
    """Build a MediaPipe PoseLandmarker over the bundled pre-trained model."""
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing pose model: {POSE_MODEL_PATH}\nSee README.md for the download command.")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_mp_image(bgr_frame):
    """OpenCV gives BGR; MediaPipe wants RGB in its own Image wrapper."""
    import cv2
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


def pose_geometry(landmarks, width, height):
    """Return (points_in_pixels, shoulder_midpoint, shoulder_width) or None.

    None means "this frame is not worth reading" — see the two guards below.
    """
    if any(landmarks[i].visibility < MIN_VISIBILITY for i in REQUIRED_LANDMARKS):
        return None

    # Normalized coords are fractions of width/height, so they have to be
    # scaled back into pixels before any distance between them means anything.
    pts = {index: (landmarks[index].x * width, landmarks[index].y * height)
           for _, index in FEATURE_LANDMARKS}

    left_shoulder, right_shoulder = pts[L_SHOULDER], pts[R_SHOULDER]
    shoulder_width = math.dist(left_shoulder, right_shoulder)
    # Everything below divides by this, so a bad value doesn't merely add noise,
    # it amplifies it. A driver standing side-on or far away has near-overlapping
    # shoulders, which would turn a level arm into full speed.
    if shoulder_width < MIN_SHOULDER_FRAC * width:
        return None

    midpoint = ((left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2)
    return pts, midpoint, shoulder_width


def features_from_landmarks(landmarks, width, height):
    """Turn 33 raw landmarks into 18 numbers a classifier can learn from.

    Raw pixel coordinates would teach the model where you stood, not what you
    did. So every point is re-expressed relative to the midpoint of your
    shoulders and divided by your shoulder width, which drops both where you
    are in frame and how close you are to the camera. What survives is the
    shape of your pose, which is the only part that should matter.

    Returns a flat list of 18 floats, or None if the pose is unusable.
    """
    geometry = pose_geometry(landmarks, width, height)
    if geometry is None:
        return None
    pts, (cx, cy), shoulder_width = geometry

    features = []
    for _, index in FEATURE_LANDMARKS:
        x, y = pts[index]
        features.append((x - cx) / shoulder_width)
        features.append((y - cy) / shoulder_width)
    return features


def draw_upper_body(frame, pts):
    """Sketch the arms and shoulders that the feature vector is built from."""
    import cv2
    if pts is None:
        return
    for shoulder, elbow, wrist in ((L_SHOULDER, L_ELBOW, L_WRIST),
                                   (R_SHOULDER, R_ELBOW, R_WRIST)):
        chain = [tuple(map(int, pts[i])) for i in (shoulder, elbow, wrist)]
        cv2.line(frame, chain[0], chain[1], (0, 220, 255), 3)
        cv2.line(frame, chain[1], chain[2], (0, 220, 255), 3)
        for point in chain:
            cv2.circle(frame, point, 6, (255, 255, 255), -1)
    cv2.line(frame, tuple(map(int, pts[L_SHOULDER])),
             tuple(map(int, pts[R_SHOULDER])), (0, 220, 255), 2)
