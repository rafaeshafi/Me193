"""VisionWorker: camera -> MediaPipe pose -> One-Euro -> immutable pose snapshots (+ tag events).

One thread owns the camera.  Every frame is stamped t_read when it is read; the pose sample
is stamped t_read - lag (the measured camera-vs-IMU lag), so the judge can look a sample up
at the IMU time of a swing without subtracting the lag twice.  The history is an immutable
tuple swapped by reference (no shared deque), so the game thread never sees a torn window.
AprilTags are only searched for in the lobby, at most ~10 Hz, on the same thread.
step() does exactly one frame and is what both the thread loop and the tests call.

Run the camera from Terminal.app: macOS gives camera access to YOUR terminal, not to the
Claude app.
"""

import queue
import threading
import time
from collections import deque

import cv2

import config
from pingpong import pose
from pingpong.clock import Clock
from pingpong.events import PaddlePose
from pingpong.oneeuro import OneEuro2D
from pingpong.tags import TagVoter

POSE_WIDTH = 640
TAG_WIDTH = 960
TAG_PHASES = ("LOBBY", "MATCH_OVER")


class VisionWorker:
    def __init__(self, capture, landmarker, *, clock=None, hand="right", lag_s=None, tag_detector=None,
                 voter=None, phase_fn=None, lock=None, to_image=None, history=64, tag_hz=10.0):
        self.capture, self.landmarker = capture, landmarker
        self.clock = clock or Clock()
        self.hand = hand
        self.lag_ns = round((config.CAMERA_LAG_S if lag_s is None else lag_s) * 1e9)
        self.tag_detector = tag_detector
        self.voter = voter or TagVoter()
        self.phase_fn = phase_fn or (lambda: "LOBBY")
        self.lock = lock
        self.to_image = to_image or _default_to_image
        self.history, self.tag_period_ns = history, round(1e9 / tag_hz)
        self._filter = OneEuro2D()
        self._poses = ()
        self._frame = None
        self._tags = queue.SimpleQueue()
        self._infer_ms = deque(maxlen=120)
        self._frame_times = deque(maxlen=60)
        self._last_ts_ms = -1
        self._t_start = None
        self._last_tag_ns = None
        self.n_no_pose = self.n_locked_out = 0
        self._last_pose_read_ns = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)

    # --- one frame ---------------------------------------------------------------------------
    def step(self):
        ok, frame = self.capture.read()
        if not ok:
            return False
        t_read = self.clock.now_ns()
        self._frame = frame
        self._frame_times.append(t_read)
        self._pose(frame, t_read)
        self._look_for_tags(frame, t_read)
        return True

    def _pose(self, frame, t_read):
        h, w = frame.shape[:2]
        small = frame if w <= POSE_WIDTH else cv2.resize(frame, (POSE_WIDTH, int(h * POSE_WIDTH / w)))
        sh, sw = small.shape[:2]
        t0 = self.clock.now_ns()
        result = self.landmarker.detect_for_video(self.to_image(small), self._timestamp_ms(t_read))
        self._infer_ms.append((self.clock.now_ns() - t0) / 1e6)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        if landmarks is None:
            self.n_no_pose += 1
            return
        if self.lock is not None and not self.lock.accepts(pose.shoulder_width_norm(landmarks, sw, sh)):
            self.n_locked_out += 1
            return
        uv = pose.paddle_uv(landmarks, self.hand, sw, sh)
        if uv is None:
            self.n_no_pose += 1
            return
        u, v = self._filter((uv[0], uv[1]), t_read / 1e9)
        sample = PaddlePose(t_scene_ns=t_read - self.lag_ns, u=u, v=v, conf=uv[2], hand=self.hand)
        self._poses = (self._poses + (sample,))[-self.history:]    # immutable tuple, swapped by reference
        self._last_pose_read_ns = t_read

    def _timestamp_ms(self, t_read):
        if self._t_start is None:
            self._t_start = t_read
        ts = int((t_read - self._t_start) / 1e6)
        self._last_ts_ms = max(self._last_ts_ms + 1, ts)           # MediaPipe needs strictly increasing ms
        return self._last_ts_ms

    def _look_for_tags(self, frame, t_read):
        if self.tag_detector is None or self.phase_fn() not in TAG_PHASES:
            return
        if self._last_tag_ns is not None and t_read - self._last_tag_ns < self.tag_period_ns:
            return
        self._last_tag_ns = t_read
        h, w = frame.shape[:2]
        view = frame if w <= TAG_WIDTH else cv2.resize(frame, (TAG_WIDTH, int(h * TAG_WIDTH / w)))
        ids = {t.id for t in self.tag_detector.detect(view)}
        for event in self.voter.update(t_read, ids, self.phase_fn()):
            self._tags.put_nowait(event)

    # --- reading results -----------------------------------------------------------------------------
    def snapshot(self):
        return self._poses

    def pose_age_s(self, now_ns):
        """Seconds since the last frame that yielded a hand reading (None if never)."""
        if self._last_pose_read_ns is None:
            return None
        return (now_ns - self._last_pose_read_ns) / 1e9

    def latest_frame(self):
        return self._frame

    def poll_tags(self):
        events = []
        while not self._tags.empty():
            events.append(self._tags.get_nowait())
        return events

    def stats(self):
        times = list(self._frame_times)
        fps = (len(times) - 1) / ((times[-1] - times[0]) / 1e9) if len(times) > 1 and times[-1] > times[0] else 0.0
        inferred = sorted(self._infer_ms)
        return {"fps": fps,
                "infer_p50_ms": inferred[len(inferred) // 2] if inferred else 0.0,
                "infer_p95_ms": inferred[int(len(inferred) * 0.95) - 1] if len(inferred) > 1 else 0.0,
                "no_pose": self.n_no_pose, "locked_out": self.n_locked_out}

    # --- thread --------------------------------------------------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                if not self.step():
                    time.sleep(0.01)
            except Exception as exc:                 # one bad frame must not kill the camera thread
                print(f"vision: {exc}")
                time.sleep(0.01)

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.capture.release()


def _default_to_image(frame):
    from pingpong.pose_features import to_mp_image

    return to_mp_image(frame)
