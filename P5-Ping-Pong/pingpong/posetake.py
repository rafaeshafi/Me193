"""A pose take: a short guided recording of the camera, and the offline comparison of pose models on it.

`./pp train_pose --record` asks the player to stand, slide, lift, follow a ball and swing for about 100 s while the
camera's frames are saved (a video, plus the time every frame was read).  Offline, every candidate pose model then
runs over the SAME footage through the game's own pipeline (body tracker, hand tracker, filter), and the one whose
readings are cleanest, that still keeps up with 30 frames a second, is the one the game will use for this player:
chosen on this camera, this room and this body rather than on a guess.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pingpong import posetrain, profile
from pingpong.body import BodyTracker
from pingpong.vision import VisionWorker

TAKES = "pose-takes"
VIDEO, FRAMES = "take.mp4", "frames.jsonl"
SIZE = (640, 360)
MAX_INFER_MS = 24.0           # p95: any slower and the camera loop cannot keep up with 30 frames a second
MIN_DETECTION = 0.93          # of the best candidate's: a model that loses the player more often than this is out
GLITCH = 0.25                 # shoulder widths: a reading this far from the smoothed track is a landmark flip
GLITCH_WEIGHT = 2.0
PREFER_CHEAPER = 0.9          # the dearer model must beat the cheap one's score by 10% to be chosen


@dataclass(frozen=True)
class Step:
    name: str
    seconds: float
    text: str


SCRIPT = (
    Step("still", 6, "Stand where you play, hold the hub like a paddle and keep your hand STILL"),
    Step("slide", 18, "Slide the paddle slowly left and right across the table, again and again"),
    Step("lift", 14, "Now slowly up and down, like reaching forward and pulling back"),
    Step("follow", 18, "Faster: follow an imaginary ball with the paddle, left and right"),
    Step("swing", 30, "Swing like in the game: a swing, a pause, a swing..."),
    Step("still", 8, "Hold still again"),
)


def where(t_s):
    """-> (index, step, seconds left of it) for t_s seconds into the script, or None after its end."""
    for index, step in enumerate(SCRIPT):
        if t_s < step.seconds:
            return index, step, step.seconds - t_s
        t_s -= step.seconds
    return None


def step_at(t_s):
    """The step of the script that is on at t_s seconds into the take (None after the end)."""
    found = where(t_s)
    return None if found is None else found[1]


def take_dir(player, root=None, wall=None):
    from pingpong import recorder

    return Path(root or recorder.default_root()) / TAKES / recorder.session_name(player, wall)


def list_takes(player, root=None):
    from pingpong import recorder

    base = Path(root or recorder.default_root()) / TAKES
    suffix = "-" + profile.slug(player)
    return sorted(p for p in base.glob("*" + suffix) if p.is_dir()) if base.exists() else []


# --- recording ---------------------------------------------------------------------------------------------------------------
class TakeCapture:
    """Wraps the camera: every frame the vision thread reads is also written to a video with the time it was read.

    The game never sees the difference (a frame comes back untouched); a take that cannot be written (no disk, no codec)
    stops writing and says so in `failed`, and the camera goes on.  `recording` switches the keeping on and off: the
    player takes up position first, and that part is not footage."""

    def __init__(self, capture, clock, directory, size=SIZE, fps=30.0, recording=True):
        self.capture, self.clock, self.size, self.recording = capture, clock, size, recording
        self.failed, self._writer, self._times = False, None, None
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(directory / VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self._writer.isOpened():
                raise OSError("the video writer did not open")
            self._times = (directory / FRAMES).open("w")
        except (OSError, cv2.error):
            self.failed = True

    def read(self):
        ok, frame = self.capture.read()
        if ok and self.recording and not self.failed:
            try:
                t = self.clock.now_ns()
                self._writer.write(cv2.resize(frame, self.size))      # the game's own resize: the model sees what it will see live
                self._times.write(json.dumps({"t": t}) + "\n")
            except (OSError, cv2.error):
                self.failed = True
        return ok, frame

    def release(self):
        for closer in (self._writer and self._writer.release, self._times and self._times.close):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        self.capture.release()

    def __getattr__(self, name):                                  # isOpened(), set(), get(): the camera's own
        return getattr(self.capture, name)


class VideoFrames:
    """A saved take played back as a camera: read() gives the next frame, `times` are the times they were read."""

    def __init__(self, directory):
        directory = Path(directory)
        self._video = cv2.VideoCapture(str(directory / VIDEO))
        self.times = [json.loads(line)["t"] for line in (directory / FRAMES).read_text().splitlines() if line]

    def isOpened(self):
        return self._video.isOpened()

    def read(self):
        return self._video.read()

    def release(self):
        self._video.release()


# --- offline ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    track: Optional[posetrain.Track]
    detection: float               # the share of frames with a hand reading
    noise: float                   # RMS of the readings around their own zero-phase smoothing (shoulder widths)
    glitches: float                # the share of readings further than GLITCH from it: landmark flips
    infer_ms_p95: float            # the model's time per frame on this machine


class _FrameClock:
    """The clock of an offline analysis: it reads the time at which the frame being looked at was recorded."""

    def __init__(self, start_ns=0):
        self.t_ns = start_ns

    def now_ns(self):
        return self.t_ns


class _Timed:
    def __init__(self, landmarker):
        self.landmarker, self.ms = landmarker, []

    def detect_for_video(self, image, ts_ms):
        t0 = time.perf_counter()
        result = self.landmarker.detect_for_video(image, ts_ms)
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        return result


def analyze(directory, landmarker, *, hand, ref_width, to_image=None, progress=None):
    """Run a pose model over a saved take through the game's own vision pipeline -> a Run."""
    frames = VideoFrames(directory)
    clock = _FrameClock()
    timed = _Timed(landmarker)
    vision = VisionWorker(frames, timed, clock=clock, hand=hand, lag_s=0.0, history=len(frames.times) + 8,
                          body=BodyTracker(ref=ref_width or None), to_image=to_image, log=lambda *_: None)
    try:
        for i, t_ns in enumerate(frames.times):
            clock.t_ns = t_ns                                       # the frame is read at the time it really was
            if not vision.step():
                break
            if progress is not None and i % 150 == 0:
                progress(i, len(frames.times))
    finally:
        frames.release()
    poses = vision.snapshot()
    if len(poses) < 30:
        return Run(None, len(poses) / max(1, len(frames.times)), float("inf"), 1.0, _p95(timed.ms))
    t0 = poses[0].t_scene_ns
    t = np.array([(p.t_scene_ns - t0) / 1e9 for p in poses])
    track = posetrain.Track(t=t, u=np.array([p.u for p in poses]), v=np.array([p.v for p in poses]),
                            ru=np.array([p.raw[0] for p in poses]), rv=np.array([p.raw[1] for p in poses]), name=Path(directory).name)
    residuals = []
    for raw in (track.ru, track.rv):
        _, x = posetrain.on_grid(t, raw)
        residuals.append((x - posetrain.teacher(x, sigma=1.0))[np.isfinite(x)])
    res = np.concatenate(residuals)
    flips = np.abs(res) > GLITCH
    return Run(track, len(poses) / max(1, len(frames.times)), float(np.sqrt(np.mean(res[~flips] ** 2))) if (~flips).any() else float("inf"),
               float(flips.mean()), _p95(timed.ms))


def _p95(values):
    return float(sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]) if values else 0.0


def choose(runs):
    """-> (name, why): the cleanest model that keeps up with the camera and rarely loses the player; ties go to the cheaper."""
    best_detection = max(r.detection for r in runs.values())
    fast = {n: r for n, r in runs.items() if r.infer_ms_p95 <= MAX_INFER_MS}
    steady = {n: r for n, r in fast.items() if r.detection >= MIN_DETECTION * best_detection}
    cheapest = "lite" if "lite" in runs else next(iter(runs))
    if not steady:
        return cheapest, "no model both keeps up with 30 frames a second and holds the player: the lighter one stays"
    score = lambda r: r.noise + GLITCH_WEIGHT * r.glitches            # noqa: E731
    best = min(steady, key=lambda n: score(steady[n]))
    slow = [n for n in runs if n not in fast]
    note = "".join(f"; {n} is too slow ({runs[n].infer_ms_p95:.0f} ms a frame, more than {MAX_INFER_MS:.0f})" for n in slow)
    if cheapest in steady and best != cheapest and score(steady[best]) > PREFER_CHEAPER * score(steady[cheapest]):
        return cheapest, f"{best} is not clearly cleaner than {cheapest}, which is cheaper" + note
    if best == cheapest and len(steady) == 1:
        return best, f"{best} is the only one that is fast enough and steady" + note
    other = next((n for n in steady if n != best), None)
    if other is None:
        return best, f"{best} is the only candidate" + note
    return best, (f"{best} has less noise ({steady[best].noise:.3f} against {steady[other].noise:.3f} shoulder widths) and "
                  f"fewer flips ({steady[best].glitches:.1%} against {steady[other].glitches:.1%})") + note
