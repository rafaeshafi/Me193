"""VisionWorker: camera frames -> lag-stamped, filtered paddle poses (+ tag events), via fakes."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong import vision
from pingpong.clock import FakeClock
from pingpong.pose import PoseLock
from pingpong.tags import Tag

MS = 1_000_000
W, H = 640, 360


def body(sh=(0.6, 0.4), wrist=(0.30, 0.40), vis=0.95):
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    lm[11] = SimpleNamespace(x=sh[0], y=0.4, visibility=vis)
    lm[12] = SimpleNamespace(x=sh[1], y=0.4, visibility=vis)
    lm[16] = SimpleNamespace(x=wrist[0], y=wrist[1], visibility=vis)
    return lm


class FakeCapture:
    def __init__(self, ok=True, n=None):
        self.ok, self.n, self.reads, self.released = ok, n, 0, False

    def isOpened(self):
        return self.ok

    def read(self):
        self.reads += 1
        if not self.ok or (self.n is not None and self.reads > self.n):
            return False, None
        return True, np.full((H, W, 3), 90, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeLandmarker:
    def __init__(self, clock=None, infer_ms=0.0, scripted=None):
        self.clock, self.infer_ms, self.scripted, self.calls = clock, infer_ms, scripted, []

    def detect_for_video(self, image, ts_ms):
        self.calls.append(ts_ms)
        if self.clock is not None and self.infer_ms:
            self.clock.advance_s(self.infer_ms / 1000.0)
        lm = self.scripted(len(self.calls) - 1) if self.scripted else body()
        return SimpleNamespace(pose_landmarks=[lm] if lm is not None else [])


class FakeTagDetector:
    def __init__(self, ids=()):
        self.ids, self.calls = ids, 0

    def detect(self, frame):
        self.calls += 1
        return [Tag(i, ((0, 0),) * 4, (0.0, 0.0), 0.0) for i in self.ids]


def make(clock=None, **kw):
    clock = clock or FakeClock(start_ns=1_000_000_000)
    cap = kw.pop("capture", FakeCapture())
    lmk = kw.pop("landmarker", FakeLandmarker(clock))
    worker = vision.VisionWorker(cap, lmk, clock=clock, to_image=lambda f: f, lag_s=kw.pop("lag_s", 0.10), **kw)
    return worker, clock, cap, lmk


def tick(worker, clock, dt=1 / 30):
    clock.advance_s(dt)
    return worker.step()


def test_samples_are_stamped_at_capture_time_minus_the_measured_camera_lag():
    worker, clock, *_ = make(lag_s=0.10)
    tick(worker, clock)
    sample = worker.snapshot()[-1]
    assert sample.t_scene_ns == clock.now_ns() - 100 * MS
    assert sample.hand == "right" and sample.conf == pytest.approx(0.95)
    assert sample.u == pytest.approx(1.0)


def test_the_snapshot_is_an_immutable_tuple_that_later_frames_do_not_mutate():
    worker, clock, *_ = make()
    for _ in range(3):
        tick(worker, clock)
    first = worker.snapshot()
    assert isinstance(first, tuple) and len(first) == 3
    for _ in range(3):
        tick(worker, clock)
    assert len(first) == 3 and len(worker.snapshot()) == 6


def test_history_is_bounded():
    worker, clock, *_ = make(history=8)
    for _ in range(30):
        tick(worker, clock)
    assert len(worker.snapshot()) == 8


def test_frames_without_a_person_add_no_samples_and_are_counted():
    lmk = FakeLandmarker(scripted=lambda i: None)
    worker, clock, *_ = make(landmarker=lmk)
    for _ in range(5):
        tick(worker, clock)
    assert worker.snapshot() == () and worker.n_no_pose == 5


def test_the_pose_lock_keeps_a_spectator_from_moving_the_paddle():
    lock = PoseLock()
    lock.calibrate(0.20 * W / W)                      # calibrated shoulder width: 20% of the frame
    spectator = lambda i: body(sh=(0.675, 0.325))      # 35% of the frame: much closer to the camera  # noqa: E731
    worker, clock, *_ = make(landmarker=FakeLandmarker(scripted=spectator), lock=lock)
    for _ in range(4):
        tick(worker, clock)
    assert worker.snapshot() == () and worker.n_locked_out == 4


def test_a_player_turned_towards_the_side_is_still_tracked_in_the_calibrated_unit():
    lock = PoseLock()
    lock.calibrate(0.20)
    turned = lambda i: body(sh=(0.56, 0.44), wrist=(0.30, 0.40))        # shoulders 0.12 wide: 60% of the calibrated 0.20  # noqa: E731
    worker, clock, *_ = make(landmarker=FakeLandmarker(scripted=turned), lock=lock)
    tick(worker, clock)
    assert worker.n_locked_out == 0
    assert worker.snapshot()[-1].u == pytest.approx((0.5 - 0.30) / 0.20)       # not (0.5 - 0.30) / 0.12


def test_the_hand_path_is_smoothed_by_the_one_euro_filter():
    import random
    rng = random.Random(2)
    jitter = lambda i: body(wrist=(0.30 + rng.gauss(0, 0.004), 0.40))        # noqa: E731
    worker, clock, *_ = make(landmarker=FakeLandmarker(scripted=jitter))
    for _ in range(120):
        tick(worker, clock)
    us = [p.u for p in worker.snapshot()[30:]]
    assert max(us) - min(us) < 0.12                    # raw jitter would span ~0.3 shoulder widths


def test_the_number_of_frames_is_reported_so_the_locked_out_ones_can_be_a_share():
    worker, clock, *_ = make()
    for _ in range(7):
        tick(worker, clock)
    assert worker.stats()["frames"] == 7


def test_inference_latency_and_fps_are_reported():
    clock = FakeClock(start_ns=1_000_000_000)
    worker, _, *_ = make(clock=clock, landmarker=FakeLandmarker(clock, infer_ms=15.0))
    for _ in range(40):
        tick(worker, clock)
    stats = worker.stats()
    assert stats["infer_p50_ms"] == pytest.approx(15.0, abs=0.5)
    assert 20 < stats["fps"] < 40


def test_mediapipe_timestamps_strictly_increase_even_if_the_clock_stalls():
    worker, clock, _, lmk = make()
    worker.step()
    worker.step()                                      # no clock advance between frames
    assert lmk.calls[1] > lmk.calls[0]


def test_tags_are_looked_for_only_in_the_lobby_and_at_most_ten_times_a_second():
    det = FakeTagDetector(ids=())
    phase = {"v": "RALLY"}
    worker, clock, *_ = make(tag_detector=det, phase_fn=lambda: phase["v"])
    for _ in range(30):
        tick(worker, clock)
    assert det.calls == 0
    phase["v"] = "LOBBY"
    for _ in range(30):                                # one second at 30 fps
        tick(worker, clock)
    assert 8 <= det.calls <= 11


def test_a_start_card_held_for_two_seconds_becomes_a_start_event_and_a_short_show_does_not():
    det = FakeTagDetector(ids=(0,))
    worker, clock, *_ = make(tag_detector=det, phase_fn=lambda: "LOBBY")
    for _ in range(45):                                # 1.5 s of the card: not yet
        tick(worker, clock)
    assert worker.poll_tags() == []
    for _ in range(30):                                # 2.5 s in all
        tick(worker, clock)
    assert [e.role for e in worker.poll_tags()] == ["START"]
    det2 = FakeTagDetector(ids=(0,))
    worker2, clock2, *_ = make(tag_detector=det2, phase_fn=lambda: "LOBBY")
    for _ in range(45):                                # 1.5 s, then taken away
        tick(worker2, clock2)
    det2.ids = ()
    for _ in range(40):
        tick(worker2, clock2)
    assert worker2.poll_tags() == []


def test_the_worker_tells_the_screen_how_far_a_card_has_been_held():
    det = FakeTagDetector(ids=(2,))
    worker, clock, *_ = make(tag_detector=det, phase_fn=lambda: "LOBBY")
    assert worker.tag_hold(clock.now_ns()) is None
    for _ in range(30):                                # one second of card 2
        tick(worker, clock)
    card, progress = worker.tag_hold(clock.now_ns())
    assert card == 2 and 0.35 < progress < 0.65
    det.ids = ()
    for _ in range(30):
        tick(worker, clock)
    assert worker.tag_hold(clock.now_ns()) is None


def test_a_failed_read_is_not_an_error_and_returns_false():
    worker, clock, *_ = make(capture=FakeCapture(ok=False))
    assert tick(worker, clock) is False


def test_the_latest_frame_is_available_for_display():
    worker, clock, *_ = make()
    assert worker.latest_frame() is None
    tick(worker, clock)
    assert worker.latest_frame().shape == (H, W, 3)


def test_the_picture_for_the_screen_is_the_size_the_pose_model_sees_not_the_cameras_full_size():
    class BigCapture(FakeCapture):
        def read(self):
            self.reads += 1
            return True, np.full((1080, 1920, 3), 90, dtype=np.uint8)

    worker, clock, *_ = make(capture=BigCapture())
    assert worker.latest_picture() is None
    tick(worker, clock)
    assert worker.latest_frame().shape == (1080, 1920, 3)
    assert worker.latest_picture().shape == (vision.POSE_WIDTH * 9 // 16, vision.POSE_WIDTH, 3)
    small, clock2 = make()[0], None
    tick(small, small.clock)
    assert small.latest_picture().shape == (H, W, 3)                    # a frame that is small already is not changed


def test_the_thread_runs_and_stops_cleanly_and_releases_the_camera():
    cap = FakeCapture()
    worker = vision.VisionWorker(cap, FakeLandmarker(), to_image=lambda f: f, lag_s=0.1)
    worker.start()
    deadline = time.monotonic() + 3.0
    while not worker.snapshot() and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()
    assert worker.snapshot() and cap.released is True
    assert not worker._thread.is_alive()


def test_pose_age_reports_how_stale_the_last_hand_reading_is():
    worker, clock, *_ = make()
    assert worker.pose_age_s(clock.now_ns()) is None
    tick(worker, clock)
    assert worker.pose_age_s(clock.now_ns()) == pytest.approx(0.0, abs=1e-9)
    clock.advance_s(2.5)
    assert worker.pose_age_s(clock.now_ns()) == pytest.approx(2.5)


def test_the_latest_shoulder_width_is_exposed_for_calibration():
    worker, clock, *_ = make()
    assert worker.last_shoulder_w is None
    tick(worker, clock)
    assert worker.last_shoulder_w == pytest.approx(0.20)               # shoulders at x = 0.4 and 0.6
    nobody = FakeLandmarker(scripted=lambda i: None)
    worker2, clock2, *_ = make(landmarker=nobody)
    tick(worker2, clock2)
    assert worker2.last_shoulder_w is None


def test_the_latest_landmarks_are_exposed_for_the_bench_preview():
    worker, clock, *_ = make()
    assert worker.last_landmarks is None
    tick(worker, clock)
    assert worker.last_landmarks[11].x == pytest.approx(0.6)            # the body of the newest frame
    nobody = FakeLandmarker(scripted=lambda i: None)
    worker2, clock2, *_ = make(landmarker=nobody)
    tick(worker2, clock2)
    assert worker2.last_landmarks is None


def test_two_workers_sharing_one_landmarker_never_send_a_timestamp_that_goes_backwards():
    # MediaPipe's video mode rejects any timestamp at or below the last one it saw.  The worker's
    # timestamps come from the clock itself, not from its own first frame, so a second worker (or
    # a restarted one) on the same landmarker keeps counting up.
    clock = FakeClock(start_ns=1_000_000_000)
    lmk = FakeLandmarker(clock)
    first, _, _, _ = make(clock=clock, landmarker=lmk)
    for _ in range(5):
        tick(first, clock)
    second, _, _, _ = make(clock=clock, landmarker=lmk)
    for _ in range(5):
        tick(second, clock)
    assert lmk.calls == sorted(set(lmk.calls)) and len(lmk.calls) == 10


def test_a_repeating_error_is_reported_once_with_a_count_not_on_every_frame():
    class Failing:
        def detect_for_video(self, image, ts_ms):
            raise RuntimeError("model blew up")

    printed = []
    worker = vision.VisionWorker(FakeCapture(), Failing(), to_image=lambda f: f, lag_s=0.1, log=printed.append)
    worker.start()
    deadline = time.monotonic() + 3.0
    while worker.n_errors < 60 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()
    assert worker.n_errors >= 60 and 1 <= len(printed) <= 3
    assert "model blew up" in printed[0]


# --- the body tracker instead of the width lock ---------------------------------------------------------------------------
from pingpong import body as body_mod                                      # noqa: E402


def tracked(**kw):
    return make(body=body_mod.BodyTracker(ref=kw.pop("ref", 0.20)), **kw)


def test_a_player_nearer_than_at_calibration_keeps_every_frame_that_the_old_lock_refused():
    near = lambda i: body(sh=(0.635, 0.365))                                # 0.27 of the frame: 1.35 x the calibrated 0.20  # noqa: E731
    old_lock = PoseLock()
    old_lock.calibrate(0.20)
    old, clock, *_ = make(landmarker=FakeLandmarker(scripted=near), lock=old_lock)
    new, clock2, *_ = tracked(landmarker=FakeLandmarker(scripted=near))
    for _ in range(30):
        tick(old, clock)
        tick(new, clock2)
    assert len(old.snapshot()) == 0 and old.n_locked_out == 30               # what the first live games lost
    assert len(new.snapshot()) == 30 and new.n_locked_out == 0


def test_the_hand_is_measured_in_the_players_shoulder_widths_at_the_players_distance():
    near = lambda i: body(sh=(0.635, 0.365), wrist=(0.4, 0.40))             # shoulders 0.27 wide, the hand 0.1 from the midpoint
    worker, clock, *_ = tracked(landmarker=FakeLandmarker(scripted=near), ref=0.27)
    for _ in range(40):
        tick(worker, clock)
    assert worker.snapshot()[-1].u == pytest.approx(0.1 / 0.27, abs=0.02)   # not 0.1 / 0.20: the unit follows the distance


def test_a_shoulder_hidden_for_a_few_frames_does_not_leave_a_hole_in_the_track():
    def no_shoulder():
        lm = body()
        lm[12] = SimpleNamespace(x=0.4, y=0.4, visibility=0.1)             # the hand is still seen, one shoulder is not
        return lm

    frames = [body() for _ in range(10)] + [no_shoulder() for _ in range(4)] + [body() for _ in range(5)]
    worker, clock, *_ = tracked(landmarker=FakeLandmarker(scripted=lambda i: frames[i]))
    for _ in range(len(frames)):
        tick(worker, clock)
    poses = worker.snapshot()
    assert len(poses) == len(frames)                                          # every frame has a hand reading
    held = poses[11]
    assert held.conf <= body_mod.HELD_CONF + 1e-9 and held.u == pytest.approx(poses[9].u)
    assert worker.stats()["lock"]["held"] == 4


def test_a_spectator_is_still_refused_and_the_reason_is_counted():
    spectator = lambda i: body(sh=(0.675, 0.325))                            # twice as wide as the calibrated body  # noqa: E731
    worker, clock, *_ = tracked(landmarker=FakeLandmarker(scripted=spectator))
    for _ in range(4):
        tick(worker, clock)
    assert worker.snapshot() == () and worker.n_locked_out == 4
    assert worker.stats()["lock"]["size"] == 4


def test_the_stats_say_how_the_body_was_tracked():
    worker, clock, *_ = tracked()
    for _ in range(5):
        tick(worker, clock)
    lock = worker.stats()["lock"]
    assert lock["ok"] == 5 and lock["held"] == 0 and lock["rejected"] == 0


def test_the_unfiltered_reading_is_kept_beside_the_filtered_one_for_training():
    worker, clock, *_ = tracked()
    for _ in range(3):
        tick(worker, clock)
    sample = worker.snapshot()[-1]
    assert sample.raw == pytest.approx((sample.u, sample.v), abs=1e-6)       # a still hand: nothing to filter away
