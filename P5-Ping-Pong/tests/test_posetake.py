"""A pose take: a short guided recording of the camera (video + the time of every frame), and the offline comparison of
pose models on it.  The same footage goes through every candidate, so the choice between them is made on the player's
own camera, room and body rather than on a guess."""

import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong import posemodel, posetake
from pingpong.clock import FakeClock

S = 1_000_000_000
SIZE = (64, 36)


class Camera:
    """A fake camera: frames of a flat grey, n of them."""

    def __init__(self, n=60):
        self.n, self.reads, self.released = n, 0, False

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if self.reads > self.n:
            return False, None
        return True, np.full((72, 128, 3), 90, dtype=np.uint8)

    def release(self):
        self.released = True


def test_every_frame_the_camera_gives_is_kept_with_the_time_it_was_read(tmp_path):
    clock = FakeClock(start_ns=5 * S)
    cam = posetake.TakeCapture(Camera(20), clock, tmp_path / "take", size=SIZE)
    for k in range(20):
        clock.advance_s(1 / 30)
        ok, frame = cam.read()
        assert ok and frame.shape[:2] == (72, 128)                       # the game sees the frame untouched
    assert cam.read() == (False, None)
    cam.release()
    times = [json.loads(line)["t"] for line in (tmp_path / "take" / "frames.jsonl").read_text().splitlines()]
    assert len(times) == 20 and times[0] == 5 * S + round(S / 30) and times == sorted(times)
    frames = posetake.VideoFrames(tmp_path / "take")
    got = []
    while True:
        ok, frame = frames.read()
        if not ok:
            break
        got.append(frame)
    assert len(got) == 20 and got[0].shape[:2] == SIZE[::-1]
    assert frames.times == times


def test_frames_are_kept_only_while_the_take_is_recording_and_the_game_sees_all_of_them(tmp_path):
    clock = FakeClock(start_ns=5 * S)
    cam = posetake.TakeCapture(Camera(30), clock, tmp_path / "take", size=SIZE, recording=False)
    for k in range(30):
        cam.recording = 10 <= k < 20                                         # the player takes up position, then the script runs
        clock.advance_s(1 / 30)
        assert cam.read()[0] is True
    cam.release()
    times = (tmp_path / "take" / "frames.jsonl").read_text().splitlines()
    assert len(times) == 10 and json.loads(times[0])["t"] == pytest.approx(5 * S + 11 * S / 30, abs=1000)       # the 11th frame
    frames = posetake.VideoFrames(tmp_path / "take")
    assert sum(1 for _ in iter(lambda: frames.read()[0], False)) == 10 and len(frames.times) == 10


def test_a_take_that_cannot_write_its_video_still_lets_the_camera_through(tmp_path):
    bad = tmp_path / "no" / "such" / "place"
    (tmp_path / "no").write_text("a file where a folder should be")
    cam = posetake.TakeCapture(Camera(3), FakeClock(), bad, size=SIZE)
    assert cam.read()[0] is True and cam.failed
    cam.release()


def test_takes_are_listed_by_player_newest_last(tmp_path):
    for name in ("20261007-100000-rafae", "20261007-120000-rafae", "20261007-110000-maya"):
        d = tmp_path / posetake.TAKES / name
        d.mkdir(parents=True)
        (d / "take.mp4").write_bytes(b"x")
    found = posetake.list_takes("rafae", root=tmp_path)
    assert [p.name for p in found] == ["20261007-100000-rafae", "20261007-120000-rafae"]


# --- the guided script ---------------------------------------------------------------------------------------------------
def test_the_script_walks_through_still_slow_fast_and_swings_and_says_where_it_is():
    steps = posetake.SCRIPT
    names = [s.name for s in steps]
    assert names[0] == "still" and "swing" in names and names[-1] == "still"
    total = sum(s.seconds for s in steps)
    assert 80 <= total <= 130                                              # long enough to train on, short enough to do
    at = posetake.step_at(0.0)
    assert at.name == "still"
    assert posetake.step_at(total - 0.1).name == names[-1]
    assert posetake.step_at(total + 5) is None


def test_where_in_the_script_says_which_step_is_on_and_how_long_is_left_of_it():
    steps = posetake.SCRIPT
    assert posetake.where(0.0) == (0, steps[0], steps[0].seconds)
    index, step, left = posetake.where(steps[0].seconds + 1.5)
    assert (index, step) == (1, steps[1]) and left == pytest.approx(steps[1].seconds - 1.5)
    total = sum(s.seconds for s in steps)
    assert posetake.where(total - 0.25)[0] == len(steps) - 1 and posetake.where(total + 1) is None


# --- running a model over a take ----------------------------------------------------------------------------------------------
def body(u, v, noise, rnd, glitch=False):
    """Landmarks whose hand sits at (u, v) shoulder widths of a 0.2-wide body, with a candidate's noise."""
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    lm[11] = SimpleNamespace(x=0.6, y=0.4, visibility=0.95)
    lm[12] = SimpleNamespace(x=0.4, y=0.4, visibility=0.95)
    aspect = 360 / 640
    gu, gv = rnd.gauss(0, noise), rnd.gauss(0, noise)
    if glitch:
        gu += 1.5
    lm[16] = SimpleNamespace(x=0.5 - (u + gu) * 0.2, y=0.4 - (v + gv) * 0.2 / aspect, visibility=0.95)
    return lm


class Candidate:
    def __init__(self, noise, infer_ms=0.0, glitch_every=0, seed=1):
        self.noise, self.infer_ms, self.glitch_every = noise, infer_ms, glitch_every
        self.rnd, self.calls = random.Random(seed), 0

    def detect_for_video(self, image, ts_ms):
        i = self.calls
        self.calls += 1
        t = i / 30.0
        u, v = 0.8 * math.sin(2 * math.pi * 0.8 * t), 0.3 * math.sin(2 * math.pi * 0.4 * t)
        return SimpleNamespace(pose_landmarks=[body(u, v, self.noise, self.rnd, glitch=bool(self.glitch_every) and i % self.glitch_every == 7)])


def make_take(tmp_path, n=450):
    clock = FakeClock(start_ns=3 * S)
    cam = posetake.TakeCapture(Camera(n), clock, tmp_path / "take", size=SIZE)
    for _ in range(n):
        clock.advance_s(1 / 30)
        cam.read()
    cam.release()
    return tmp_path / "take"


def test_a_candidate_run_over_a_take_gives_its_raw_and_filtered_track_and_its_numbers(tmp_path):
    take = make_take(tmp_path)
    run = posetake.analyze(take, Candidate(noise=0.01), hand="right", ref_width=0.2)
    assert len(run.track.t) > 420 and run.track.ru is not None and run.detection > 0.97
    assert run.track.t[0] == pytest.approx(0.0, abs=0.05)
    assert run.noise < 0.05 and run.glitches < 0.01
    assert run.infer_ms_p95 >= 0.0


def test_the_comparison_prefers_the_model_whose_readings_are_cleaner_and_names_why(tmp_path):
    take = make_take(tmp_path)
    runs = {"lite": posetake.analyze(take, Candidate(noise=0.06, glitch_every=40), hand="right", ref_width=0.2),
            "full": posetake.analyze(take, Candidate(noise=0.015), hand="right", ref_width=0.2)}
    name, why = posetake.choose(runs)
    assert name == "full" and "noise" in why
    assert runs["full"].noise < runs["lite"].noise and runs["full"].glitches < runs["lite"].glitches


def test_a_model_too_slow_for_thirty_frames_a_second_is_not_chosen_however_clean():
    fast = posetake.Run(track=None, detection=0.99, noise=0.05, glitches=0.02, infer_ms_p95=11.0)
    slow = posetake.Run(track=None, detection=0.99, noise=0.01, glitches=0.0, infer_ms_p95=posetake.MAX_INFER_MS + 10)
    name, why = posetake.choose({"lite": fast, "full": slow})
    assert name == "lite" and "slow" in why


def test_a_model_that_loses_the_player_often_is_not_chosen_and_a_tie_goes_to_the_cheaper_one():
    steady = posetake.Run(track=None, detection=0.98, noise=0.03, glitches=0.0, infer_ms_p95=11.0)
    flaky = posetake.Run(track=None, detection=0.80, noise=0.01, glitches=0.0, infer_ms_p95=20.0)
    assert posetake.choose({"lite": steady, "full": flaky})[0] == "lite"
    same = posetake.Run(track=None, detection=0.98, noise=0.03, glitches=0.0, infer_ms_p95=20.0)
    assert posetake.choose({"lite": steady, "full": same})[0] == "lite"


def test_the_noise_of_a_run_is_what_is_left_after_a_zero_phase_smoothing_of_its_own_readings(tmp_path):
    take = make_take(tmp_path)
    clean = posetake.analyze(take, Candidate(noise=0.0), hand="right", ref_width=0.2)
    noisy = posetake.analyze(take, Candidate(noise=0.05), hand="right", ref_width=0.2)
    assert clean.noise < 0.01 and 0.025 < noisy.noise < 0.09
