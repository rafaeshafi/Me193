"""./pp train_pose: a guided take, the comparison of pose models on it, the filter tuned, the hand predictor trained, and
the player's pose model saved.  Exercised here on fake hardware and synthetic hands."""

import math

import pytest

from pingpong import posemodel, posetake, posetrain
from pingpong.sources_fake import FakeEnv, FakeLandmarker
from test_posetake import Candidate
from test_posetrain import filtered, raw_track, write_session
from tools import train_pose

S = 1_000_000_000
SHORT = (posetake.Step("still", 3, "hold still"), posetake.Step("slide", 9, "slide"),
         posetake.Step("swing", 9, "swing"), posetake.Step("still", 3, "hold still"))              # 24 s


@pytest.fixture(autouse=True)
def short_script(monkeypatch):
    monkeypatch.setattr(posetake, "SCRIPT", SHORT)


def hand_at(t):
    return 0.8 * math.sin(2 * math.pi * 0.7 * t), 0.3 * math.sin(2 * math.pi * 0.35 * t)


def recording_env(arrive_s=0.0, camera="ok"):
    """A camera in front of a player who walks into position after arrive_s and then follows any script."""
    env = FakeEnv(camera=camera)
    t0 = env.clock.now_ns()
    env.make_landmarker = lambda model="lite": FakeLandmarker(
        lambda t_ns: hand_at((t_ns - t0) / S) if (t_ns - t0) / S >= arrive_s else None, env.clock, full_body=True)
    return env


def record(env, tmp_path, wait_key=lambda ms: 255, shown=None, out=None, *extra):
    args = train_pose.parse_args(["--player", "rafae", "--record", *extra])
    return train_pose.record_take(env, args, "right", None, root=tmp_path, show=(shown if shown is not None else []).append,
                                  wait_key=wait_key, out=out or (lambda *_: None), width=160)


def args_for(*extra):
    return train_pose.parse_args(["--player", "rafae", *extra])


def sessions(root, seeds=(1, 2, 3), with_raw=True, seconds=40.0):
    for s in seeds:
        write_session(root, f"2026100{s}-100000-rafae", filtered(raw_track(s, noise=0.04, glitches=3, seconds=seconds)),
                      with_raw=with_raw)


def train(env, tmp_path, *extra, lines=None):
    lines = [] if lines is None else lines
    code = train_pose.train(env, args_for(*extra), player_root=tmp_path / "players", record_root=tmp_path / "rec",
                            out=lines.append)
    return code, "\n".join(lines)


# --- recording a take ---------------------------------------------------------------------------------------------------------
def test_a_take_waits_until_the_player_stands_in_position_and_keeps_only_the_script(tmp_path):
    env = recording_env(arrive_s=4.0)
    t0, shown = env.clock.now_ns(), []
    take = record(env, tmp_path / "rec", shown=shown)
    assert take is not None and take.parent == tmp_path / "rec" / posetake.TAKES and take.name.endswith("-rafae")
    frames = posetake.VideoFrames(take)
    total = sum(s.seconds for s in posetake.SCRIPT)
    assert total - 0.5 <= (frames.times[-1] - frames.times[0]) / S <= total + 0.5           # the script, not the waiting
    assert (frames.times[0] - t0) / S >= 4.0 + train_pose.POSITION_HOLD_S - 0.3             # it started once the player was there
    assert sum(1 for _ in iter(lambda: frames.read()[0], False)) == len(frames.times)
    assert shown and shown[0].shape[1] == 160                                                # the player watched it in a window


def test_q_cancels_the_take_and_leaves_no_half_take_behind(tmp_path):
    keys = iter([255] * 400)
    take = record(recording_env(), tmp_path / "rec", wait_key=lambda ms: next(keys, ord("q")))
    assert take is None
    assert list((tmp_path / "rec" / posetake.TAKES).glob("*")) == []


def test_enter_in_the_window_starts_the_script_without_waiting_for_the_hold(tmp_path):
    env = recording_env()
    t0 = env.clock.now_ns()
    calls = {"n": 0}

    def key(ms):
        calls["n"] += 1
        return 13 if calls["n"] == 5 else 255

    take = record(env, tmp_path / "rec", wait_key=key)
    frames = posetake.VideoFrames(take)
    assert (frames.times[0] - t0) / S < train_pose.POSITION_HOLD_S - 0.5


def test_a_camera_that_will_not_open_is_said_plainly_and_nothing_is_recorded(tmp_path):
    lines = []
    assert record(recording_env(camera="closed"), tmp_path / "rec", out=lines.append) is None
    assert any("camera" in line for line in lines) and not (tmp_path / "rec" / posetake.TAKES).exists()


def test_the_script_is_shown_step_by_step_with_what_to_do_and_the_time_left(tmp_path, monkeypatch):
    texts = []
    real_render = train_pose.render

    def spy(frame, landmarks, caption, headline, progress, width):
        texts.append((caption, headline))
        return real_render(frame, landmarks, caption, headline, progress, width)

    monkeypatch.setattr(train_pose, "render", spy)
    record(recording_env(), tmp_path / "rec")
    captions = {c for c, _ in texts}
    assert {"hold still", "slide", "swing"} <= captions
    heads = [h for _, h in texts if "STEP" in h]
    assert any(h.startswith("STEP 2/4") for h in heads) and any("s left" in h for h in heads)


# --- training ---------------------------------------------------------------------------------------------------------------------
def test_nothing_recorded_yet_says_how_to_start_and_trains_nothing(tmp_path):
    code, text = train(FakeEnv(), tmp_path)
    assert code == 2 and "--record" in text
    assert posemodel.load_for("rafae", root=tmp_path / "players") is None


def test_training_on_recorded_sessions_tunes_the_filter_and_saves_the_players_model(tmp_path):
    sessions(tmp_path / "rec")
    code, text = train(FakeEnv(), tmp_path)
    assert code == 0
    model = posemodel.load_for("rafae", root=tmp_path / "players")
    assert model is not None and model.landmarker == "lite"                  # no take, so nothing to compare the pose models on
    assert model.filter != posemodel.FilterParams()                           # the settings were tuned to the player's readings
    assert model.meta["sessions"] == 3 and model.meta["filter"]["rmse_tuned"] < model.meta["filter"]["rmse_default"]
    assert "filter" in text.lower() and "saved" in text.lower() and "./pp play --player rafae" in text


def test_a_dry_run_shows_the_result_and_saves_nothing(tmp_path):
    sessions(tmp_path / "rec")
    code, text = train(FakeEnv(), tmp_path, "--dry-run")
    assert code == 0 and "not saved" in text
    assert posemodel.load_for("rafae", root=tmp_path / "players") is None


def test_the_hand_predictor_ships_only_when_it_pays_and_the_report_says_why(tmp_path, monkeypatch):
    sessions(tmp_path / "rec")
    monkeypatch.setattr(posetrain, "worth_shipping", lambda cv: (False, "test says no"))
    code, text = train(FakeEnv(), tmp_path)
    assert posemodel.load_for("rafae", root=tmp_path / "players").predictor is None
    assert "not used" in text and "test says no" in text
    monkeypatch.setattr(posetrain, "worth_shipping", lambda cv: (True, "test says yes"))
    code, text = train(FakeEnv(), tmp_path)
    model = posemodel.load_for("rafae", root=tmp_path / "players")
    assert model.predictor is not None and "test says yes" in text and model.meta["predictor"]["shipped"] is True


def test_sessions_recorded_before_raw_readings_were_kept_are_left_out_once_the_filter_is_tuned(tmp_path):
    sessions(tmp_path / "rec", seeds=(1, 2))
    write_session(tmp_path / "rec", "20261009-100000-rafae", filtered(raw_track(9, noise=0.04, glitches=3, seconds=40.0)),
                  with_raw=False)
    code, text = train(FakeEnv(), tmp_path)
    assert code == 0 and "left out" in text and "1 older session" in text
    assert posemodel.load_for("rafae", root=tmp_path / "players").meta["sessions"] == 3


def test_without_any_raw_readings_the_filter_stays_as_it_is_and_the_predictor_still_trains(tmp_path):
    sessions(tmp_path / "rec", with_raw=False)
    code, text = train(FakeEnv(), tmp_path)
    assert code == 0
    model = posemodel.load_for("rafae", root=tmp_path / "players")
    assert model.filter == posemodel.FilterParams() and model.meta["filter"] is None
    assert "no unfiltered readings" in text


# --- the take decides which pose model runs -----------------------------------------------------------------------------------------
def test_the_take_decides_which_pose_model_runs_and_its_track_joins_the_training(tmp_path):
    take = record(recording_env(), tmp_path / "rec")
    assert take is not None
    env = FakeEnv()
    env.make_landmarker = lambda model="lite": Candidate(noise=0.06, glitch_every=40) if model == "lite" else Candidate(noise=0.015)
    code, text = train(env, tmp_path)
    assert code == 0
    model = posemodel.load_for("rafae", root=tmp_path / "players")
    assert model.landmarker == "full" and model.meta["take"] == take.name
    assert set(model.meta["landmarkers"]) == {"lite", "full"} and model.meta["landmarkers"]["full"]["noise"] < model.meta["landmarkers"]["lite"]["noise"]
    assert "full" in text and "noise" in text


def test_a_pose_model_that_cannot_start_is_reported_and_the_other_one_still_runs(tmp_path):
    record(recording_env(), tmp_path / "rec")
    env = FakeEnv()

    def factory(model="lite"):
        if model == "full":
            raise RuntimeError("the full pose model is missing")
        return Candidate(noise=0.02)

    env.make_landmarker = factory
    code, text = train(env, tmp_path)
    assert code == 0 and "the full pose model is missing" in text
    assert posemodel.load_for("rafae", root=tmp_path / "players").landmarker == "lite"


def test_the_hand_and_the_shoulder_width_come_from_the_players_calibration(tmp_path):
    from pingpong import profile
    from pingpong.calibration import SwingCalibration
    from pingpong.paddle import ReachBox

    profile.save("rafae", profile.Calibration(swing=SwingCalibration((0.0, 1.0, 0.0), 280.0, 1100.0),
                                              box=ReachBox(-1.5, 1.5, -0.9, 0.7), shoulder_w=0.2, hand="left"),
                 root=tmp_path / "players")
    assert train_pose.player_setup("rafae", None, tmp_path / "players") == ("left", 0.2)
    assert train_pose.player_setup("rafae", "right", tmp_path / "players") == ("right", 0.2)       # --hand wins
    assert train_pose.player_setup("newbie", None, tmp_path / "players") == ("right", None)


def test_the_selftest_runs_the_whole_flow_on_fake_hardware():
    assert train_pose.main(["--selftest"]) == 0
