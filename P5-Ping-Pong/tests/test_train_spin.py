"""tools/train_spin.py: guided collection of flat / top / back swings -> a per-player spin model."""

import numpy as np
import pytest

from pingpong import fakerig, profile, spin
from pingpong.sources_fake import FakeEnv
from tools import train_spin as tool

S = 1_000_000_000


PER_CLASS = 8


def make(tmp_path, *, scramble=False, calibrate=True, hub_found=True, seed=3):
    env = FakeEnv(hz=66.0, hub_found=hub_found)
    t0 = env.clock.now_ns()
    labels = [c for _ in range(PER_CLASS) for c in spin.CLASSES]
    script = fakerig.SpinScript(labels, seed=seed, scramble=scramble)
    env.scenario = lambda now: script.imu_raw((now - t0) / S)
    if calibrate:
        profile.save("rafae", profile.Calibration.default(), root=tmp_path)
    return env


def args_for(*extra):
    return tool.parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "rafae",
                            "--per-class", str(PER_CLASS), *extra])


def run(env, tmp_path, *, keys=None, notes=None, args=None):
    return tool.run(env, args or args_for(), profile_root=tmp_path, show=lambda f: None,
                    wait_key=keys or (lambda ms: 255), notify=(notes.append if notes is not None else (lambda n: None)),
                    size=(320, 180), frame_hz=2.0)


def test_the_whole_tool_on_fake_hardware_trains_and_saves_a_model_the_game_will_use(tmp_path):
    notes = []
    env = make(tmp_path)
    assert run(env, tmp_path, notes=notes) == 0
    model = spin.load_for("rafae", root=tmp_path)
    assert model is not None
    features, labels = fakerig.spin_dataset(2, seed=50)
    probs = model.probs(features[labels.index("top")])
    assert max(probs, key=probs.get) == "top"
    assert any("will use spin" in n for n in notes) and sum("captured" in n for n in notes) == 3 * PER_CLASS


def test_swings_that_do_not_match_what_was_asked_make_a_model_that_is_saved_but_not_used(tmp_path):
    notes = []
    env = make(tmp_path, scramble=True)
    assert run(env, tmp_path, notes=notes) == 0
    assert spin.load_for("rafae", root=tmp_path) is None                       # below the bar: the game stays flat
    assert spin.load_for("rafae", root=tmp_path, force=True) is not None       # but it is on file for inspection
    assert any("stays flat" in n for n in notes)


def test_it_needs_the_players_calibration_first_because_the_features_come_from_their_detector(tmp_path, capsys):
    env = make(tmp_path, calibrate=False)
    assert run(env, tmp_path) == 2 and "calibrate_swing" in capsys.readouterr().err


def test_pressing_q_cancels_saves_nothing_and_lets_the_hub_go(tmp_path):
    env = make(tmp_path)
    presses = iter([255, 255, ord("q")])
    assert run(env, tmp_path, keys=lambda ms: next(presses, ord("q"))) == 1
    assert spin.load_for("rafae", root=tmp_path, force=True) is None
    assert [c[0] for c in env.hub_device.calls][-2:] == ["motor_stop", "disconnect"]


def test_a_hub_that_is_not_found_is_a_clear_failure(tmp_path):
    env = make(tmp_path, hub_found=False)
    assert run(env, tmp_path) == 2


def test_too_few_swings_per_class_is_refused_at_the_command_line():
    with pytest.raises(SystemExit):
        tool.parse_args(["--per-class", "3"])


def test_the_window_frame_shows_the_prompt_progress_and_the_latest_note():
    from pingpong.spinflow import SpinCollector
    from pingpong.swing import SwingParams

    c = SpinCollector(SwingParams(), 6)
    a = tool.render_frame(c, "", size=(640, 360))
    b = tool.render_frame(c, "flat swing 1/18 captured", size=(640, 360))
    assert a.shape == (360, 640, 3) and int(np.abs(a.astype(int) - b.astype(int)).sum()) > 1000


def test_the_report_text_names_the_accuracy_and_the_verdict():
    text = tool.format_report({"n_per_class": {"flat": 12, "top": 12, "back": 12}, "cv_accuracy": 0.83, "cv_folds": 4,
                               "ship": True, "confusion": {"flat": {"flat": 10, "top": 1, "back": 1},
                                                           "top": {"flat": 1, "top": 11, "back": 0},
                                                           "back": {"flat": 2, "top": 0, "back": 10}}})
    assert "83%" in text and "will use spin" in text and "flat" in text
    bad = tool.format_report({"n_per_class": {"flat": 12, "top": 12, "back": 12}, "cv_accuracy": 0.4, "cv_folds": 4,
                              "ship": False, "confusion": {c: {d: 4 for d in spin.CLASSES} for c in spin.CLASSES}})
    assert "stays flat" in bad


def test_the_selftest_is_green():
    assert tool.main(["--selftest"]) == 0
