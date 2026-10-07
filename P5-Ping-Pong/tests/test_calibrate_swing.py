"""tools/calibrate_swing.py: the guided calibration, end to end on fake hardware."""

import math

import numpy as np
import pytest

from pingpong import fakerig, profile
from pingpong.calibflow import CalibrationFlow
from pingpong.sources_fake import FakeEnv, FakeLandmarker
from tools import calibrate_swing as tool

S = 1_000_000_000


def fake_env(script):
    env = FakeEnv(hz=66.0)
    t0 = env.clock.now_ns()
    env.scenario = lambda now: (0, 0, 1000, *script.imu_raw((now - t0) / S))
    env.make_landmarker = lambda: FakeLandmarker(lambda t_ns: script.hand_uv((t_ns - t0) / S), env.clock)
    return env


def args_for(tmp_path, *extra):
    return tool.parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "rafae", *extra])


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1, min(1, sum(x * y for x, y in zip(a, b))))))


def test_the_whole_tool_on_fake_hardware_saves_a_calibration_the_game_can_load(tmp_path):
    script = fakerig.CalibrationScript()
    env = fake_env(script)
    frames, notes = [], []
    code = tool.run(env, args_for(tmp_path), profile_root=tmp_path, show=frames.append,
                    wait_key=lambda ms: 255, notify=notes.append, size=(320, 180))
    assert code == 0
    cal = profile.load("rafae", root=tmp_path)
    assert cal is not None and cal.calibrated and cal.hand == "right"
    assert angle_deg(cal.swing.u_fwd, script.u_true) < 5.0
    assert cal.shoulder_w == pytest.approx(0.20, abs=0.01)                   # what the fake landmarks encode
    assert cal.box.u_max - cal.box.u_min > 1.8 and cal.box.v_max - cal.box.v_min > 1.0
    assert frames and any("calibration done" in n for n in notes)


def test_pressing_q_cancels_without_saving_and_lets_the_hub_go(tmp_path):
    script = fakerig.CalibrationScript()
    env = fake_env(script)
    presses = iter([255, 255, ord("q")])
    code = tool.run(env, args_for(tmp_path), profile_root=tmp_path, show=lambda f: None,
                    wait_key=lambda ms: next(presses, ord("q")), notify=lambda n: None, size=(320, 180))
    assert code == 1 and profile.load("rafae", root=tmp_path) is None
    assert [c[0] for c in env.hub_device.calls][-2:] == ["motor_stop", "disconnect"]


def test_a_hub_that_is_not_found_is_a_clear_failure(tmp_path):
    env = FakeEnv(hub_found=False)
    assert tool.run(env, args_for(tmp_path), profile_root=tmp_path, show=lambda f: None,
                    wait_key=lambda ms: 255, notify=lambda n: None) == 2


def test_the_window_frame_shows_the_prompt_progress_and_where_the_hand_is():
    flow = CalibrationFlow(gyro_per_dps=10.0)
    background = np.full((360, 640, 3), 90, dtype=np.uint8)
    plain = tool.render_frame(flow, background, None, "", size=(1280, 720))
    with_hand = tool.render_frame(flow, background, (0.4, 0.2), "corner 1/4 captured", size=(1280, 720))
    assert plain.shape == (720, 1280, 3) and int(np.abs(plain.astype(int) - with_hand.astype(int)).sum()) > 5000
    assert tool.render_frame(flow, None, None, "", size=(640, 360)).shape == (360, 640, 3)


def test_a_missing_card_is_reported_before_any_hardware_is_touched(capsys):
    assert tool.main(["--player", "rafae"]) == 2
    assert "scan_hubs" in capsys.readouterr().err


def test_the_selftest_is_green():
    assert tool.main(["--selftest"]) == 0
