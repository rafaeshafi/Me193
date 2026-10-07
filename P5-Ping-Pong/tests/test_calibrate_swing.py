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
    env.scenario = lambda now: (*script.accel_raw((now - t0) / S), *script.imu_raw((now - t0) / S))
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
                    wait_key=lambda ms: 255, notify=notes.append, size=(320, 180), frame_hz=5.0)
    assert code == 0
    cal = profile.load("rafae", root=tmp_path)
    assert cal is not None and cal.calibrated and cal.hand == "right"
    assert angle_deg(cal.swing.u_fwd, script.u_true) < 5.0
    assert cal.shoulder_w == pytest.approx(0.20, abs=0.01)                   # what the fake landmarks encode
    assert cal.box.u_max - cal.box.u_min > 1.8 and cal.box.v_max - cal.box.v_min > 1.0
    assert cal.tilt is not None and angle_deg(cal.tilt.axis, script.tilt_axis) < 6.0     # the paddle will turn with the hub
    assert frames and any("calibration done" in n for n in notes)


def test_the_window_says_what_the_tilt_step_is_waiting_for_under_the_prompt(monkeypatch):
    from pingpong import canvas

    seen, real = [], canvas.draw_text
    monkeypatch.setattr(canvas, "draw_text", lambda frame, text, *a, **k: (seen.append(text), real(frame, text, *a, **k))[1])
    flow = CalibrationFlow(gyro_per_dps=10.0)
    flow.step = "tilt"
    tool.render_frame(flow, None, None, "", size=(1280, 720))
    assert flow.hint() and flow.hint() in seen and "S to skip" in " ".join(seen)
    seen.clear()
    tool.render_frame(CalibrationFlow(gyro_per_dps=10.0), None, None, "", size=(1280, 720))
    assert not any("steady" in text or "S to skip" in text for text in seen)         # no hint on the other steps


def test_pressing_s_at_the_tilt_step_skips_it_and_the_calibration_is_saved_without_a_tilt(tmp_path):
    env = fake_env(fakerig.CalibrationScript())
    notes = []
    code = tool.run(env, args_for(tmp_path), profile_root=tmp_path, show=lambda f: None, wait_key=lambda ms: ord("s"),
                    notify=notes.append, size=(320, 180), frame_hz=5.0)
    assert code == 0 and any("skipped" in n for n in notes)
    cal = profile.load("rafae", root=tmp_path)
    assert cal is not None and cal.tilt is None and cal.swing.omega_hi > cal.swing.omega_lo


def test_no_tilt_leaves_the_step_out_altogether(tmp_path):
    env = fake_env(fakerig.CalibrationScript())
    notes = []
    code = tool.run(env, args_for(tmp_path, "--no-tilt"), profile_root=tmp_path, show=lambda f: None,
                    wait_key=lambda ms: 255, notify=notes.append, size=(320, 180), frame_hz=5.0)
    assert code == 0 and not any("upright" in n or "tilt measured" in n or "tilt skipped" in n for n in notes)
    assert profile.load("rafae", root=tmp_path).tilt is None


def saved_without_a_tilt(tmp_path):
    from pingpong.calibration import SwingCalibration
    from pingpong.paddle import ReachBox

    old = profile.Calibration(swing=SwingCalibration((0.35, 0.88, -0.32), 330.0, 1100.0),
                              box=ReachBox(-1.3, 1.5, -1.4, 0.9), shoulder_w=0.13, hand="right")
    profile.save("rafae", old, root=tmp_path)
    return old


def same_but_the_tilt(a, b):
    """Equal up to the last bit of the re-normalised swing axis."""
    return (a.swing.u_fwd == pytest.approx(b.swing.u_fwd, abs=1e-12)
            and (a.swing.omega_lo, a.swing.omega_hi, a.box, a.shoulder_w, a.hand) ==
            (b.swing.omega_lo, b.swing.omega_hi, b.box, b.shoulder_w, b.hand))


def test_tilt_only_adds_the_tilt_to_the_saved_calibration_and_changes_nothing_else(tmp_path):
    old, script = saved_without_a_tilt(tmp_path), fakerig.CalibrationScript()
    notes = []
    code = tool.run(fake_env(script), args_for(tmp_path, "--tilt-only"), profile_root=tmp_path, show=lambda f: None,
                    wait_key=lambda ms: 255, notify=notes.append, size=(320, 180), frame_hz=5.0)
    assert code == 0 and any("tilt measured" in n for n in notes)
    new = profile.load("rafae", root=tmp_path)
    assert new.tilt is not None and angle_deg(new.tilt.axis, script.tilt_axis) < 6.0
    assert same_but_the_tilt(new, old)


def test_tilt_only_skipped_with_s_leaves_the_saved_calibration_as_it_was(tmp_path):
    old = saved_without_a_tilt(tmp_path)
    notes = []
    code = tool.run(fake_env(fakerig.CalibrationScript()), args_for(tmp_path, "--tilt-only"), profile_root=tmp_path,
                    show=lambda f: None, wait_key=lambda ms: ord("s"), notify=notes.append, size=(320, 180), frame_hz=5.0)
    assert code == 0 and profile.load("rafae", root=tmp_path).tilt is None
    assert same_but_the_tilt(profile.load("rafae", root=tmp_path), old)


def test_tilt_only_without_a_calibration_to_add_it_to_says_so_before_touching_any_hardware(tmp_path, capsys):
    env = fake_env(fakerig.CalibrationScript())
    code = tool.run(env, args_for(tmp_path, "--tilt-only"), profile_root=tmp_path, show=lambda f: None,
                    wait_key=lambda ms: 255, notify=lambda n: None)
    assert code == 2 and "calibrate_swing --player rafae" in capsys.readouterr().err
    assert not env.hub_device.calls                                              # nothing was connected or beeped


def test_what_the_tool_complains_about_beeps_low_and_shows_red_and_what_it_accepts_does_not():
    for complaint in ("that swing was too gentle (80 dps): swing a bit firmer", "turn it both ways: right, then left",
                      "the hub hardly turned: turn it side to side, further", "turn it about one axis, like a doorknob",
                      "hold the hub steady for a second first, then turn it side to side",
                      "that was waving, not a swing: one clean swing at a time"):
        assert tool.is_complaint(complaint), complaint
    for fine in ("soft swing 1/5: peak 330 dps", "upright captured: now turn the hub side to side like a doorknob, right first",
                 "tilt measured: the paddle on screen will turn with the hub", "corner 1/4 captured (top-left)"):
        assert not tool.is_complaint(fine), fine


def test_pressing_q_cancels_without_saving_and_lets_the_hub_go(tmp_path):
    script = fakerig.CalibrationScript()
    env = fake_env(script)
    presses = iter([255, 255, ord("q")])
    code = tool.run(env, args_for(tmp_path), profile_root=tmp_path, show=lambda f: None,
                    wait_key=lambda ms: next(presses, ord("q")), notify=lambda n: None, size=(320, 180), frame_hz=5.0)
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


# --- the camera as the swing sensor ---------------------------------------------------------------------------------
def camera_env(script):
    env = FakeEnv(hz=66.0)
    t0 = env.clock.now_ns()
    env.make_landmarker = lambda: FakeLandmarker(lambda t_ns: script.hand_uv((t_ns - t0) / S), env.clock)
    return env


def run_camera(tmp_path, env, *extra):
    return tool.run(env, args_for(tmp_path, "--swing-source", "pose", *extra), profile_root=tmp_path,
                    show=lambda f: None, wait_key=lambda ms: 255, notify=lambda n: None, size=(320, 180), frame_hz=5.0)


def test_the_tool_calibrates_from_the_camera_alone_and_saves_a_camera_calibration(tmp_path):
    script = fakerig.CalibrationScript(u_true=(0.8, 0.6, 0.0), camera=True)
    env = camera_env(script)
    assert run_camera(tmp_path, env, "--no-hub") == 0
    cal = profile.load("rafae", root=tmp_path, source="pose")
    assert cal is not None and cal.calibrated and cal.swing.source == "pose"
    assert angle_deg(cal.swing.u_fwd, script.u_true) < 8.0
    assert profile.load("rafae", root=tmp_path) is None                     # no hub calibration was invented
    assert env.hub_device.calls == []                                       # and no hub was touched


def test_with_a_hub_the_camera_calibration_still_beeps_the_captures_and_ignores_the_hubs_gyro(tmp_path):
    script = fakerig.CalibrationScript(u_true=(0.8, 0.6, 0.0), camera=True)
    env = camera_env(script)
    env.scenario = lambda now: (0, 0, 1000, 3000, -2000, 500)               # a hub reading nonsense the whole time
    assert run_camera(tmp_path, env) == 0
    assert any(c[0] == "beep" for c in env.hub_device.calls)
    assert profile.load("rafae", root=tmp_path, source="pose").swing.source == "pose"


def test_no_hub_with_an_explicit_hub_gyro_is_refused_before_anything_starts(tmp_path):
    env = FakeEnv()
    code = tool.run(env, args_for(tmp_path, "--no-hub", "--swing-source", "imu"), profile_root=tmp_path,
                    show=lambda f: None, wait_key=lambda ms: 255, notify=lambda n: None)
    assert code == 2 and env.hub_device.calls == []


def test_calibrating_on_the_hub_before_its_units_were_measured_warns_but_the_camera_never_does(tmp_path, monkeypatch, capsys):
    import config

    monkeypatch.setattr(config, "MEASURED", set())
    script = fakerig.CalibrationScript()
    tool.run(fake_env(script), args_for(tmp_path), profile_root=tmp_path, show=lambda f: None,
             wait_key=lambda ms: ord("q"), notify=lambda n: None, size=(320, 180), frame_hz=5.0)
    assert "bench_hub" in capsys.readouterr().err
    camera = fakerig.CalibrationScript(u_true=(0.8, 0.6, 0.0), camera=True)
    run_camera(tmp_path, camera_env(camera), "--no-hub")
    assert "bench_hub" not in capsys.readouterr().err
