"""CalibrationFlow: stand still, hold four reach corners, hold upright and turn side to side, 5 soft + 5 full swings.

Driven by a scripted person (pingpong.fakerig.CalibrationScript) whose hub is mounted at an
arbitrary angle, so the learned forward axis has something real to find.
"""

import math

import pytest

from pingpong import fakerig
from pingpong.calibflow import CalibrationError, CalibrationFlow
from pingpong.events import ImuSample

GPD = fakerig.GPD
S = 1_000_000_000


def angle_deg(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))


def run_script(script, flow=None, **flow_kw):
    flow = flow or CalibrationFlow(gyro_per_dps=GPD, **flow_kw)
    steps_seen = []
    fakerig.drive_calibration(script, flow, on_step=steps_seen.append)
    return flow, steps_seen


def test_the_flow_walks_stand_corners_tilt_soft_full_done_with_a_prompt_for_each():
    flow, steps = run_script(fakerig.CalibrationScript())
    assert steps == ["stand", "corners", "tilt", "soft", "full", "done"]
    assert flow.finished()


def test_after_the_corners_the_hub_is_held_upright_then_turned_side_to_side():
    script = fakerig.CalibrationScript()
    tilt = run_script(script)[0].calibration().tilt
    assert tilt is not None
    assert angle_deg(tilt.axis, script.tilt_axis) < 5.0                    # the first move was to the right: positive about it
    assert angle_deg(tilt.neutral, (0, 0, 1)) < 3.0                        # upright is how the hub was held still


def test_the_tilt_step_asks_for_upright_and_still_first_and_then_for_the_turn():
    flow = CalibrationFlow(gyro_per_dps=GPD)
    flow.step = "tilt"
    assert "upright" in flow.prompt().lower() and flow.progress() == (0, 2)
    for i in range(int(4.5 * 64)):                                         # 4.5 s of holding the hub still, upright
        flow.feed_imu(ImuSample(t_ns=5 * S + round(i * S / 64), g=(0, 0, 0), a=(0, 0, 1000)))
    assert "side to side" in flow.prompt().lower() and flow.progress() == (1, 2)
    assert any("upright" in note for note in flow.take_notes())


def test_a_player_who_barely_turns_the_hub_is_told_to_turn_it_further_and_stays_on_the_step():
    notes, flow = [], CalibrationFlow(gyro_per_dps=GPD)
    fakerig.drive_calibration(fakerig.CalibrationScript(tilt_deg=8.0), flow, on_note=notes.append,
                              stop_after_notes=("further",))
    assert flow.step == "tilt" and any("further" in note for note in notes)


def test_the_forward_axis_strengths_box_and_shoulder_width_come_out_right():
    script = fakerig.CalibrationScript()
    flow, _ = run_script(script)
    cal = flow.calibration()
    assert angle_deg(cal.swing.u_fwd, script.u_true) < 4.0                  # the hub's mounting angle is learned
    assert cal.swing.omega_lo == pytest.approx(sorted(script.soft)[2], rel=0.08)
    assert cal.swing.omega_hi == pytest.approx(sorted(script.full)[2], rel=0.08)
    us = [c[0] for c in script.corners]
    vs = [c[1] for c in script.corners]
    assert cal.box.u_min == pytest.approx(min(us), abs=0.15) and cal.box.u_max == pytest.approx(max(us), abs=0.15)
    assert cal.box.v_min == pytest.approx(min(vs), abs=0.15) and cal.box.v_max == pytest.approx(max(vs), abs=0.15)
    assert cal.shoulder_w == pytest.approx(script.shoulder_w, abs=0.01) and cal.calibrated is True


def test_a_backswing_larger_than_the_forward_stroke_in_one_soft_take_does_not_flip_the_axis():
    script = fakerig.CalibrationScript(back_ratios=(0.5, 0.5, 1.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    flow, _ = run_script(script)
    assert angle_deg(flow.calibration().swing.u_fwd, script.u_true) < 6.0


def test_the_detector_with_the_new_calibration_fires_on_the_calibration_swings_themselves():
    flow, _ = run_script(fakerig.CalibrationScript())
    flow.calibration()
    detected, total = flow.self_check
    assert total == 10 and detected >= 9


def test_left_handed_players_get_a_left_handed_calibration():
    flow, _ = run_script(fakerig.CalibrationScript(hand="left"), hand="left")
    assert flow.calibration().hand == "left"


def test_prompts_name_the_step_and_progress_counts_swings():
    flow = CalibrationFlow(gyro_per_dps=GPD)
    assert "still" in flow.prompt().lower() and flow.progress() == (0, 1)
    seen = {}

    def on_step(name):
        seen.setdefault(name, flow.prompt())

    fakerig.drive_calibration(fakerig.CalibrationScript(), flow, on_step=on_step)
    assert "corner" in seen["corners"].lower() and "soft" in seen["soft"].lower() and "full" in seen["full"].lower()


def test_notes_are_handed_out_once_and_say_what_was_captured():
    flow = CalibrationFlow(gyro_per_dps=GPD)
    notes = []
    fakerig.drive_calibration(fakerig.CalibrationScript(), flow, on_note=notes.append)
    text = " | ".join(notes)
    assert "shoulder" in text and text.count("corner") >= 4 and text.count("soft swing") >= 5
    assert text.count("full swing") >= 5 and flow.take_notes() == []


def test_moving_around_does_not_capture_a_corner_and_a_still_hand_does():
    flow = CalibrationFlow(gyro_per_dps=GPD)
    flow.step = "corners"
    flow._shoulder_w = 0.2
    for k in range(60):                                          # a hand that never settles
        flow.feed_pose(k * S // 30, 0.5 * math.sin(k / 3), 0.3 * math.cos(k / 3), 0.9, 0.2)
    assert flow.progress() == (0, 4)
    for k in range(40):                                          # now it holds still for >0.8 s
        flow.feed_pose(3 * S + k * S // 30, 1.0, 0.6, 0.9, 0.2)
    assert flow.progress() == (1, 4)


def test_holding_still_at_the_same_corner_twice_does_not_count_it_twice():
    flow = CalibrationFlow(gyro_per_dps=GPD)
    flow.step = "corners"
    for k in range(120):                                         # 4 s at one spot
        flow.feed_pose(k * S // 30, 1.0, 0.6, 0.9, 0.2)
    assert flow.progress() == (1, 4)


def test_a_reach_box_that_is_too_small_is_rejected_and_the_corners_are_asked_for_again():
    script = fakerig.CalibrationScript(corners=((-0.25, 0.2), (0.25, 0.2), (0.25, -0.2), (-0.25, -0.2)))
    flow = CalibrationFlow(gyro_per_dps=GPD)
    notes = []
    fakerig.drive_calibration(script, flow, on_note=notes.append, stop_after_notes=("too small",))
    assert any("too small" in n for n in notes)
    assert flow.step == "corners" and flow.progress() == (0, 4)


def test_twitches_and_endless_waving_are_not_counted_as_swings():
    flow = CalibrationFlow(gyro_per_dps=GPD)
    flow.step = "soft"
    t = 0
    for k in range(200):                                         # rest
        flow.feed_imu(ImuSample(t_ns=t, g=(0, 0, 0), a=(0, 0, 1000)))
        t += S // 66
    for k in range(10):                                          # a 60 dps twitch: below the floor
        flow.feed_imu(ImuSample(t_ns=t, g=(round(60 * GPD * math.sin(k / 3)), 0, 0), a=(0, 0, 1000)))
        t += S // 66
    for k in range(300):                                         # 4.5 s of continuous waving: not one swing
        flow.feed_imu(ImuSample(t_ns=t, g=(round(500 * GPD * math.sin(k / 2)), 0, 0), a=(0, 0, 1000)))
        t += S // 66
    for k in range(100):
        flow.feed_imu(ImuSample(t_ns=t, g=(0, 0, 0), a=(0, 0, 1000)))
        t += S // 66
    assert flow.progress() == (0, 5)


def test_soft_and_full_swings_that_are_not_clearly_different_ask_for_harder_full_swings():
    script = fakerig.CalibrationScript(soft=(400,) * 5, full=(450,) * 5)
    flow = CalibrationFlow(gyro_per_dps=GPD)
    notes = []
    fakerig.drive_calibration(script, flow, on_note=notes.append, stop_after_notes=("harder",))
    assert any("harder" in n for n in notes)
    assert flow.step == "full" and flow.progress() == (0, 5)


def test_swings_so_gentle_that_random_movement_would_count_are_refused():
    script = fakerig.CalibrationScript(soft=(110,) * 5, full=(900,) * 5)
    flow = CalibrationFlow(gyro_per_dps=GPD)
    notes = []
    fakerig.drive_calibration(script, flow, on_note=notes.append, stop_after_notes=("firmer",))
    assert any("firmer" in n for n in notes)
    assert flow.step == "soft" and flow.progress() == (0, 5)


def test_the_calibration_is_unavailable_until_the_flow_has_finished():
    with pytest.raises(CalibrationError):
        CalibrationFlow(gyro_per_dps=GPD).calibration()


def test_the_result_is_independent_of_the_hub_unit_scale():
    # A wrong gyro_per_dps guess scales every peak the same way, so the thresholds stay consistent.
    script = fakerig.CalibrationScript(gpd=100.0)
    flow = CalibrationFlow(gyro_per_dps=100.0)
    fakerig.drive_calibration(script, flow)
    assert angle_deg(flow.calibration().swing.u_fwd, script.u_true) < 4.0


# --- the camera as the swing sensor ---------------------------------------------------------------------------------
def camera_flow(**kw):
    from pingpong import posegyro

    return CalibrationFlow(gyro_per_dps=posegyro.GYRO_PER_DPS, accel_per_g=posegyro.ACCEL_PER_G,
                           fs_raw=posegyro.FS_RAW, source="pose", **kw)


def test_the_flow_can_calibrate_from_the_hand_speed_alone():
    script = fakerig.CalibrationScript(u_true=(0.8, 0.6, 0.0), camera=True)
    flow = camera_flow()
    steps = []
    fakerig.drive_calibration(script, flow, on_step=steps.append, camera=True)
    cal = flow.calibration()
    assert steps == ["stand", "corners", "soft", "full", "done"]                  # no hub, no tilt: the camera has no hub to turn
    assert cal.tilt is None
    assert cal.swing.source == "pose" and angle_deg(cal.swing.u_fwd, script.u_true) < 8.0
    assert cal.swing.omega_hi > 1.5 * cal.swing.omega_lo
    detected, total = flow.self_check
    assert total == 10 and detected >= 9                   # the camera-source detector finds its own calibration swings
    assert cal.shoulder_w == pytest.approx(script.shoulder_w, abs=0.01)


def test_a_flow_for_the_camera_never_produces_a_hub_calibration():
    assert camera_flow().prompt().lower().startswith("stand still")
    flow = camera_flow()
    fakerig.drive_calibration(fakerig.CalibrationScript(u_true=(0.8, 0.6, 0.0), camera=True), flow, camera=True)
    assert flow.calibration().swing_source == "pose"
