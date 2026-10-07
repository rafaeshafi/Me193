"""The calibration's tilt step with a HUMAN hand: a shaky hold, a turn that wanders off the axis, turning too early.

The first real attempt never got past this step: holding a hub 'still' in a hand is nowhere near still enough for a
rate threshold, a wrist turn is not about one perfect axis, and nothing on the screen said what the step was waiting for.
"""

import numpy as np
import pytest

from pingpong.calibflow import CalibrationFlow
from pingpong.events import ImuSample
from test_tilt import GPD, UP, rotate

APG, HZ = 1000.0, 64.0
R = np.array([0.8, 0.6, 0.0])                         # the axis the hub is turned about (perpendicular to 'upright')
R2 = np.array([-0.6, 0.8, 0.0])                       # another axis, for a turn that wanders
S = 1_000_000_000


def person(*, hold_s=3.0, turn_deg=40.0, turn_s=4.0, tremor_dps=0.0, off_axis=0.0, first=1.0, after_s=0.5, seed=1):
    """Raw samples: hold the hub upright (a shaky hand), then turn it side to side about R (two cycles in turn_s)."""
    rng = np.random.default_rng(seed)
    r = R / np.linalg.norm(R)
    r2 = R2 / np.linalg.norm(R2)
    out = []
    for i in range(int((hold_s + turn_s + after_s) * HZ)):
        tau = i / HZ - hold_s
        turning = 0.0 <= tau < turn_s
        phi = first * turn_deg * np.sin(2 * np.pi * 0.5 * tau) if turning else 0.0
        rate = first * turn_deg * np.pi * np.cos(2 * np.pi * 0.5 * tau) if turning else 0.0
        g = rate * r + off_axis * rate * 0.8 * r2 * np.sin(1.3 * tau) + rng.normal(0, tremor_dps, 3)
        a = rotate(UP, r, -phi) + rng.normal(0, 0.012, 3)
        out.append(ImuSample(t_ns=5 * S + round(i * S / HZ), g=tuple(np.round(g * GPD).astype(int)),
                             a=tuple(np.round(a * APG).astype(int))))
    return out


def tilt_flow():
    flow = CalibrationFlow(gyro_per_dps=GPD, accel_per_g=APG)
    flow.step = "tilt"
    return flow


def feed(flow, samples):
    for s in samples:
        flow.feed_imu(s)
    return flow


def angle_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def test_a_hand_that_shakes_while_holding_the_hub_is_still_steady_enough():
    flow = feed(tilt_flow(), person(tremor_dps=12.0))                 # 12 dps of tremor on every axis, as a held hub has
    assert flow.step == "soft", flow.take_notes()
    assert angle_between(flow.tilt.axis, R) < 6.0


def test_turning_without_waiting_for_the_prompt_is_measured_too():
    flow = feed(tilt_flow(), person(hold_s=2.1, tremor_dps=6.0))      # barely a second of holding, then straight into it
    assert flow.step == "soft", flow.take_notes()


def test_a_wrist_turn_that_wanders_off_the_axis_is_accepted():
    flow = feed(tilt_flow(), person(off_axis=1.0, tremor_dps=6.0))    # a second rotation at 80% of the first one's size
    assert flow.step == "soft", flow.take_notes()
    assert angle_between(flow.tilt.axis, R) < 30.0


def test_turning_before_the_hub_was_ever_steady_asks_to_hold_it_steady_first_and_listens_again():
    flow = feed(tilt_flow(), person(hold_s=0.3))
    assert flow.step == "tilt" and any("steady" in note for note in flow.take_notes())
    later = person(hold_s=2.5, tremor_dps=6.0)
    shift = 5 * S + round(6.0 * S)
    feed(flow, [ImuSample(t_ns=s.t_ns + shift, g=s.g, a=s.a) for s in later])
    assert flow.step == "soft"


def test_the_window_hint_says_what_the_step_is_waiting_for():
    flow = tilt_flow()
    assert flow.hint() == "" or "steady" in flow.hint().lower()
    samples = person(hold_s=3.0)
    feed(flow, samples[: int(1.0 * HZ)])
    assert "steady" in flow.hint().lower() and "of" in flow.hint()
    feed(flow, samples[int(1.0 * HZ): int(3.6 * HZ)])
    assert "turn" in flow.hint().lower()                              # steady, so now it waits for the turn
    feed(flow, samples[int(3.6 * HZ): int(5.0 * HZ)])
    assert "turning" in flow.hint().lower()
    assert CalibrationFlow(gyro_per_dps=GPD).hint() == ""             # other steps have no such line


def test_a_turn_that_is_too_gentle_to_start_is_told_to_turn_harder():
    flow = feed(tilt_flow(), person(hold_s=3.0, turn_deg=9.0))        # peaks at 28 dps: below what starts a turn
    assert flow.step == "tilt" and "harder" in flow.hint().lower()


def test_skipping_the_tilt_step_moves_on_and_leaves_the_paddle_upright():
    flow = tilt_flow()
    flow.skip_tilt()
    assert flow.step == "soft" and flow.tilt is None
    assert any("skipped" in note for note in flow.take_notes())
    other = CalibrationFlow(gyro_per_dps=GPD)
    other.skip_tilt()                                                 # only the tilt step can be skipped
    assert other.step == "stand"


def test_a_flow_told_not_to_ask_for_a_tilt_goes_straight_from_the_corners_to_the_swings():
    flow = CalibrationFlow(gyro_per_dps=GPD, tilt=False)
    flow.corners = [(-1.1, 0.7), (1.1, 0.7), (1.1, -0.6)]
    flow.step = "corners"
    flow._corner(5 * S, -1.1, -0.6)
    flow._window.clear()
    flow.corners = [(-1.1, 0.7), (1.1, 0.7), (1.1, -0.6), (-1.1, -0.6)]
    flow._check_box()
    assert flow.step == "soft"
