"""The paddle on screen turns with the hub: a fully assembled rig on fake hardware."""

import dataclasses

import numpy as np
import pytest

from pingpong import fakerig
from pingpong.events import ImuSample
from pingpong.profile import Calibration
from pingpong.tilt import TiltCalibration

S = 1_000_000_000


def test_a_rig_assembled_with_a_tilt_calibration_turns_the_paddle_with_the_hub_and_one_without_does_not():
    def turned(calibration, deg=30.0):
        rig = fakerig.FakeRig(calibration=calibration)
        for i in range(81):                                          # about the x axis, up to deg degrees, then held
            angle = min(deg, i * 0.8)
            a = (0, round(1000 * np.sin(np.radians(angle))), round(1000 * np.cos(np.radians(angle))))
            rig.rig.imu.samples.put_nowait(ImuSample(t_ns=5 * S + i * 15_000_000, a=a,
                                                     g=(round(0.8 * 66.7 * fakerig.GPD) if i * 0.8 < deg else 0, 0, 0)))
        rig.rig.imu.step()
        return rig.rig.hud_state().paddle_angle

    tilt = TiltCalibration(axis=(1.0, 0.0, 0.0), neutral=(0.0, 0.0, 1.0), bias_dps=(0.0, 0.0, 0.0))
    assert turned(dataclasses.replace(Calibration.default(), tilt=tilt)) == pytest.approx(30.0, abs=4.0)
    assert turned(Calibration.default()) == 0.0
