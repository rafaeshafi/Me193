"""The real-robot loop, run against a pretend robot (no Bluetooth)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "real"))

import drive_policy  # noqa: E402

REAL = {"forward_sign": -1, "left_count_sign": 1, "right_count_sign": -1, "yaw_sign": 1, "rad_s_per_percent": 0.2}


class PretendRobot:
    """Beams follow the last command at once; the heading never changes."""

    def __init__(self):
        self.left = self.right = 0.0
        self.sent = []

    def raw(self):
        return self.left, self.right, 0.0

    def zero(self):
        self.left = self.right = 0.0

    def drive(self, left, right):
        self.sent.append((left, right))
        self.left += left * 0.5       # degrees per step; the two motors count in opposite directions
        self.right -= right * 0.5


def test_open_loop_sends_the_asked_percent_and_logs(tmp_path, monkeypatch):
    params = dict(json.loads((ROOT / "robot" / "params.json").read_text()), real=REAL)
    monkeypatch.setattr(drive_policy, "ROOT", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    robot = PretendRobot()
    drive_policy.drive(robot, params, seconds=1.0, open_loop=40)
    assert len(robot.sent) >= params["loop_hz"] - 2                       # it kept to the loop rate
    assert np.allclose(robot.sent, -40)                                   # forward_sign -1 flips the command
    rows = list(csv.DictReader(next((tmp_path / "data" / "real").glob("run_*.csv")).open()))
    assert float(rows[-1]["left_rad"]) < 0 and float(rows[-1]["right_rad"]) < 0   # both count the same way after the signs


def test_reversed_robot_swaps_its_sides(tmp_path, monkeypatch):
    """Driving the other way round, the hub's left motor is the robot's right one."""
    params = dict(json.loads((ROOT / "robot" / "params.json").read_text()), real=dict(REAL, forward_sign=-1))
    monkeypatch.setattr(drive_policy, "ROOT", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "")

    class OneSided(drive_policy.Policy):
        def __init__(self):
            pass

        def act(self, obs):
            return np.array([1.0, 0.0])        # the robot's left beam only

    monkeypatch.setattr(drive_policy, "Policy", OneSided)
    robot = PretendRobot()
    drive_policy.drive(robot, params, seconds=0.5, open_loop=None)
    left, right = robot.sent[-1]
    assert left == 0 and right < 0             # sent to the hub's right motor, backwards for the hub
    rows = list(csv.DictReader(next((tmp_path / "data" / "real").glob("run_*.csv")).open()))
    assert float(rows[-1]["left_rad"]) != 0 and float(rows[-1]["right_rad"]) == 0   # and read back as the robot's left
