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
        self.left += left * 0.5       # degrees per step, any number will do
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
