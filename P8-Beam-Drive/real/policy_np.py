"""The trained policy with numpy only (no torch, no MuJoCo), for the laptop that talks to the real robot.

The observation is 11 numbers in the same order the simulation used (sim/beam_task.py):
  sin(heading), cos(heading), turning speed (rad/s), left and right beam speed (rad/s),
  sin(left angle), sin(right angle), cos(left angle), cos(right angle), last left and right command.
Heading is positive to the left. A beam angle of 0 is the pose in the Onshape assembly. A positive beam
speed rolls the robot forward. The output is two commands, about -1 to 1, where 1 means params "max_speed_rad_s".
"""
import math
from pathlib import Path

import numpy as np

EPS = 1e-2   # rsl_rl's observation scaling adds this to the standard deviation


def make_obs(heading, turn_rate, beam_speeds, beam_angles, last_action):
    left, right = beam_angles
    return np.array([math.sin(heading), math.cos(heading), turn_rate, beam_speeds[0], beam_speeds[1],
                     math.sin(left), math.sin(right), math.cos(left), math.cos(right),
                     last_action[0], last_action[1]])


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class Policy:
    def __init__(self, path=Path(__file__).resolve().parent / "policy.npz"):
        self.p = dict(np.load(path))
        if self.p["w0"].shape[1] != 11 or self.p["w2"].shape[0] != 2:
            raise ValueError(f"{path}: expected a policy with 11 inputs and 2 outputs")

    def act(self, obs):
        p = self.p
        x = (np.asarray(obs, dtype=float) - p["mean"][0]) / (p["std"][0] + EPS)
        x = _elu(p["w0"] @ x + p["b0"])
        x = _elu(p["w1"] @ x + p["b1"])
        return p["w2"] @ x + p["b2"]
