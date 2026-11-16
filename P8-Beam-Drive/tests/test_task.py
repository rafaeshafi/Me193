"""The mjlab task builds and steps on the laptop's CPU (training itself needs the cluster's GPU).

Slow (about a minute: mjlab compiles its physics the first time).  .venv/bin/python -m pytest tests/test_task.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "sim"))

import beam_task  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402


@pytest.fixture(scope="module")
def env():
  cfg = beam_task.beam_straight_env_cfg()
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield env
  env.close()


def test_control_rate_matches_params():
  assert beam_task.DECIMATION * beam_task.TIMESTEP == pytest.approx(1 / beam_task.P["loop_hz"])


def test_actor_sees_11_numbers_and_critic_15(env):
  obs, _ = env.reset()
  assert obs["actor"].shape == (4, 11)
  assert obs["critic"].shape == (4, 15)


def test_beams_start_at_different_angles(env):
  env.reset()
  angles = env.scene["robot"].data.joint_pos
  assert angles.std() > 0.3


def test_equal_beam_speeds_move_it_forward(env):
  env.reset()
  robot = env.scene["robot"]
  start_x = (robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]).clone()
  action = torch.full((4, 2), 0.5)
  for _ in range(3 * beam_task.CONTROL_HZ):
    obs, reward, *_ = env.step(action)
  moved = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0] - start_x
  assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
  assert moved.mean() > 0.1
