"""mjlab task "Beam-Straight": teach the beam robot (robot/beam_bot.xml) to drive straight down +x.

Based on the course task 09_onshape_robot/mjlab_task/drive_task.py. What is different, and why:

- The policy only sees what the real LEGO Double Motor can measure: heading (IMU yaw), turning speed (gyro),
  beam speeds and beam angles (encoders), and its own last command. The critic also sees the robot's true
  velocity and sideways offset, which helps training but is never needed on the real robot.
- The beam angles are an input because a beam pushes only once per turn, so the two must be kept in step.
- It runs at the real robot's control rate (params.json "loop_hz"), with random command delay, friction,
  motor strength and sensor noise, so the policy does not depend on one exact simulation.

Importing this file registers the task (see train.py / make_video.py next to it).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  base_lin_vel,
  dr,
  joint_vel_rel,
  last_action,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- the robot's numbers come from robot/params.json ---------------------------------------------------
_HERE = Path(__file__).resolve().parent
ROBOT_DIR = next(p for p in (_HERE / "robot", _HERE.parent / "robot") if p.exists())
P = json.loads((ROBOT_DIR / "params.json").read_text())

TASK = "Beam-Straight"
LEFT, RIGHT = "left_wheel", "right_wheel"   # joint names in beam_bot.xml
TIMESTEP = 0.0025                           # s, physics step (same as sim/open_loop.py)
CONTROL_HZ = P["loop_hz"]                   # the policy acts this often, like the real Bluetooth loop
DECIMATION = round(1 / (CONTROL_HZ * TIMESTEP))
MAX_SPEED = P["max_speed_rad_s"]            # rad/s; a policy output of 1.0 means this beam speed

# ---- constants you may want to change ------------------------------------------------------------------
TARGET_SPEED = 0.10         # m/s forward the robot is rewarded for (set it to what the real robot manages)
START_YAW = 0.3             # rad; the robot starts pointing up to this far off +x
MAX_OFF_LINE = 1.0          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
MAX_DELAY_S = 0.1           # the command reaches the motor up to this late
FRICTION_SPREAD = (0.7, 1.3)    # each robot's friction is the params.json value times a number in this range
MOTOR_SPREAD = (0.8, 1.2)       # same for motor strength
# ---------------------------------------------------------------------------------------------------------


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "beam_bot.xml"))
  for actuator in list(spec.actuators):     # mjlab adds its own motors (below)
    spec.delete(actuator)
  spec.body("chassis").add_freejoint()
  return spec


def _robot_cfg() -> EntityCfg:
  # from_onshape.py sets this high enough that a beam pointing straight down starts above the floor
  start_height = float(_get_spec().body("chassis").pos[2])
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, start_height), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed it is given
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT),
          stiffness=0.0,
          damping=P["motor_kv"],
          effort_limit=P["motor_torque_limit"],
          delay_min_lag=0,
          delay_max_lag=round(MAX_DELAY_S / TIMESTEP),
          delay_update_period=DECIMATION,
        ),
      ),
    ),
  )


# ---- what the robot measures -----------------------------------------------------------------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line (its y axis), in metres."""
  return env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)], what the IMU's yaw gives. Shape [num_envs, 2]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def yaw_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Turning speed about the vertical axis in rad/s, what the gyro gives. Shape [num_envs, 1]."""
  return env.scene["robot"].data.root_link_ang_vel_w[:, 2:3]


def beam_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin, cos] of each beam's angle (0 = lying flat, bar end forward). Shape [num_envs, 4]."""
  angle = env.scene["robot"].data.joint_pos
  return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).unsqueeze(-1)


# ---- rewards ---------------------------------------------------------------------------------------------


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """0 when standing still, 1 at TARGET_SPEED, less beyond it, negative backwards."""
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - torch.square((vx - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env))


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment -----------------------------------------------------------------------------------------


def beam_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # The order here is the order of the 11 numbers the real robot must feed the policy (real/drive_policy.py).
  actor_terms = {
    "heading": ObservationTermCfg(func=heading, noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03)),
    "yaw_rate": ObservationTermCfg(func=yaw_rate, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
    "beam_vel": ObservationTermCfg(func=joint_vel_rel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5)),
    "beam_angles": ObservationTermCfg(func=beam_angles, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
    "last_action": ObservationTermCfg(func=last_action),
  }
  critic_terms = {
    **{name: ObservationTermCfg(func=term.func) for name, term in actor_terms.items()},
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "off_line": ObservationTermCfg(func=off_line),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "beams": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_SPEED, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
    "reset_beams": EventTermCfg(       # the two beams start at any angle, not lined up
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0)},
    ),
  }
  if not play:
    events["friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)), "operation": "scale", "ranges": FRICTION_SPREAD},
    )
    events["motor_strength"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={"kp_range": (1.0, 1.0), "kd_range": MOTOR_SPREAD, "operation": "scale"},
    )

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=1.0),
    "face_forward": RewardTermCfg(func=face_forward, weight=1.0),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(70.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.7,
      elevation=-25.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(njmax=128, mujoco=MujocoCfg(timestep=TIMESTEP)),   # the CAD shapes need more contact room than the default
    decimation=DECIMATION,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def beam_straight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="beam_straight",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )


register_mjlab_task(
  task_id=TASK,
  env_cfg=beam_straight_env_cfg(),
  play_env_cfg=beam_straight_env_cfg(play=True),
  rl_cfg=beam_straight_ppo_runner_cfg(),
)
