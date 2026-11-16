# P8: Beam Drive

A LEGO Education Double Motor with two U-shaped beams instead of wheels, taught to walk in a straight line.
Modelled in Onshape, simulated in MuJoCo, trained with mjlab, then run on the real robot over Bluetooth.

Each beam pushes the robot once per turn, and the two beams sit half a turn apart, so at equal motor speeds
the robot zig-zags and drifts. The policy times the two beams against each other to hold a heading.

## Results

| | Simulation | Real robot |
|---|---|---|
| No policy, both motors at 50 % | ends about 8 degrees off line after 5 s | drifted up to 52 degrees off line in 10 s |
| With the policy | 0.87 m in 10 s, within 1 cm of the line, heading within 6 degrees | heading within 12 to 18 degrees over four 10 s runs |

The real runs used an early checkpoint of the same training run (round 50 of 300). The robot's weight in the
simulation is an estimate (150 g, 4 g per beam frame), and the real robot's distance travelled was not measured.

Videos: `data/sim/open_loop_onshape.mp4` (no policy), `data/sim/policy_onshape.mp4` (trained policy).
Real-robot logs: `data/real/run_*.csv`.

## How it fits together

| Step | File | What it does |
|---|---|---|
| CAD to MuJoCo | `onshape/` | The Onshape export (onshape-to-robot). The revolute mates `dof_left_wheel` and `dof_right_wheel` become the two joints. |
| Prepare the robot | `robot/from_onshape.py` | Turns the export upright, names the body `chassis`, sets weights, friction and speed motors from `robot/params.json`, writes `robot/beam_bot.xml`. |
| Drive with no policy | `sim/open_loop.py` | Both beams at one speed, with a video. |
| The task | `sim/beam_task.py` | The mjlab task `Beam-Straight`: observations, reward, randomisation. |
| Train | `sim/train.py` | PPO through mjlab. |
| Film a policy | `sim/make_video.py` | A 10 s drive of a checkpoint. |
| Export | `sim/export_policy.py` | Checkpoint to `real/policy.npz` (plain numbers). |
| Real robot | `real/drive_policy.py` | Runs `policy.npz` on the Double Motor at 20 Hz with numpy only. |

## The policy

A network with two hidden layers of 64 units, run 20 times a second. It reads 11 numbers, all of which the
real Double Motor can measure, and outputs a speed for each beam.

| Input | Count | From |
|---|---|---|
| Heading, as sine and cosine | 2 | IMU |
| Turning speed | 1 | IMU |
| Speed of each beam | 2 | motor encoders |
| Angle of each beam, as sine and cosine | 4 | motor encoders |
| Its own last command | 2 | memory |

The critic also sees the robot's true velocity and its distance from the line. That helps training and is not
needed on the real robot.

## The reward

| Term | Weight | Rewards |
|---|---|---|
| `forward_speed` | +1.0 | 1 at 0.10 m/s, falling with the square of the speed error, negative when going backwards |
| `stay_on_line` | +1.0 | minus the square of the sideways distance from the start line (m) |
| `face_forward` | +1.0 | minus the square of the heading error (rad) |
| `action_rate` | -0.05 | the square of the change in command between steps |

An episode lasts 10 s and ends early if the robot tips past 70 degrees or is more than 1 m off the line.
A perfect episode scores 10; the trained policy averages 8.9.

Randomised for every simulated robot, so the policy survives on hardware: friction x0.7 to 1.3, motor strength
x0.8 to 1.2, command delay up to 0.1 s, sensor noise, start heading up to 17 degrees, start beam angles.

## Run it

```
python -m venv .venv && .venv/bin/pip install mujoco==3.11.0 mjlab==1.6.0 legoeducation pytest imageio imageio-ffmpeg
.venv/bin/python -m pytest tests -q                      # 14 tests
.venv/bin/python robot/from_onshape.py                   # after a new Onshape export
.venv/bin/python sim/open_loop.py 50 5                   # no policy, 50 % for 5 s
cd runs && ../.venv/bin/python ../sim/train.py Beam-Straight --env.scene.num-envs 1024 --agent.max-iterations 300 --gpu-ids None
.venv/bin/python sim/export_policy.py <path to model_299.pt>
```

On the real robot (set the Connection Card under `"real"` in `robot/params.json` first):

```
.venv/bin/python real/drive_policy.py --check            # once, robot held in the air
.venv/bin/python real/drive_policy.py                    # drive with the policy for 10 s
.venv/bin/python real/drive_policy.py --open-loop 50     # no policy, to compare
```

Before each drive, put the beams in the Onshape pose: both flat, left bar forward, right bar back.
