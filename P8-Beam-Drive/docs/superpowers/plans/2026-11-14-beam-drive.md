# P8 Beam Drive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LEGO Double Motor with a beam on each axle (no wheels) drives straight under a policy trained in MuJoCo with mjlab, and the real robot's run matches the simulated run closely enough to show side by side.

**Architecture:** One parametric MuJoCo model of the robot (measured dimensions, box beams) is the single source for training and for sim-to-real tuning. The course's `Drive-Straight` mjlab task is copied and changed so the policy only sees what the real Double Motor can measure (IMU yaw, gyro, motor speed and position). The trained network is exported to plain numbers and run on the Mac at 20 Hz, talking to the Double Motor over Bluetooth.

**Tech Stack:** MuJoCo 3 + mjlab 1.6 (PPO, rsl_rl) on the Tufts Pax cluster through `~/ME193/mujoco-hpc/runIt.py`; local MuJoCo in `~/ME193/mujoco-hpc/.venv`; `legoeducation` Python API; Onshape for the CAD drawing; Notion for the writeup.

---

## What gets graded, and which task covers it

| Rubric item | Covered by |
|---|---|
| Notion page with GitHub link, good layout | Task 9 |
| CAD drawing | Task 1 |
| MuJoCo video | Task 5 |
| Description of reward system and overall policy | Task 4 (design), Task 9 (text) |
| Reflection questions | Task 9 (needs the question list from the assignment page) |
| Simulation and reality somewhat close | Tasks 2, 6, 7, 8 |

## Decisions already made (change here if you disagree)

1. **LEGO Double Motor, not the UNO Q.** It has two encoders, an IMU with yaw and a Python API already in `~/ME193/lego-docs`. The UNO Q would need motor drivers, encoders and a chassis built from scratch.
2. **Hand-written MuJoCo model for training, Onshape for the drawing.** The current export in `mujoco-hpc/09_onshape_robot/onshape/robot.xml` has no wheel joints (only a freejoint, 3 fixed parts), and mesh beams collide badly. A model made of boxes with the measured numbers in one place is quicker to fix and to match to reality.
3. **The policy only sees what the real robot measures.** No base velocity and no sideways offset for the actor (the critic keeps them during training).
4. **20 Hz control** in simulation and on the real robot, because the Bluetooth loop cannot go much faster.
5. **Project folder `~/ME193/P8-Beam-Drive`**, tracked in the `Me193` GitHub repo (that gives the GitHub link). `mujoco-hpc` stays untracked.

## File structure

```
P8-Beam-Drive/
  README.md                  how to run each step (Task 9)
  robot/beam_bot.xml         the robot: chassis box, two hinge joints, two box beams, skid
  robot/params.json          measured numbers (dimensions, mass, max speed, friction, loop rate)
  robot/build_xml.py         writes beam_bot.xml from params.json
  sim/open_loop.py           drive both beams at one speed, print x/y/yaw, save mp4 (no learning)
  sim/beam_task.py           the mjlab task "Beam-Straight" (from the course drive_task.py)
  sim/train.py, make_video.py  copies of the course files, importing beam_task
  sim/export_policy.py       checkpoint -> policy.npz (weights + observation normaliser)
  real/policy_np.py          numpy forward pass of policy.npz (shared by sim check and real robot)
  real/sysid.py              real robot: step response, loop rate, open-loop straight run -> CSV
  real/drive_policy.py       real robot: run the policy for 10 s -> CSV
  compare/plot_compare.py    overlay sim and real yaw / beam speed / distance -> PNG + table
  tests/test_model.py, tests/test_policy_np.py
  data/                      CSV logs and videos from the real robot
  docs/                      this plan, CAD drawing PDF, images for Notion
```

---

### Task 0: Hardware and tooling check (30 min, user + Claude)

- [ ] Confirm the hardware in hand: Double Motor, its connection card (colour + serial), two equal beams, axle pins, a front skid piece.
- [ ] Update the LEGO library (course notes say hub firmware now needs 1.2.0; installed is 1.1.1):
  `~/ME193/lego-single-motor/my_env/bin/pip install -U legoeducation`
- [ ] Run `~/ME193/lego-docs/examples/doublemotor.py` with the card values. Expected: the motor drives a square.
- [ ] Check `runIt.py` accepts a script outside its folder: from `~/ME193/mujoco-hpc`,
  `python3 runIt.py ../P8-Beam-Drive/sim/open_loop.py` (after Task 3). If it refuses, symlink `P8-Beam-Drive` into `mujoco-hpc/11_beam_drive`.

### Task 1: Build the robot and draw it in Onshape (user, 1 to 1.5 h)

- [ ] Build: one beam centred on each motor axle, both beams at the same angle to start, a smooth skid at the front so the body rests on beams + skid. The beam half-length must be larger than the axle height above the table, or the beams never touch the ground.
- [ ] Measure with calipers and a kitchen scale, write into `robot/params.json`:
  body length, width, height; axle position (from the rear, height from the bottom); distance between the two beams (track); beam length, width, thickness; skid position and height; total mass; beam mass.
- [ ] Onshape: model the body as a block, the two beams, the skid; assemble with two revolute mates named `dof_left_wheel` and `dof_right_wheel` (keeps the course exporter usable later).
- [ ] Make an Onshape Drawing: top, side and isometric views, with the dimensions from the list above. Export PDF + PNG to `P8-Beam-Drive/docs/`.

### Task 2: Measure the real robot (system identification, 45 min, user runs, Claude writes `real/sysid.py`)

`real/sysid.py` runs three tests and saves one CSV each in `data/` (columns: `t, left_pos, right_pos, left_speed, right_speed, yaw, gyro_z`):

- [ ] **Loop rate:** read sensors + send `motor_run` as fast as possible for 5 s; print the mean and worst period. Decides the control rate (plan assumes 20 Hz; use the measured value rounded down).
- [ ] **Step response, beams in the air:** speed 0 -> 50 % -> 100 % -> 0, 2 s each. Gives max beam speed in rad/s and how fast the motor reaches a new speed.
- [ ] **Open-loop straight run on the floor:** both motors 50 % for 5 s, three runs, phone filming from above with a tape measure in frame. Record distance forward, sideways drift and final yaw for each run.
- [ ] Write the results into `robot/params.json`: `max_speed_rad_s`, `motor_time_constant_s`, `loop_hz`, and the three open-loop results.

These open-loop numbers are the target the simulation has to hit in Task 3, and the "before" row of the comparison table.

### Task 3: MuJoCo model + open-loop match (Claude, 1.5 h)

**Files:** `robot/build_xml.py`, `robot/beam_bot.xml`, `sim/open_loop.py`, `tests/test_model.py`

- [ ] Write `tests/test_model.py` first (run with `~/ME193/mujoco-hpc/.venv/bin/python -m pytest tests/ -v`):
  - model loads and has joints `left_wheel`, `right_wheel` and body `chassis` (same names as the course task);
  - total mass equals `params.json` within 1 %;
  - after 2 s at rest the robot has not moved or tipped;
  - 5 s with both beams at 50 % speed moves the robot forward (x > 0) and it stays upright.
- [ ] `build_xml.py`: chassis box, hinge joints on the y axis, each beam a **box geom** centred on its joint, skid sphere with low friction, `<compiler angle="radian"/>`, beams with `condim="3"` and friction from `params.json`. Velocity actuators with `kv` and `forcerange` set from the step response.
- [ ] `open_loop.py`: same 50 % for 5 s as the real test; prints x, y, yaw each second; saves an mp4.
- [ ] Tune three numbers until the simulated open-loop run is within about 20 % of the real one in distance: beam friction, motor `kv`, torque limit. Write the final values and the sim-vs-real open-loop numbers in `params.json`.
- [ ] Commit.

Known MuJoCo traps from the two-paddle project: angles default to degrees; a contact uses the larger friction of the two geoms; short impacts can be missed at a 20 ms control step, so keep the physics step at 5 ms.

### Task 4: The mjlab task "Beam-Straight" (Claude, 1.5 h)

**Files:** `sim/beam_task.py` (copy of `mujoco-hpc/09_onshape_robot/mjlab_task/drive_task.py`), `sim/train.py`, `sim/make_video.py`

Changes from the course task:

| Part | Course task | Beam-Straight |
|---|---|---|
| Robot | Onshape export | `robot/beam_bot.xml` |
| Actor sees | base velocity (3), turn rate (3), heading + offset (3), wheel speed (2), last action (2) | sin/cos yaw (2), yaw rate (1), beam speeds (2), sin/cos of each beam angle (4), last action (2) = **11 numbers, all measurable on the Double Motor** |
| Critic sees | same as actor | actor's 11 + forward/sideways velocity + sideways offset |
| Action | 2 wheel speeds x 10 rad/s | 2 beam speeds x measured `max_speed_rad_s` |
| Control rate | 50 Hz (`decimation=4`) | 20 Hz (`decimation=10`, physics still 5 ms) |
| Reset | random yaw up to 0.4 rad | random yaw up to 0.3 rad **and random beam angles** |
| Randomised each episode | nothing | beam friction +-30 %, motor `kv` +-20 %, 0 to 2 steps of action delay, sensor noise on yaw and speeds |

The beam angles are in the observation because a beam, unlike a wheel, pushes only twice per turn: to go straight the policy has to keep the two beams in step, and it can only do that if it knows where they are.

**Reward (per step, summed over a 10 s episode):**

| Term | Weight | Meaning |
|---|---|---|
| `forward_speed` | +1.0 | 0 standing still, 1 at `TARGET_SPEED`, less beyond it, negative backwards. `TARGET_SPEED` = the real open-loop speed from Task 2, not the course's 0.5 m/s |
| `stay_on_line` | +1.0 | minus (sideways distance from y = 0) squared |
| `face_forward` | +1.0 | minus (heading error) squared; raised from 0.5 because heading is the one thing the real robot can sense |
| `action_rate` | -0.05 | minus (change in command) squared, keeps the motors from buzzing |

Episode ends early if the robot tips more than 60 degrees or is more than 1 m off the line.

- [ ] Write `beam_task.py` with the changes above; register it as `Beam-Straight`. Check the exact mjlab names for the delay and randomisation terms on the cluster first:
  `python3 runIt.py scripts/list_mdp.py` where the script prints `dir(mjlab.envs.mdp)` and `dir(mjlab.envs.mdp.dr)`.
- [ ] Short load check (64 robots, 10 min limit), from `~/ME193/mujoco-hpc`, in a visible terminal (password + Duo):
  ```
  python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with ../P8-Beam-Drive/sim/beam_task.py --with ../P8-Beam-Drive/robot ../P8-Beam-Drive/sim/train.py Beam-Straight --env.scene.num-envs 64
  ```
  Expected: iterations print with a mean reward, no traceback.
- [ ] Commit.

### Task 5: Train and record the MuJoCo video (user runs, about 30 min of waiting)

- [ ] Train 2000 iterations with 4096 robots (same command, `--time 00:45:00 --env.scene.num-envs 4096 --agent.max-iterations 2000`). Checkpoints come back to `mujoco-hpc/runs/train-*/logs/rsl_rl/beam_straight/<date>/`.
- [ ] Record the movie with `make_video.py model_1999.pt` (third `--with` for the checkpoint, as in the course README). It prints x, y, heading each second.
- [ ] Success for this task: in 10 s the simulated robot ends within 0.15 m of the line and within 10 degrees of straight, starting 0.3 rad off. If not: raise `face_forward`, lower `TARGET_SPEED`, train longer.
- [ ] Also record the **open-loop baseline** in simulation (equal speeds, no policy) from the same start. Policy vs baseline is the clearest picture of what the policy does.
- [ ] Copy the final checkpoint, video and the printed table into `P8-Beam-Drive/data/sim/`. Commit.

### Task 6: Export the policy and prove the copy is exact (Claude, 45 min)

**Files:** `sim/export_policy.py`, `real/policy_np.py`, `tests/test_policy_np.py`

- [ ] `export_policy.py` (runs on the cluster next to the checkpoint): loads the actor, writes `policy.npz` with each layer's weights and biases plus the observation normaliser's mean and std, and writes `check.npz` with 100 random observations and the network's outputs for them.
- [ ] `policy_np.py`: `act(obs) -> 2 numbers` using only numpy (normalise, two ELU layers of 64, linear output, clip to -1..1).
- [ ] `tests/test_policy_np.py`: `act()` reproduces every output in `check.npz` within 1e-5. This is the proof that the robot runs the same policy as the simulation.
- [ ] Commit.

### Task 7: Run the policy on the real robot (user runs, Claude writes `real/drive_policy.py`, 1 h)

- [ ] `drive_policy.py`: connect, `imu_reset_yaw_axis(0)`, then every 1/`loop_hz` s: read yaw, gyro, both motor speeds and positions; build the same 11 numbers in the same order and units as `beam_task.py` (radians, rad/s, yaw sign checked against the simulation); `act()`; send each motor its speed as a percentage of `max_speed_rad_s`. Stop after 10 s or on Ctrl-C, always calling `movement_stop()`. Log every step to `data/real/policy_<time>.csv`.
- [ ] Sign check before the floor run: beams in the air, turn the robot left by hand; the right beam must speed up (the policy steering back). If it steers the wrong way, flip the yaw sign in one place.
- [ ] Floor runs, filmed from above with the tape measure in frame: 3 runs starting straight, 3 runs starting about 15 degrees off. Record distance forward, final sideways offset, final yaw.
- [ ] Repeat the 6 runs with the open-loop baseline (already in `sysid.py`).

### Task 8: Compare simulation and reality, close the gap (Claude + user, 1 to 2 h)

**Files:** `compare/plot_compare.py`

- [ ] Replay the same starts in simulation with the same policy, log the same columns.
- [ ] `plot_compare.py` makes three overlays (yaw vs time, beam speeds vs time, distance vs time) and one table:

  | | Forward in 10 s | Sideways at end | Final heading |
  |---|---|---|---|
  | Sim, open loop | | | |
  | Real, open loop | | | |
  | Sim, policy | | | |
  | Real, policy | | | |

- [ ] "Somewhat close" target: real distance within 30 % of simulated, real final heading within 15 degrees, and the policy clearly straighter than open loop on the real robot.
- [ ] If it misses, fix in this order, one change at a time, and keep each before/after for the writeup: (1) friction and motor values in `params.json`, rerun Task 3's match; (2) wider randomisation ranges; (3) more action delay in training; (4) retrain and repeat Task 7.
- [ ] Side-by-side video: MuJoCo movie next to the overhead phone video of the same start.

### Task 9: Writeup (user's words, Claude builds the page, 1.5 h)

- [ ] `README.md` in `P8-Beam-Drive` with the run commands; push to `github.com/rafaeshafi/Me193` (only when you say "push it").
- [ ] Notion page, in this order: side-by-side video at the top; what the robot is (photo + CAD drawing); the simulation (model, how it was matched to the real robot); the policy (11 inputs, network, 2 outputs, 20 Hz) and the reward table from Task 4; training curve + MuJoCo video; sim-vs-real plots and the table; what did not match and why; reflection questions; GitHub link.
- [ ] Paste the reflection questions from the assignment page into the Notion page; they are not in the text given so far.

---

## Order and who does what

| Session | Tasks | Needs you |
|---|---|---|
| 1 | 0, 1, 2 | yes: build, measure, CAD, run the robot tests |
| 2 | 3, 4 | only to start cluster jobs (password + Duo) |
| 3 | 5, 6 | start training, watch the video |
| 4 | 7, 8 | yes: floor runs and filming |
| 5 | 9 | your own words on the page |

Tasks 3 and 4 can start before Task 2 with guessed numbers, but the tuning in Task 3 and the final training in Task 5 need the real measurements.

## Risks

- **Beams slip and hop**, so the real robot is noisier than the simulation. The randomisation in Task 4 and the heading-only policy are the answer; the course author's own rod-wheel robot wobbled within 27 degrees after 2000 iterations.
- **Bluetooth delay** larger than expected: measured in Task 2, trained against in Task 4.
- **IMU yaw sign or drift**: sign check in Task 7; drift over 10 s is small.
- **No Double Motor available**: fall back to the UNO Q with two DC motors and an IMU; Tasks 3 to 6 and 8 stay the same, Tasks 2 and 7 are rewritten for the board.
