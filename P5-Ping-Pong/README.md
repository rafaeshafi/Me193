# P5: Ping-Pong

**Play virtual table tennis with a LEGO Education Double Motor as your paddle.**

ME 193 – AI & Robotics, Project 5. (It is P5 because `P4-Door-To-Door-Service` already exists.)

You stand about 1.8 m from the laptop with the hub in your fist. The laptop shows a table tennis table in
perspective, seen from behind your end like an arcade game (a ball with a shadow that bounces, a paddle with depth), and
the computer serves a ball at you. Your webcam picture is a small corner of the screen, with a ring on the hand it tracks.

- **Pose (MediaPipe)** says *where* your paddle is: your hand, relative to your shoulders. Across the box is across the
  table; up and down is how far up the table the paddle stands (raise your hand to meet the ball sooner).
- **The hub's IMU** says *when* you swung and *how hard*: the gyro peak sets the speed of your return.
  (If the bench finds the hub too slow to see a swing, the camera's hand speed does this job instead.)
- **AprilTag cards** start the game (card 0) and set the level (cards 1–3 = Rookie, Club, Pro),
  which sets how fast the computer's balls come.
- **MQTT** carries your score live: the best streak of continuous hits goes to
  `ME193/Rogers/RafaeShafi` as a float.
- **Haptics (the new thing)**: the hub's own motors, beep and light give you a cue for a perfect
  hit, an early or late one, a fault, a point and a new record, so you can play without
  looking at the screen.

Two modes: **Rally** (called Survival in the code and on `--mode survival`: the computer is a player with its own
paddle and never misses; how long can you keep the rally going? the balls speed up) and **Match** (first to 7; the
computer misses more the harder, spinnier and wider you hit).

> Run everything that touches Bluetooth or the camera from **Terminal.app**, not from inside the
> Claude app (macOS aborts the process there). The tools say so instead of crashing.

## Contents

- [What you need](#what-you-need)
- [Setup](#setup)
- [First time: the bench, in order](#first-time-the-bench-in-order)
- [Playing](#playing)
- [The score on MQTT](#the-score-on-mqtt)
- [What it shows and records](#what-it-shows-and-records)
- [Demo day: the graded take](#demo-day-the-graded-take)
- [When something goes wrong](#when-something-goes-wrong)
- [Project layout and tests](#project-layout-and-tests)
- [Questions](#questions)
- [Note on versions](#note-on-versions)

## What you need

| | |
|---|---|
| Mac | webcam, speakers, Bluetooth; a table to stand the laptop on, lid tilted so your head to hips is in frame at ~1.8 m, front-lit (no window behind you) |
| LEGO Education Double Motor | held in your dominant fist; a wrist lanyard on every player; its Connection Card colour + 4-digit serial |
| Printed AprilTag cards | `./pp make_cards` writes `docs/cards/card0-3.pdf`; print at **Actual size** on matte paper and check the tag edge with a ruler (15 cm) |

## Setup

```bash
cd ~/ME193/P5-Ping-Pong
python3.12 -m venv my_env && my_env/bin/pip install -r requirements.txt   # exact pins: see the note on versions
./pp ready                                                                # imports + pytest + every tool's selftest
```

`./pp` always runs the project's own venv Python. Never type bare `python` here.
`./pp ready` needs no hardware and is the gate to run before every session.

## First time: the bench, in order

Every tool that measures something writes it to `config_local.json` (untracked); numbers it does
not trust are reported and **not** written. Do these from Terminal.app:

```bash
./pp scan_hubs                 # passive BLE scan: shows the card colour + serial of hubs in range
./pp env_check                 # camera frame, hub connect + beep, MQTT round trip, hub IMU rate at the play spot
./pp bench_hub --guided        # IMU rate, units (six faces, three turns), clipping, and swing fixtures
./pp bench_cam                 # pose fps, tag read rate at 1.8 m, camera-vs-IMU lag, hub rate under load (a live
                               # window shows whether you are in position; Enter works there and in the terminal)
./pp bench_haptics             # which motor pulses you feel, and how long the IMU rings after one
./pp calibrate_swing --player rafae   # shoulders, four reach corners, hub upright + turned side to side, 5 soft + 5 full swings (~4 minutes)
./pp calibrate_swing --player rafae --tilt-only   # just the paddle turning with the hub (~15 s; the rest is kept)
./pp train_spin --player rafae        # optional: 12 flat, 12 top, 12 back swings -> your spin model (~3 minutes)
./pp train_pose --player rafae --record   # optional: a 95 s guided take -> your pose model, a steadier hand (~3 minutes)
```

If `env_check` or `bench_hub` measure the hub's IMU **below 25 Hz** (too slow to see a quick swing), `play` and
`calibrate_swing` switch by themselves to the camera's hand speed as the swing sensor (`--swing-source auto` is the
default; `imu` or `pose` forces one). That needs its own calibration, a separate file that never replaces the hub's:

```bash
./pp calibrate_swing --player rafae --swing-source pose     # the hub stays on for haptics and beeps
./pp calibrate_swing --player rafae --no-hub                # camera only (implies --swing-source pose)
```

The card is saved in `config_local.json` after `env_check` connects (or pass
`--card-color red --card-serial 1131`). If a killed run left the hub invisible
(it stays "connected" for ~24 s), `./pp reset_hub` frees it.

**A steadier, more accurate hand.** The paddle is your hand measured from your shoulders, so anything that makes the
shoulders or the hand jump makes the paddle jump. The game follows a smoothed shoulder midpoint (`body.py`: the unit is
the 80th percentile of your shoulder width over the last 2 s, so turning your torso in a swing does not shrink it; it keeps
the anchor for a second when a shoulder is hidden, and only refuses a body that jumps or changes size for good), holds each
finger's position from the wrist when a finger point flickers, and filters the hand with a One-Euro filter behind a gate
that drops a one-frame landmark flip (`posemodel.py`). `./pp train_pose --player rafae --record` makes that fit *you*:
you follow the screen for 95 s (stand still, slide, lift, follow an imaginary ball, swing) while the camera is recorded, and
the same footage is then run through MediaPipe's light and its full pose model, so the one with the least noise that still
keeps up with 30 frames a second is chosen on your camera, room and body; the filter's three settings are tuned so the live
(causal) filter follows a zero-phase smoothing of your own raw readings, among the settings that jitter a still hand no more
than the defaults do (clean readings would otherwise "win" with no filtering at all and the paddle would shiver); and a ridge-regression predictor of where the hand
will be when the frame is seen is trained on your tracks, but only used if it cuts the error against simply holding the hand
by 10% on tracks it was not trained on without making the paddle shimmer more than 1.5 times as much. On the six games
I had recorded when I first trained it, it did not (0% better), so the paddle is drawn where the camera read the hand. The result is
`data/players/<name>/pose_model.json`; `./pp train_pose --dry-run` shows it without saving, and every game keeps the readings
before and after the filter, so more games make the next training better. `./pp report` says how the shoulders were followed.

## Playing

```bash
./pp play --player rafae                 # live: hub + camera + tags, scores to the broker
./pp play --player rafae --level 2 --mode match --target 7
./pp play --player rafae --learn         # the computer learns where you fail (Q-learning, kept between games)
./pp play --player guest                 # no saved calibration; never publishes the score
./pp play --fake                         # no hardware: mouse is the paddle, SPACE/J/K swing
./pp play --no-publish --no-motor        # rehearse without the broker / without motor pulses
./pp play --player rafae --swing-source pose   # the camera's hand speed detects swings (auto if the hub is < 25 Hz)
./pp play --no-hub                       # camera only: no hub, no haptics (bring-up, or a flat battery)
./pp play --board                        # the leaderboard and nothing else
```

Show the **START card** (or press SPACE): 3-2-1, then the computer serves. Show **card 1, 2 or 3**
between rallies to change the level; **M** switches Rally/Match; **1–3** also set the level.

| Key | |
|---|---|
| SPACE | start (and swing, in `--fake`) |
| 1 2 3 | level: Rookie, Club, Pro |
| M | Rally / Match |
| X | x-ray: why each swing did or did not count |
| D | disarm the motors (beep and light stay) |
| S | mute / unmute the sounds |
| R | ask a lost hub to reconnect |
| Q / Esc | quit (stops the motors, saves, lets the hub go) |
| J / K | soft / hard swing (`--fake` only) |

| Level | ball speed | flight (3 m) | hit window early / late | reach across the table | faults | balls arrive in |
|---|---|---|---|---|---|---|
| Rookie | 2.5 m/s | 1.2 s | 0.50 / 0.22 s | 0.70 shoulder widths | never | the middle 60% of your reach box |
| Club | 5.0 m/s | 0.60 s | 0.30 / 0.16 s | 0.55 | never | the middle 80% |
| Pro | 7.0 m/s | 0.43 s | 0.22 / 0.12 s | 0.42 | hard and sloppy | the whole box |

**What the screen shows.** A table in perspective, seen from behind your end (the camera that draws it is 1.4 m up and
1.8 m behind your edge, pitched down 28 degrees; `pingpong/court3d.py`). The **red paddle is yours**: it stands on the table
where your hand puts it, **across** the table as you move left and right and **up the table** as you raise your hand, gets
smaller and higher on the screen as it reaches forward, **turns when you turn the hub** in your fist, and has a fist
round its handle. The **blue paddle is the computer**: it waits where it hit, moves to where your shot will land, and
flicks when it hits the ball back. The ball is a yellow disc with a **shadow on the table**, so how high it is and how far
off it is can be read, and it **bounces once** on each side. The **amber oval on the table** is your paddle's reach: it is
as wide as the level allows across and as long as the level's timing window is along the table; a ball whose shadow is
inside it can be hit, and over the amber line across the paddle it is exactly on time. The two boxes at the sides are
the streak and the best (Rally) or the points (Match); the camera picture is in the bottom corner. `--set level.radius_sw=0.9`
widens the oval, `--set level.early_s=0.6` lengthens it.

**One shot.** The computer's ball bounces on your half and passes your paddle at a known instant. You swing; the swing
detector reports the gyro's peak (back-dated to when it really peaked), the judge dates the **contact** a stroke later
(the peak plus 0.16 s: the forward stroke of a swing ends a median 0.20 s after the peak of its rate on my 28 recorded
swings, that is when I mean the paddle to meet the ball, and the hub's 40 ms are already in the stamp) and checks six
gates (below). On a hit the **paddle lunges forward to wherever the ball
is** at the contact (so an early swing meets it further up the table), and the ball goes back from there. The peak gyro
rate becomes the speed of your return (`3 + 11·s^0.8` m/s, `s` = 0..1 between your soft and full calibration swings), your
hand position the aim, and at Pro and Insane a deterministic risk rule makes a net or out fault when you swing hard *and*
sloppily (`s·(1 − quality) > threshold`; a near-perfect hit never faults). Rookie and Club never fault: a hit is a hit.
With a trained **spin model** the swing's own rotation and acceleration decide flat, topspin or backspin:
topspin shortens the ball's flight after the bounce, backspin lengthens it, and in Match more spin makes the
computer more likely to miss. Without a model (or one that did not reach 75% cross-validated accuracy) every
ball is flat; `--no-spin` ignores the model.

**Where the time goes.** Every stage has a delay and the game compensates for each one so that what you see, what is
judged and what you feel line up (`pingpong/latency.py`, tunable with `--set latency.display_s=0.08` or in
`config_local.json`): the hub's samples are stamped when they *arrive*, about 40 ms after your hand did it, and the
swing is dated accordingly; the screen shows a frame 50 ms after it is drawn, so the ball and the computer's paddle (whose
flights are known exactly) are **drawn ahead** by that much and what you see is where the ball is; your hand (the camera
pipeline is 100 ms or more behind it) is drawn where the camera read it, because extrapolating its speed across that gap,
which is what I first did, was no more accurate than holding it on my recorded games (3% worse) and made the paddle shimmer four
times as much (a trained predictor can lead it again, see `./pp train_pose`); and the thump of a hit and its sound are **sent
early** by the motors' and the speakers' delays so they arrive when the picture shows the contact. The camera's own lag is the
one delay measured on this hardware (`./pp bench_cam`).

**The six gates** (the x-ray shows each one with its reason): **J1** the contact inside the level's window around that moment ·
**J2** your hand level with the ball *across the table* (its height does not matter) from 0.3 s before to 0.05 s after the
impact and at the impact; the ball is where its flight puts it at the contact, the moment is when the ball is over the
paddle your hand height puts on the table · **J3** swing big and
clean enough · **J4** pose and IMU agree on the moment (logged only) · **J5** one hit per ball, not
too fast · **J6** paddle not locked after the hub was shaken.

## The score on MQTT

| | |
|---|---|
| broker | `test.mosquitto.org:1883` (override with `PP_BROKER=host[:port]`) |
| topic | `ME193/Rogers/RafaeShafi` |
| payload | a float as text: `0.0`, `12.0`; QoS 1, retained |
| value | the **best streak this run** (it ticks up live while you build a new best and holds after a miss) |

Nothing is published at start-up (a restart never resets the retained value); the first publish is
your first hit. A plain run starts from zero ("best streak this run"), so its first hit publishes `1.0` over
whatever the broker held: the game subscribes to its own topic (read-only), and when the broker already holds a
value the lobby and the console say so before anyone swings. `./pp play --resume` keeps that value as the best
to beat (nothing is published until you pass it), which is also what you want after a crash in the graded run. Fake, simulated and demo values go to `ME193-pp/RafaeShafi/demo/score`, replays and
`--no-publish` publish nothing, a guest never publishes, and the last-will ("offline") lives on
`ME193-pp/RafaeShafi/status`, never on the score topic. Watch it the way the instructor will:

```bash
mosquitto_sub -h test.mosquitto.org -t 'ME193/Rogers/#' -v
./pp watch_score --secs 120      # the same, kept in recordings/watch_score.log as durable evidence
./pp republish_best --yes        # if the broker lost the retained value: re-send the best once (dry run without --yes)
```

## What it shows and records

- **The screen**: the table in perspective with the ball and its shadow, your paddle and the computer's, the amber
  reach oval, the streak and best, km/h of your last shot and its quality, and your camera picture (mirrored) in the
  corner with a ring on your tracked hand.
  A strip chart shows the **IMU swing rate with the threshold line** ("CAMERA SWING" when the camera is the
  sensor); **X** adds the x-ray gate list.
- **Leaderboard**: every finished live game is saved to `data/pingpong.db` (Rally by best streak,
  Match by wins); the end screen shows the top five and highlights you.
- **Recordings**: each live session writes `recordings/<time>-<player>/` (`session.json`, every raw IMU
  sample, every pose, every game event with all six gate results). `./pp report` turns it into numbers
  (hits, timing, which gate rejected what, hub rate and worst gap, pauses, loop time).
  `./pp replay <session> --set level.late_s=0.4` re-runs the recorded sensor data through the real code
  with changed settings, so a tuning question never needs another round of swinging.
- **Sounds**: a pop whose pitch tells you the hit quality, a buzz for a miss, arpeggios for a point or
  record, countdown ticks (`--no-audio` to turn off). Use the Mac's built-in speakers: Bluetooth headphones add
  150-250 ms of delay (the cues would arrive after the swing) and share the radio with the hub. `--no-record` and `--no-store` switch the
  recordings and the leaderboard off.

**Tuning without editing code.** `./pp play --set level.radius_sw=0.8 --set level.late_s=0.25` changes the hit
radius or window for every level (tags and keys keep it), `--set swing.t_pk=150` the weakest swing that counts,
`--set judge.d95_s=0.2` how long after the window a miss is declared. The names are the fields of the level table
in `pingpong/levels.py`, the detector's `SwingParams` and the judge. `./pp report` lists the settings a session ran
with, and `./pp replay` of that session starts from them (`--set` there wins), so the loop is: play, read the
x-ray / report ("hand 0.62 SW from the ball (limit 0.45)"), try a number, replay it on the same swings.

**How soft a swing can be.** The weakest swing that counts is 0.42 x the strength of your calibration's soft swings
(the line on the IMU strip chart; `--set swing.t_pk=N` sets it in dps). It used to be 0.7 x, until two real games showed
that half of the balls I missed had a swing of 210-300 dps right at them, gentler than the "soft" swings I had calibrated
with (458 dps), so the threshold dropped them (`tests/test_play_swings.py` keeps six of those swings). The price: a big
wobble of the hub or a hesitation before a stroke can now count as a weak swing too (the judge ignores a swing when no
ball is near and takes one hit per ball, and a backswing still never counts). The shake lock is unchanged, and so is the
camera swing detector, which was only ever tuned on scripted hands.

**Tuning the levels.** `./pp sim` plays the whole game loop with a scripted player and prints, per level and
swing strength, how often the computer misses a ball, how often you would fault, and how long rallies last
(`--quality 0.4` for a sloppier player: hard swings start to fault where the level's threshold is low).

## Demo day: the graded take

**Ten minutes before:** hub charged and awake, Continuity Camera off, the cards on a stand inside the camera frame,
`./pp ready`, `./pp env_check`, one rehearsal with `--no-publish`. See what the broker holds right now:

```bash
mosquitto_sub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi --retained-only -C 1 -W 5
```

**The take** is a fresh session and the last one that publishes: `./pp play --player rafae --level 2`. In a second
Terminal keep `./pp watch_score --secs 600` running as durable evidence, screen-record (Cmd+Shift+5) and film the
haptic with a phone. A crash mid-take: restart with `--resume` so the best so far stays on the broker.

**After it** use only `--no-publish` (a rehearsal would publish its own `1.0, 2.0, ...` over the graded value; the
lobby warns you when the broker holds one). If the broker lost the value: `./pp republish_best --yes`.

**Five-minute live demo:** subscriber window visible; START card, then level cards 1 → 2 (the flight visibly
shortens); one Rally with the x-ray (**X**) and km/h; a Match at Club to 7 (or the first three points); the
haptic on camera; `./pp play --board`. The recorded video is the fallback.

## When something goes wrong

| Problem | What the game does |
|---|---|
| hub goes silent | the ball clock **pauses** (never a miss or a fault) and resumes where it was; it tries to reconnect (with the camera as the swing sensor the game carries on: only the haptics go quiet) |
| the hub's IMU is too slow (< 25 Hz at the bench) | the camera's hand speed detects swings instead (`--swing-source`); spin is off then |
| you leave the camera view for more than 0.6 s | paused, with "PAUSED: pose lost" on screen |
| tags unreadable | SPACE starts, 1–3 set the level |
| broker unreachable | the game runs; the score is sent when it reconnects. The public broker drops a connection attempt now and then (3 of 4 in one probe), so `MQTT OFFLINE` for the first seconds is normal: it retries every 1-5 s |
| motors weak | beep + light carry every cue; `D` disarms the motors |
| someone shakes the hub | an FFT of the last second detects it and locks the paddle for a second |

## Project layout and tests

```
play.py  config.py  pp  requirements.txt  README.md
pingpong/   the game: sensing (hub, imu_worker, swing, shake, vision, pose, tags), game (judge, shot, physics,
            rules, policy, pd, qbandit, levels, spin), output (haptics, feedback, audio, hud, canvas), glue (live, app, profile,
            spinflow, store, recorder, replay, sessionreport, overrides, livebuild, posegyro, fakerig, threadrig,
            sources_fake), the hand (body, posemodel, posetrain, posetake)
tools/      scan_hubs  env_check  bench_hub  bench_cam  bench_haptics  calibrate_swing  reset_hub
            report  replay  train_spin  train_pose  sim  watch_score  republish_best  make_cards
tests/      one file per module; the whole pipeline also runs on fake hardware (test_fakerig.py)
docs/       PLAN.md (the full design), JOURNAL.md (one line per surprise), diagram.md, cards/
data/ recordings/ calibration*.json config_local.json   (never committed: players, videos, measurements)
```

`./pp ready` runs the import check, every test and every tool's `--selftest`, all without hardware.
The same scripted player that tests the game also plays through the *real* HubLink parser, swing
detector, vision worker, tag voter and haptics on a simulated clock (`./pp play --selftest`), and
`tests/test_threadrig.py` plays it again in real time with the real threads (a notifier thread at 66 Hz, a camera
thread blocking at 30 fps, the IMU and actuator workers), which is what finds deadlocks and races.

## Questions

> **DRAFT: written by Claude from the code as it stands. Rewrite (a), (b) and (c) in your own words
> before submitting, and replace every `[measured: ...]` with your own bench numbers.**

### (a) Describe the policy: how does it make decisions?

My game is a stack of small explicit rules that I can test without hardware. **Perception** turns raw
sensors into events: a state machine projects the hub's gyro onto my learned forward-swing axis, arms
on a threshold, fires at the peak, and ignores backswings and waving (when the hub is too slow the same machine
runs on the camera's hand velocity); AprilTag ids must be seen in 4 of
6 frames (START also held 0.4 s) before they count; pose is used only when the landmarks are visible.
**The hit judge** is a conjunction of five deciding gates: the swing's contact (its back-dated peak plus the stroke's
length, less the time the hub took to tell me) inside a window around the moment the ball passes my paddle, my hand
(from pose, shifted by the measured camera lag) level with the ball across the table within a level-dependent reach
around the impact and at the impact itself, the swing big and clean enough, one hit per ball, and no shake lock. A sixth gate only measures and logs whether the camera and the IMU saw
the swing at the same moment. Each gate's result and reason appears on screen. **The shot policy** turns a valid
swing into a shot: peak gyro rate sets ball speed, my hand position sets aim, a trained spin model (if I
made one) sets topspin or backspin, and a deterministic risk rule (speed × sloppiness against a per-level
threshold) decides net or out faults. **The computer**
picks a target zone by a softmax over a utility that wrong-foots my tracked hand (plus, if I turn it on,
what a Q-learning table has learned about where I fail), then moves a speed-limited paddle to the landing
point with a PD controller after its reaction delay, and in Match misses with a probability that grows with my shot's
speed, spin and the distance its paddle cannot cover; in Rally it never misses and only the ramp
(speed ×1.03 per hit, up to 1.8, a special ball every tenth) changes. A rules state machine turns hits
and faults into the score; sensor loss pauses the game rather than scoring against me; the record of
continuous hits goes to MQTT whenever it improves.

### (b) What are the potential limitations of your game?

- **Camera lag and one camera.** Pose trails the IMU by roughly 70–150 ms [measured: ___ ms] and a
  single webcam gives no depth, so "the paddle is at the ball" is judged across the table only and the paddle's depth
  comes from my hand's height rather than from how far forward I reach; the screen does not hide the camera's delay for
  the hand (extrapolating its speed was no more accurate on my recordings and made the paddle shimmer), so the paddle trails
  my hand by the camera's lag. The hand also moves during a swing, so the judge looks at where it is at the impact and how
  close it came just before.
- **Delays I could only estimate.** The hub's transport (~40 ms), the length of a stroke (0.20 s measured on 28 swings), the screen (~50 ms),
  the speakers (~25 ms) and the motors (~50 ms) cannot be measured with this hardware (only the camera against the hub
  can); they are typical values, and every one is a knob (the stroke's length, 0.20 s, is measured on my swings).
- **The hub IMU.** About 64 Hz over Bluetooth [measured: 63.9 Hz, worst gap 93 ms] (below 25 Hz the camera
  takes over, see below), undocumented units (measured: 0.99 gyro counts per deg/s, about 1017 accelerometer
  counts per g, and the accelerometer saturates at about 8 g in a hard swing, which clips the spin features),
  no timestamps (samples are stamped on arrival), so the swing peak is only good to ~15 ms. My real swings
  build up for about half a second and the detector waits for the rate to fall back from the peak, so a
  swing is reported 33–150 ms after it happened. Shot speed is a calibrated relative measure, not true racket speed.
- **The camera as a swing sensor (the fallback).** It sees only the hand's motion across the picture, so a
  swing straight at the camera barely registers, and a quick sideways reposition of the hand can look like a
  swing (the duration limits and the judge's timing and pose gates stop most of those, not all). It runs at 15-30
  Hz, its peak is about 30 ms later than the hub's, the camera lag cannot be measured against an IMU it replaces,
  there is no accelerometer so there is no spin, and a very hard swing carries the hand out of the ball's radius.
  Best at Rookie and Club. The camera detector and its shake lock keep the settings tuned on scripted hand
  motion; they have never seen a real camera recording.
- **Haptics.** The motors are weak unless the hub has some inertia on it; their vibration shakes the
  hub's own IMU, so the detector ignores a short window after every pulse [measured: ___ ms]; the felt
  pulse arrives 100–200 ms after the swing peak, so timing cues are dropped at the fastest levels.
- **The game itself.** Arcade ball physics, not validated table tennis; Pro is best-effort; Match is
  receive-only (the computer serves every point); calibration and tuning come from one player in one
  room, so another person needs a quick re-calibration.
- **The environment.** Pose needs front light and my upper body in frame; tags need a matte print and
  a white margin; the score goes to a public, unauthenticated broker, so it can be dropped or spoofed,
  and my reading of "record" (best streak this run of live, unassisted hits) is an interpretation.

### (c) Which AI/ML algorithms did you use, and how do they work? (two sentences or fewer each)

| Algorithm | Where | How it works |
|---|---|---|
| MediaPipe BlazePose landmarker (a pre-trained CNN) | `pose_features.py`, `pose.py`, `vision.py` | A convolutional network regresses 33 body landmarks per frame and tracks them between frames. I use the hand (the index, pinky and thumb points, not the wrist), normalised by shoulder width, as the paddle position. |
| AprilTag / ArUco 36h11 detection | `tags.py` | It thresholds the image, finds square quads and decodes a Hamming-protected bit grid into an id and corners. I vote over frames before an id counts as START or LEVEL. |
| One-Euro filter | `oneeuro.py` | A low-pass filter whose cutoff rises with signal speed. The paddle point is smooth when I am still and nearly lag-free in a swing. |
| Signed-axis swing detector (a threshold state machine; the axis comes from an SVD) | `swing.py`, `calibration.py`, `posegyro.py` | It projects the gyro onto the forward axis learned from my calibration swings, arms on a threshold, tracks the peak and fires on the falling edge with an oscillation guard. A backswing projects negative, so it never fires; with a slow hub the same detector runs on the camera's hand velocity. |
| FFT shake discriminator | `shake.py` | The last 1.5 seconds of gyro are resampled and run through `rfft`; a narrow, strong peak between 1.2 and 8 Hz (my real shakes and waves were 1.6–2.4 Hz, my swings repeat at 0.4–0.8 Hz) with several full cycles is a shake. It locks the paddle for a second, while a single swing's smooth spectrum does not. |
| PD controller | `pd.py`, `policy.py` | Its velocity command is Kp·error + Kd·(filtered error rate), saturated at a per-level paddle speed and only starting after a reaction delay. Whether the computer reaches a ball is simulated physics, and the screen draws the paddle chasing (or missing) my shot. |
| Softmax (Boltzmann) policy | `policy.py` | Each of nine target zones gets a utility and is sampled ∝ exp(utility / temperature). Lower temperature plays sharper at higher levels. |
| StandardScaler + Logistic Regression | `spin.py`, `spinflow.py` | It standardises 12 swing features (the unit directions of the gyro peak, the net rotation and the linear acceleration, plus peak rate, duration and backswing ratio) and learns a linear softmax boundary between flat, top and back from my own labelled swings. The three probabilities become continuous topspin or backspin, and the model only ships if its cross-validated accuracy is at least 75%. |
| Tabular Q-learning (opt-in, `--learn`) | `qbandit.py`, `policy.py` | It keeps Q(s,a) for nine states (my hand's third of the reach box × the column of the zone it served last) and nine target zones, and after every ball updates Q += α(r + γ·max Q' − Q) with reward 1 when I miss or fault. The computer adds Q to its softmax utility with a weight that grows with the level, so it learns to serve where I am weakest. |
| Cross-correlation | `benchstats.py` | Hand speed from the camera is correlated against gyro magnitude over a hand wave to find how much later the camera sees the same motion. That lag aligns the two sensors. |

Everything in this table is built and tested. The two *learned* pieces, the spin classifier and the Q-learning opponent, are per player and optional: the game plays without either (flat balls, a fixed opponent).

### Reflection questions (Notion template)

Write these in your own words; `docs/JOURNAL.md` has one line for every surprise, failure and fix as raw
material.

- **What we learned.** [your words]
- **What we are most proud of.** [your words]
- **What took a while (was tough).** [your words]
- **Most spectacular failure.** [your words]

## Note on versions

The versions in `requirements.txt` are pinned on purpose (the same as `P1-Pose-Race`): mediapipe 0.10.21
needs numpy < 2 and so OpenCV 4.10.0.84, it has no Python 3.13 wheel, and newer MediaPipe aborts on
Apple silicon. `pip check` prints one expected message about mediapipe's platform; gate on `./pp ready`
instead.
