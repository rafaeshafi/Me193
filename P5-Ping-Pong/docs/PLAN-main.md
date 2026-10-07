# P5-Ping-Pong — Implementation Plan v3 (re-scoped for a Sat 10/10 deadline, schedule-reviewed)

Student: Rafae Shafi · ME193 AI & Robotics (Prof. Rogers) · written Tue 2026-10-06 (Day 0)
Status: **awaiting your approval. Nothing is built; no project files exist yet.** Review history: v1 7-day plan (18-agent research/design/review) → v2 compressed to Sat 10/10 → **v3 after two adversarial reviews** (schedule realism: "not realistic as written" → re-planned at 4.5 h/day with night shifts, clock ladder and a complete G0; consistency: 10 major + ~20 minor issues, all applied) and the **P4 name collision** (renamed P5). Three things I need from you: P4's status/due date, the exact due date/time, and a yes/no on the night-shift permission allowlist (§0.5, §10).
Evidence labels: (V) verified in research or by a reviewer's local test · (E) estimate settled by a named bench · (U) unverified, has a probe or fallback.
Full reference design (667 lines, from an 18-agent research + adversarial-review run), persisted next to this file with an OVERRIDES header: `docs/PLAN.md` (also in the workflow journal `…/subagents/workflows/wf_67349f48-b16/journal.jsonl`). On approval it is copied to `~/ME193/P5-Ping-Pong/docs/PLAN.md`. Where this file and that one disagree on **anything** (schedule, scope, haptic coding, gate names, spin classes, flags, file tree, acceptance lists), **this file wins**; the appendix is a formula / pseudocode / DB-schema reference only.

---

## 0. What you are approving (read this first)

1. **The game — "Tiered Hub-Paddle Pong".** One Python process on the Mac (the P1 package pins) does everything on the required path. The **LEGO Double Motor is the paddle**: its IMU says *when and how hard* you swung; its motors/beep/light are the **haptics** (the "fun/cool" item). **Webcam pose says *where* the paddle is.** **AprilTags** start the game and set the level (ball speed). The score goes live to `ME193/Rogers/RafaeShafi` as a float. The ball is virtual, drawn over your webcam feed in one OpenCV window.
2. **Schedule, honest version.** Day 0 = tonight. Build days Wed 10/7 – Fri 10/9, **a complete minimum submission exists by Friday night**, polish + submit by **Sat 10/10, 6 pm**, Sun 10/11 is real buffer only if that is the actual deadline. Planned at **~19 h of your time (4.5 h/day, inside your 4–6)**; Claude works **named night shifts** between your blocks. An adversarial schedule review rated the first draft (5.5 h/day, G0 on Thursday) at only ~35–45% for G0 and ~20–25% for a full Saturday package; with the changes below the same reviewer's judgement is **~70% for a complete minimum by Friday night and a polished package by Saturday 6 pm** (judgement, not measurement).
3. **Scope.** Your headline asks (per-shot speed + spin, profiles + leaderboard, Survival, difficulty-selectable Match) are **targeted** (SHOULD, gate G1), **ordered by value and time-boxed**, and degrade in this order if time runs out: Q-bandit → softmax only, spin → flat, Match difficulty → one level (§2). Nothing is cut at 11 pm by accident. **Parked for after submission:** UNO Q + MPU-6050 + LDR + resistor (matrix scoreboard, cover-to-serve, bench ruler), web leaderboard page, tag login, Insane level, 5-class spin, whistle serve, player-serve. Reason: the assignment says "double motor *or* UnoQ", the UNO Q path carries the most untested risk (hotspot, Bridge, wiring, container Bluetooth), and there is no slack. **Tell me if you would rather keep the UNO Q scoreboard and drop something else.**
4. **Locked decisions (yours):** topic `ME193/Rogers/RafaeShafi` · score = **best streak this run** (ticks live while building a new best, holds after a miss; one config switch changes it) · demo **live and recorded** · Match is **receive-only** (computer serves every point) · light sensor is an LDR (parked).
5. **On approval, tonight (Claude's Tue-night shift), in this order:** P5's own `.gitignore` first, before any `git add` (`data/`, `recordings/`, `calibration*.json`, `__pycache__/`, `*.pyc`, `my_env/`) → `tools/make_cards.py` (so you can print tonight) → `env_check` → scaffold `~/ME193/P5-Ping-Pong/` + venv from the P1 pins + the `./pp` wrapper + `pingpong/__init__.py` + `pytest.ini` (`pythonpath = .`) → copy-with-edit of `pose_features`, `find_apriltag`, `PD` → freeze `events.py` (renamed from `types.py`, which would shadow the stdlib module), `clock.py` (Clock + FakeClock), `config.py` → bench tools (each with a `--selftest` on a fake transport, each writing its own `config_local.json`, each starting with a `sys.path` shim) → start Wave 1. The appendix is copied to `docs/PLAN.md` **with its OVERRIDES header**: judge gates are J1–J6 (the appendix's G1–G6); milestone gates are only G0/G1/G2; early/late haptics are coded by pulse count, not left/right motor; a "good" hit has no motor cue; spin default is 3 classes; metronome offset is optional (default 0.0); level tags are 1–3; UNO Q / LDR / MPU / webboard / boardlink / whistle and the appendix's Day 6–7 are PARKED; `--mqtt off` is an alias of `--no-publish`; the folder is `P5-Ping-Pong`; §5 and §10 here replace the appendix's §11 and §13. The module table is the task-card list (I run `superpowers-writing-plans` in parallel, not before); every fan-out task card carries this boilerplate: *no Co-Authored-By trailer (`attribution.commit` is unset; your `~/CLAUDE.md`), every file <500 lines (`tests/test_file_sizes.py`), pre-commit check = `./pp ready`.* I also propose a small **permission allowlist** (edits inside `P5-Ping-Pong/`, `my_env/bin/python`, `pytest`, `git add/commit` with pathspec) so night shifts never stall on a prompt — **you approve that separately**; I will not change settings on my own.
6. **You do tonight (≤1.25 h) — §10.** The first real hardware contact is `./pp env_check` in Terminal.app: it also measures the biggest unknown (hub IMU rate at the play position) **12 hours earlier than the old plan**.
7. **Found while checking the repo — please read.** Tonight at 19:56 a commit added **`P4-Door-To-Door-Service/`** (two UNO Q Arduino apps + a minifig tracker; its README lists the laptop tracker `live_tracker.py` as still missing). So this project is named **`P5-Ping-Pong`** here (rename if the course numbers it differently). **If P4 is still open, it competes for the same Wed–Sat hours and the 4.5 h/day budget in §5 is wrong** — tell me P4's due date and how much is left; if it is due before Saturday I will re-cut §2 before we start. P4 also keeps the UNO Q busy (one app at a time), one more reason the UNO Q stays parked.

---

## 1. Context

Assignment (Prof. Rogers): *play virtual ping-pong with your double motor or UnoQ — pose for paddle location, IMU for swing, AprilTag for player level (ball speed) and game start, score posted in real time to `ME193/Rogers/YourName` as a float = the record number of continuous hits; use everything learned in class; do something fun not done in class (e.g. haptics); answer (a) describe the policy, (b) the limitations, (c) which AI/ML algorithms you used and, in ≤2 sentences each, how they work; answer the template reflection questions (What we learned / What we are most proud of / What took a while (was tough) / Most spectacular failure).*

Your ambitions ("make it insane"): control ball **speed and spin on every shot**, **player profiles + leaderboard**, **Survival** (computer never misses; how long can you rally), **full Match vs the computer with a choice of difficulty**, and IMU + pose visibly showing you hit the ball.

Why this architecture (all (V) unless marked): the Double Motor IMU is available over BLE at ≤ ~66 Hz (15 ms notify) with undocumented units and no timestamps, so everything is calibrated at the bench; `blocking=False` still blocks the caller on the BLE write, so BLE lives on its own thread; haptic pulses shake the hub's own IMU, so the swing detector is blanked around each pulse; the P1 venv pins (mediapipe 0.10.21 → numpy<2 → opencv 4.10.0.84, Python 3.12) already import pose + `cv2.aruco` + paho + bleak + legoeducation and must be reused; pose stays on the Mac (no measured board FPS). **Operational facts from your own session history (V, by the schedule reviewer):** Claude's shell cannot open the camera (macOS only lets *your* terminal use it). **Correction verified 2026-10-06 20:34:** in *this* app (Claude desktop) the shell cannot use Bluetooth either (CoreBluetooth aborts the process, exit 134, TCC "responsible process: claude"; earlier sessions that could scan used a different host). So **every BLE and camera run is launched by you in Terminal.app** with output tee'd to `recordings/*.log`; the tools refuse politely from this app (`pingpong/hostcheck.py`). Your first P1 live run lost ~20 min to the wrong interpreter — hence the `./pp` wrapper that always uses `my_env/bin/python`.

---

## 2. Scope tiers

| Tier | Features |
|---|---|
| **MUST = G0, a complete *submittable* minimum** | Pose paddle gate (reach box + a fixed camera-lag constant) · IMU swing detection (signed axis) · START + LEVEL tags (Rookie/Club; Pro as a config row) · live MQTT best-streak float · H0 haptics = hit pulse + miss buzz + beep + light · **per-ball JSONL log** + `--fake` Rookie rally · **GitHub push that clones and runs** · **Notion v0 page with video + README answers draft** |
| **SHOULD** (in value order, each time-boxed on Friday) | Speed calibration + km/h + shot map → Survival ramp → Match to 7 (Rookie/Club, Pro best-effort) → profiles (`--player NAME` creates/loads; guest = quick-tune; calibrations persist) + SQLite + end-screen top-5 (`--board` prints it) → core haptic language (perfect, early/late by pulse count, fault, record; a "good" hit = beep + light only) + countdown → **spin 3-class** (only if the items before it are green at the block midpoint) → **Q-learning bandit** (only if everything above is green with 1 h left; otherwise (c) says it was not built) → wave-test lag + metronome offset · 15-min soak · `--fake` Survival/Match |
| **CUT NOW** (kept as config rows or parked) | Insane level + tag 4 · Pro tuning · audio beyond 3 clips · 20-trial blind-ID gate (do 10, report the number) · `register_player` / leaderboard CLI · tag login · web board · guest session (optional Saturday, `--no-publish`) |
| **PARKED** (post-submission, design retained in the detailed doc §3.2, §4.4) | UNO Q Arena Node · LDR cover-to-serve · MPU-6050 bench ruler · whistle serve · player-serve · SPIN DIAL tag |

**Cut order (first dropped → last; each cut keeps the demo working):** every CUT-NOW item → Pro tuning → Q-bandit (then say so in (c)) → spin 3-class → flat → profile polish (keep guest + one named profile, keys only) → full haptic language → 4 core patterns (ready, countdown, hit, miss; the blind-ID test is then dropped) → Match difficulty variants → one difficulty → Survival ramp extras (keep the basic ramp).

**Never cut:** pose gate · IMU swing gate · tag START + LEVEL · live MQTT best streak · H0 haptics · per-ball log · `--fake` · written answers · **the two explain-backs** (the assignment is "demonstrate that you understand").

---

## 3. Architecture

```
 PLAYER: dominant fist = Double Motor (paddle); off hand = AprilTag cards (or a stand in frame)
   | webcam                                   | BLE: IMU 15 ms notify in, haptic cmds out
   v                                          v
+--------------------- MAC: one Python process (P1 pins, Python 3.12) ---------------------+
| [V] VisionWorker: newest frame, t_read stamp, unmirrored 640x360 -> MediaPipe pose       |
|     -> One-Euro -> immutable PoseSnapshot (tuple swap, no shared deque)                  |
| [T] TagWorker: 960 px gray, LOBBY only -> TagVoter (4 of 6 frames; START held 0.4 s)     |
| [H] legoeducation loop thread: callback ONLY parses + stamps monotonic_ns -> queue       |
| [I] IMU worker: blocks on queue -> SwingDetector -> SwingEvent -> main queue             |
| [A] Actuator: sole sender of hub commands, <=10 writes/s, reports real write times       |
| MAIN ~60 Hz: judge -> shot -> Leg physics -> Opponent -> Rules/Modes -> Haptics -> HUD   |
|      |                      |                       |                                    |
|      v                      v                       v                                    |
| [Q] paho thread      [S] StoreWriter (SQLite)   [L] per-ball recorder (JSONL)            |
+------+----------------------+-----------------------+------------------------------------+
       v                      v                       v
test.mosquitto.org:1883    data/pingpong.db        recordings/<session>/<ball>.jsonl
ME193/Rogers/RafaeShafi = "12.0"
```

Rules: only queues, immutable snapshots and locked scalars cross threads; the main thread never calls BLE/MQTT/SQLite synchronously; `cv2.VideoCapture` opens on the main thread from the terminal that was granted Camera/Bluetooth; **every tool installs SIGTERM + SIGINT handlers** so a timeout, `pkill` or Ctrl-C still runs teardown (a killed process leaves the hub "connected" ~24 s and invisible to the next scan); teardown order = stop+join actuator → `cancel_batch()` → `motor_stop(MOTOR_BOTH)` → `disconnect()`, each in its own try/except.

**Per-ball recording is a G0 item (~45 min of Claude time, zero student time):** every ball writes `recordings/<session>/<ball>.jsonl` (IMU window, pose snapshots, judge gate verdicts, timestamps); `play.py --imu replay:FILE` replays it. This turns every failed live attempt into an offline fixture so Claude can reproduce a problem with a failing test instead of relying on a verbal description.

**One shot, end to end.** CPU launches a ball (exact arrival time `t_c`, analytic flight) → you swing → `SwingDetector` emits IMPACT with a back-dated peak time `t_i` (known 35–80 ms after the true peak (E)) → **judge** checks timing, pose, swing, refractory → valid hit: `shot.make()` turns peak gyro into speed, features into spin, hand position into aim, and a deterministic risk rule into net/out faults → outgoing flight, haptic + sound fired ~40 ms early, streak/record updated, MQTT published → CPU replies (Survival always; Match per miss model).

**Degradation ladder.** Broker down → QoS1 queue + republish on reconnect. Pose lost >1.5 s → PAUSED (never a hit or fault). Hub silent beyond the measured p99.9 gap → ball clock stretches, then PAUSED + reconnect with 15 ms. Tags unreadable → keys (SPACE start, 1–3 level, M mode). Motors weak/dead → beep + light, then Mac sound + screen shake. Hub IMU too slow (<25 Hz) → pose wrist-speed swing detector, Rookie/Club only.

---

## 4. Hardware (required path has no wires)

| Item | Role |
|---|---|
| Mac (FaceTime camera, speakers, BLE) | Everything required; sole MQTT publisher; SQLite host. Laptop on a table, lid tilted to frame head-to-hips of a player 1.5–2.0 m away (floor mark at 1.8 m), front-lit, no window behind; needs ~2.5 m of room depth. |
| LEGO Double Motor | **The paddle.** Hold in the dominant fist; tape an arrow on the casing so grip orientation repeats; wrist lanyard. Weight is 80 g vs 301 g in conflicting sources → weigh it. Optional shaft masses (coin/nut/LEGO wheel, **taped and axle-stopped**) make haptics felt; if bare shafts are too weak, strap the hub to the forearm and store the offset in the profile. Lanyard on **every** player including guests; fingers clear of the shafts; if the hub weighs >150 g a guest turn is ≤5 min. Log battery % in every bench; charge between sessions (hub auto-sleeps; wake with its button). |
| Printed AprilTag cards (36h11) | **0 = START, 1 = Rookie, 2 = Club, 3 = Pro.** Tag 15 cm + 2 cm white margin = 19 cm (fits Letter; P2's `make_tag` pad would print ~21 cm, so use `tools/make_cards.py`). Print at **Actual size** and check the tag edge with a ruler (15 cm ±3 mm); matte, on foam board. Phone screen at full brightness is the fallback (inverted detection on). Cards must be **inside the camera frame** at 1.8 m (hold-up with the off hand, or a stand at player distance). |
| Camera selection | OpenCV on macOS picks by index. First turn **Continuity Camera off on the phone** (Settings → General → AirPlay & Handoff) and use the built-in index; only if that fails add `pyobjc-framework-AVFoundation` at the installed pyobjc 12.2.2 to the pins. |
| Built-in Mac speakers | Game audio (Bluetooth audio adds 150–250 ms and shares the radio; keep AirPods disconnected). |
| Parked: UNO Q, MPU-6050, LDR, one resistor, hotspot, jumpers | Post-submission; nothing to order for the build days. |

**Session ritual (README checklist, ~5 min, every block):** tilt + floor mark + front light → wake hub → run under `caffeinate -dims` with Focus on → `./pp ready` (pytest + selftests) → capture fixtures / run the 60 s quick-tune **first** (arm fatigue drifts calibrations; rest between sets).

### Bench probes (Wed; every bench writes its own `config_local.json`, no hand-copied numbers)

| Probe | What | GO | If not GO |
|---|---|---|---|
| **P1 hub rate at the play position** (first run **tonight** inside `env_check`, full version Wed) | 30 s stream at `notify 15 ms`, then 20 swings at 1.8 m, hub in fist, body between hub and Mac; Hz, worst gap, gap histogram; repeat with camera + pose + MQTT running | ≥40 Hz | 25–40 Hz: NOTIFY_MS 20/30, windows ×1.3. **<25 Hz: pose wrist-speed detector** (hub kept for haptics + gate), Rookie/Club only — decided by **Wed +1:15** |
| P2 units | 6-face rest test; three 360° turns | \|a\| CV ≤3% → `ACCEL_PER_G`; `GYRO_PER_DPS` ±5%; yaw sign | thresholds fall back to fixture percentiles |
| P3 clipping/axes + fixtures | 10 max-effort swings; **20 soft + 20 hard + 10 backswing-only + 10 shakes + 30 s waving** | plateau count sets `HUB_FS_RAW`; PCA forward axis `u_fwd` | clip flagged on plateau only |
| P5 weigh/mount | kitchen scale; fist vs forearm; **20 hard swings with masses fitted** | mass + mount chosen and nothing loosens | drop the masses / forearm strap |
| P6 haptics (reduced) | 3 recipes × mass/no mass (also `motor_set_duty_cycle`), felt rating 1–5 + IMU accel RMS during the pulse | ≥1 recipe felt 3/5; `BLANK_AFTER_PULSE` = RMS-return + 20 ms; cap ≤10 writes/s | beep + light + Mac sound primary; say so in the write-up |
| P8 vision (`bench_cam`, merged with the MQTT echo + walking skeleton) | pose FPS, printed-tag hit rate at 1.8 m (hold-up vs stand), camera-vs-IMU lag by wave cross-correlation (becomes the Wed constant), **hub Hz + worst gap while camera + pose + MQTT run**, HUD legibility on the 14" screen at 1.8 m | ≥20 fps; tags ≥90% of frames; HUD readable | larger cards / ROI / external display or play at 1.5 m and re-measure tag range |
| P9 MQTT | retained QoS1 `0.0` to the scratch topic `ME193-pp/RafaeShafi/selftest` **and once to the official topic, then cleared** with `mosquitto_pub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi -n -r` (with your OK at the time), so Thursday's first official publish is not the first ever; **repeat from the venue/campus network and from the iPhone hotspot** (outbound 1883 is often blocked on campus Wi-Fi); a 60 s `mosquitto_sub -t 'ME193/Rogers/#' -W 60` to see classmates' payload format | round trip <2 s on the network you will demo on | blocked 1883 → tether to the hotspot; a local `mosquitto -p 18831` + `PP_BROKER` keeps the *demo* alive but cannot satisfy the instructor's subscription |
| *Cut / folded:* P4 gestures, P11 audio latency (no decision depends on them) · **P7 batching = a `--selftest` in `bench_haptics`** on the fake transport (a L+R batch packs into one write; the same motor twice raises) · P10 UNO Q probes parked | | | |

---

## 5. Schedule

**Principle.** The real critical path alternates *student block → Claude night shift → student block*. Each night shift has a named deliverable due **before** your next block, and **a block starts only when `./pp ready` is green** (pytest + tool selftests + `tests/test_contracts.py`, which checks that every module imports and dataclass fields match `events.py`, so interface drift between parallel agents fails in seconds). One worktree, disjoint file ownership per agent. Hours below are **your hands-on time**, planned at 4.5 h/day; mark your real free blocks Wed–Sat on a calendar tonight — **any block under 4 h triggers the §2 cuts that morning.**

**Clock ladder (offsets from the start of that day's block; replaces the old "2× estimate" rule):**
- Wed +1:15 — hub rate <25 Hz → pose wrist-speed detector (Rookie/Club only).
- Thu +0:30 — live swing events not ≥8/10 clean, or any backswing IMPACT → same fallback.
- Thu +1:45 — tags unreadable at 1.8 m → keys + larger cards; show the tag close-up in the video.
- Thu +2:30 — a Rookie ball cannot be hit ≥5/10 → Rookie radius ×1.5 and windows ×1.5, stated in (b).
- Fri +3:00 — G0 still failing → whatever passes is the submission; only speed and the Survival ramp (0.5 h each) may be added.
- Fri with 1 h left — anything unfinished on the Friday list is cut.

**Night shifts.** Tue = `env_check` + cards tool + scaffold + `clock.py` + bench tools (+ pose-fallback detector if tonight's rate <25 Hz). Wed = Wave 1 live wiring + swing detector tuned on Wed's real fixtures that same evening, then Wave 2 SHOULD modules. Thu = judge/swing retuned from Thursday's per-ball logs + Wave 2. Fri = fixes + docs.
**Waves.** *Wave 1 (G0 path only, due Wed afternoon; `pose_features`, `find_apriltag`, `PD` are Tue-night copy-edits):* `hub`, `actuator`, `vision`/`pose`/`paddle`, `tags`, `swing`, `judge`, `physics`, `rules`-lite, `scoring`, `mqtt_pub`, `canvas`/`hud`, `play.py` live path, per-ball log. *Wave 2 (Wed evening → Thu night):* `shot`, `spin`, `policy`, `modes`, `store`, `leaderboard`, `audio` (≤3 clips), Q-bandit, `sim`.

### Day 0 · Tue 10/6 (tonight) — you ≤1.25 h
In this order: (0) **tell me whether P4 Door-To-Door is still open and when it is due** (it shares your hours); (1) **find the due date and time today** (assignment page, course Notion, or a classmate — 2 minutes; if Saturday, Friday's minimum *is* the submission and Saturday is polish; if Sunday, Saturday is polish and Sunday is buffer; if the hour is before Sat 6 pm, the Saturday block moves to the morning and guest session/Pro/final-video edit are cut); (2) send the §10 email (Claude drafts it); (3) wake the hub, read colour + serial off the card (tools connect **only** to the configured card — no "try all five", since 13–18 LEGO motors were in range in earlier scans); (4) from **Terminal.app** run `./pp env_check`: one camera frame, BLE connect/beep/disconnect on your card, one MQTT round trip, and **a 30 s hub IMU stream at 15 ms with the hub held at the play position** (prints Hz + worst gap; if camera access fails, grant Camera to that app and restart it); (5) print cards 0–3 on the Brother at Actual size, ruler-check, mount; (6) weigh the hub; find tape, two coins, a lanyard; (7) upload a ~20 MB test clip to the place the demo video will live and confirm it plays from a share link (Notion's free plan caps uploads at ~5 MB per file (U), so the video is linked, not embedded); (8) mark free blocks Wed–Sat and confirm a suitable room is free for each; (9) skim your class syllabus against the §9 crosswalk (2 min) and tell me any topic that is missing.
**Acceptance (Tue night):** `env_check` green for camera, BLE, MQTT and hub rate; cards 0–3 printed; every bench tool passes `--selftest`.

### Day 1 · Wed 10/7 — you 4.5 h, hard stop
- **+0:00–1:15** `./pp bench_hub --guided` (one connection, beep prompts, auto-advance): P1 rate at 1.8 m, P2 six-face + three turns, P3 ten max swings, then 20 soft / 20 hard / 10 backswing-only / 10 shakes / 30 s waving (BLE: run it in Terminal.app; Claude cannot use Bluetooth from this app). **Decision at +1:15:** rate ≥25 Hz, else the pose fallback.
- **+1:15–2:15** `./pp bench_cam` from Terminal.app: pose FPS, tag hit rate at 1.8 m, wave-lag cross-correlation, hub Hz + worst gap under camera + pose + MQTT load, HUD legibility, camera index, MQTT scratch echo + the one official `0.0` publish and clear.
- **+2:15–3:00** P5 + reduced P6: weigh, mount decision, haptic recipes.
- **+3:00–3:30** `python play.py --fake` Rookie playtest (mouse + space) for game feel and HUD feedback.
- **+3:30–4:30** buffer for re-runs. P1–P3 are blocking (no defaults for hub rate or units); everything else carries a default.
- **Claude in parallel:** Wave 1, then the swing detector tuned on the real fixtures the same evening, then Wave 2.
**Acceptance (Wed night):** imports + a dummy `detect_for_video`; all probe numbers in `config_local.json`; swing detector ≥19/20 on the real fixtures and **0 IMPACT on backswing-only**; `play.py --fake` reaches 10 hits at Rookie; ≥25 tests green incl. `test_contracts`.

### Day 2 · Thu 10/8 — G0 live, 4.5–5 h you (starts only when `./pp ready` is green)
Time-boxed in order (stop at the box; fallback in brackets):
- **+0:00–0:30** live hub + live swing events on the HUD only, no camera: 10 forward swings, 10 backswing-only, 30 s waving [<8/10 clean or any backswing IMPACT → pose wrist-speed detector].
- **+0:30–1:15** calibration, two steps only: reach box (4 corners) + 5 soft / 5 full swings (`omega_lo/hi`, `u_fwd`); camera lag = Wednesday's constant; **shoulder width captured standing still for 1 s** (feeds the pose lock); metronome offset = 0.0 until play shows otherwise.
- **+1:15–2:30** pose gate + START/LEVEL tags + a Rookie ball judged live; 3–5 tune loops on windows/radius from `recordings/` (Claude reads the logs, you swing) [tags unreadable by +1:45 → keys + larger cards].
- **+2:30–3:00** H0 haptics: hit pulse, miss buzz, beep + light.
- **+3:00–3:30** first official publish, `mosquitto_sub` check, retained late joiner, Wi-Fi-off reconnect.
- **+3:30–4:15** G0 acceptance (below).
- **+4:15–4:45** 60–90 s backup video, `git add P5-Ping-Pong/ && git diff --cached --stat -- P5-Ping-Pong` → commit + push + tag `v0-minimum`, **first fresh-clone run**, Notion v0 page (Claude fills via Playwright as in P3; you paste the video link), **explain-back #1 (25 min)**.
Checkpoints: G0a at +2:30 (IMU + pose + tags live, one ball judged), G0b at +3:30, go/no-go at +4:45. If G0b fails, Friday's first block (max 3 h) is G0 overflow.
**G0 acceptance (Rookie + Club):** (1) START tag → 3-2-1 → a ball; **(1b) START presented 3 times fires exactly 3 times; a 0.2 s flash and a card shown mid-rally never fire**; (2) a live streak ≥5 at Rookie, and `mosquitto_sub -t 'ME193/Rogers/#'` shows 1.0…N.0 equal to the HUD, holds after the miss, a retained late joiner returns the best; **(2b) 20 real swings give ≥19 IMPACTs and 10 soft swings ≥9**; (3) hand >2R away: 0 hits in 10 swings; hand on the ball without a swing: 0 in 10; hub shaken with the arm still: 0 in 10; **(3b) 30 s of 3 Hz waving with the hub gives 0 hits**; (4) **tags 1 and 2 each latch 20/20 at 1.8 m (tag 3 smoke-tested) and the logged flight-time ratio tag 1 : tag 2 is ≥1.3** (0.86 s vs 0.60 s = 1.43 before the shared 0.9–1.1 modulation, which cancels in the ratio); (5) a pulse is felt, or beep + light fire; (6) `--fake` publishes nothing to the official topic (automated test); (7) pytest green + a **5-minute** unattended soak running while you record + main-loop p95 <20 ms (`./pp report`) + Ctrl-C then hub reconnect <5 s; (8) README answers draft + Notion v0 exist; **(9) the full-screen large-glyph HUD is readable at 1.8 m** (external display/TV if not).

### Day 3 · Fri 10/9 — G1, 4.5–5.5 h you; every item has a box, stop at the box
Claude's Thursday-night shift has already built the modules, so your time is verification and tuning.
1. speed calibration + km/h + shot map **0:30** · 2. Survival ramp live **0:30** · 3. Match to 7 at Rookie/Club (Pro best-effort) **0:45** · 4. profiles: guest + `rafae` via `--player`, calibrations persist, SQLite end-screen top-5, restart test **0:30** · 5. core haptic patterns (perfect/good, early/late by pulse count, fault, record) **0:30** · 6. the five **failure drills** (hub asleep, tags unreadable, pose lost, broker down, motors dead — each ends with continue or a clean PAUSED + HUD message) **0:45** (moved from Saturday) · 7. spin 3-class **1:00**, only if items 1–4 are green at the block midpoint (12 swings/class, hold out 4/class; ≥9/12 held out to ship; below that it ships as flat and (b) says so; (b) states it is a 12-swing, single-player estimate) · 8. Q-bandit only if everything above is green with 1 h left, else (c) says it was not built · 9. around the rest: 15-min unattended soak while you write, **fresh-clone run** (20 min), **explain-back #2 (25 min)**, capture footage (Survival record rally, a Match point, a phone shot of the haptic, `mosquitto_sub` or a phone MQTT client on screen). Tag `v1-features`. **By the end of the last Friday block a submit-minimum exists** (Notion complete with video, GitHub pushed).
**G1 acceptance (reported, not all gated):** median soft/hard out-speed ratio ≥2.0 · Survival ramp visible and ends on the first fault · a full Match to 7 completes at Club · restart keeps profile + calibration · end-screen top-5 correct · spin accuracy and blind-ID (10 trials) are **reported**, not gated.
**Friday triggers:** after 3 h of the block, if the speed map + Survival ramp + Match are not live-demonstrable, cut spin to flat and cut the Q-bandit; after 5 h, scope freeze — any G1 line not passing is cut in §2 order. If G1 is incomplete Friday night, Saturday is final on whatever passes, bug fixes only.

### Day 4 · Sat 10/10 — final, ≤4 h, **submit by 6 pm, nothing new**
Already done by Friday night: submit-minimum, the five drills, a fresh-clone run, both explain-backs.
**+0:00–1:00** fix list from Friday's drills and soak · **+1:00–1:45** your own-words (a)(b)(c) + the four reflection bullets (What we learned / What we are most proud of / What took a while (was tough) / Most spectacular failure) from `docs/JOURNAL.md` (one line per surprise, failure or fix, kept all week) · **+1:45–2:30** final video cut from footage captured at G0/G1 plus one live graded take (the **last publishing run**) · **+2:30–3:00** README numbers (`./pp report`, Claude) + Notion final · **+3:00–3:30** fresh clone (`pip install -r requirements.txt` → `pytest -q` → `python play.py --fake`), push, tag `v1-submit-ready`, `mosquitto_sub` evidence, open the Notion page and the GitHub repo from a logged-out window, **submit**.
**G2 acceptance:** (1) the five failure drills each end CONTINUE or a clean PAUSED with a HUD message (re-run only if code changed after Friday); (2) a 15-min soak clean on the tagged build; (3) the fresh clone → `pip install -r requirements.txt` → `pytest -q` → `python play.py --fake` passes; (4) the retained topic equals the last live best; (5) every algorithm named in (c) is grep-able in tag `v1-submit-ready`; (6) Notion + README answers are in your own words; (7) the link is submitted.
Only if all green and time remains: a guest session — each guest runs `--player <name>` (a flat-spin profile), does the 60 s quick-tune (incl. the 1 s shoulder-width capture), then one Survival run, all with `--no-publish`; ask consent before filming; guest names and recordings stay in gitignored `data/` and `recordings/` and out of Notion and GitHub.
Guest, rehearsal, soak and drill sessions all run with `--no-publish`; the final graded take is the only publishing run after them. Keep the video and a `watch_score.py` log as durable evidence (test.mosquitto.org says not to rely on it); `tools/republish_best.py` re-sends the best value once.
**Live demo script (5 min, Mac only, if Prof. Rogers wants a live run; fallback = the recorded video):** 1) `mosquitto_sub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi` visible, check the retained value; 2) START tag, then level tag 1 → 2 (flight visibly shorter); 3) one Survival record rally with the x-ray HUD and km/h; 4) Match at Club to 7 (or the first 3 points); 5) the haptic shown on camera; 6) `./pp play --board`. **Pre-flight 10 min before:** hub charged and awake, Continuity Camera off, cards on a stand, broker reachable on the venue network, one `--no-publish` rehearsal, then the graded fresh session.

### Day 5 · Sun 10/11 — buffer (≤3 h; no new features before the submission is confirmed)
Real buffer only if the deadline is Sunday: re-record, local-broker swap rehearsal (`PP_BROKER`, <60 s). UNO Q work stays parked until the submission is confirmed — the board runs one app at a time, so deploying a ping-pong node would also stop the P4 minifig apps (and RateNgo); do that only after P4 is graded.

**Hour budget (your hands-on time):** D0 1.25 + D1 4.5 + D2 4.75 + D3 4.5–5.5 + D4 3.5–4.0 = **18.5–20 h**. Reading for the explain-backs is ~0.5 h/day on top. Claude's work is scheduled separately as night shifts.

---

## 6. Game design (condensed; full formulas in the detailed doc §6)

**Geometry.** Table 2.74 × 1.525 m, net 0.1525 m; arrival points on a 3×3 grid; hand maps from shoulder-width units to the calibrated reach box (frames fed to MediaPipe unmirrored; mirror only for display). Flight is an **analytic leg** (exact arrival time and point, stylised spin; no RK2/Magnus).

**Levels (starting values (E); `tools/sim.py` + play-testing tune them).**

| | Rookie (tag 1) | Club (tag 2) | Pro (tag 3, best-effort) |
|---|---|---|---|
| v_tier m/s (flight s at 3 m) | 3.5 (0.86) | 5.0 (0.60) | 7.0 (0.43) |
| window early/late (s) | 0.30/0.18 | 0.22/0.14 | 0.16/0.10 |
| pose radius R (shoulder widths) | 0.55 | 0.45 | 0.35 |
| CPU reaction τ (s) | 0.40 | 0.28 | 0.18 |
| fault threshold F_TH | 0.60 | 0.50 | 0.42 |

Rookie and Club are the levels supported for live acceptance; Pro is best effort (sim ordering + a 10-ball live smoke if time). Insane (9.5 m/s, 0.32 s flight) stays a config row only: it stacks 40–90 ms detection + 70–150 ms camera lag against a 0.32 s flight and is probably unplayable. Incoming speed is dominated by the tag level: `v_in = v_tier × (0.9 + 0.2·s_prev) × r(n)`, flight floor 0.30 s.

**Per-shot speed / spin / aim / risk.**
- Speed: `s = clip((w_pk − ω_lo)/(ω_hi − ω_lo), 0, 1)` from the peak gyro rate; `v_out = 3.0 + 11.0·s^0.8` m/s (km/h = 3.6·v). Clipped swings count `s = 1`.
- Spin (3-class, if shipped): `StandardScaler + LogisticRegression` on 12 swing features (unit gyro-peak direction, unit net rotation, unit linear-accel direction, log peak, duration, back ratio). Gestures: **flat** = push straight through · **top** = brush up (wrist rolls forward-up) · **back** = chop (brush down). `T = (p_top − p_back)·A`, `A = 0.4 + 0.6·s`; after the bounce (at 55% of the flight) the flight-time factor is `f = clip(1 + 0.25·T, 0.75, 1.25)`. If max probability <0.5 → flat. No model → "SPIN: uncalibrated". 12 swings/class, 4/class held out (n = 12), gate ≥0.75.
- Aim from your hand's lateral position in the reach box. Quality `q_total = 0.5·q_pos + 0.5·q_time`; perfect ≥0.9.
- **Risk (deterministic, explainable):** fault if `s·(1 − q_total) > F_TH[level]` (out if `s>0.7`, else net); a perfect hit can never fault; faulted shots never increment the streak.

**Swing detector (signed, pure, tested offline).** Every threshold is stored through `GYRO_PER_DPS`/fixture percentiles so a 10× unit error is a one-constant fix. Forward axis `u_fwd` = principal axis (SVD) of gyro vectors at the peaks of calibration swings, sign-fixed so the forward pulse is positive; signed rate `s = (g − bias)·u_fwd`. A backswing projects **negative** and never arms. FSM: IDLE → arm when `s > ARM = 0.4·T_PK` and rising → FWD (SWING_START at `s > 0.5·T_PK`; abort after 0.45 s, or if `s < 0.3·ARM` with `w_pk < T_PK`) tracks the peak on raw/median-3 magnitude (never EMA) → fires IMPACT when `w_pk ≥ T_PK = 0.6·ω_lo` and (`w < 0.9·w_pk` or two consecutive falling samples), duration 60–400 ms, ≤2 sign reversals in the previous 0.8 s → COOL 0.30 s; gyro bias re-zeroed after 0.3 s with `w < 25 dps`. Samples inside any haptic blank window are ignored. A magnitude-only version fired on the backswing 100% of the time in a reviewer's simulation, so a backswing-only fixture must give 0 IMPACT (test).

**Judge (J1–J6; G0/G1 are milestone gates).** A swing is a HIT only if: **J1 timing** `t_i ∈ [t_c − E, t_c + L]` (earlier is ignored; no hit by `t_c + L + D95` is a MISS, D95 = measured p95 detection lag, default 0.12 s) · **J2 pose** paddle point within R of the ball's plane point over the approach window `[t_i − 0.30, t_i − 0.05]` (lag-corrected snapshot, visibility ≥0.6 on ≥2 frames or 100 ms of coverage, whichever is lower) **and** the newest sample at detection within 1.6·R · **J3 swing** `w_pk ≥ T_PK`, duration + oscillation limits · **J4 cross-sensor** pose wrist-speed peak within ±150 ms of the IMU peak (**logged only**) · **J5 refractory** one hit per ball, 0.35 s after a counted hit, ≤3 hits/s · **J6 shake lock** high gyro energy or a dominant **3–8 Hz `rfft` peak** for >1 s with no ball → paddle locked 1 s. Each failed gate shows in the **x-ray policy HUD**. Pose lock: one player, shoulder width captured in calibration (1 s standing still, also in the guest quick-tune) and locked ±25% — a second person entering the frame never moves the paddle, and an out-of-range width gives PAUSED with a HUD message (tested).

**Opponent policy (answers question (a)).** Perceive the ball after τ_react with noise → forward-simulate the leg to the intercept → move a speed-limited **PD paddle** (P2's `PDController`, gains 1.0/0.35, derivative low-pass 0.5) → choose a target zone by a **softmax** over a utility that wrong-foots your tracked hand (+ the learned Q bonus if shipped) → in **Match** it misses with probability `clip(p0 + k_v·max(0, v − v_ref) + k_w·max(0, spin − spin_ref) + k_d·reach_deficit, 0, 0.95)`, so fast, spinny, wide shots win points and speed/spin control *is* the strategy; in **Survival** it never misses and only the ramp changes. **Survival ramp:** `r(n) = min(1 + 0.03n, 1.8)` on speed, spin amplitude `min(1, 0.10 + 0.03n)`, wide-ball probability `min(0.8, 0.3 + 0.01n)`, every 10th ball a special; the HUD and end-screen also show your max km/h so swinging hard has a payoff. **Match:** first to 7, rally scoring, CPU serves every point (stated limitation). **Q-bandit (conditional):** 9 states (hand lateral bin × last return zone) × 9 zone actions, α 0.1, γ 0.9, decay ×0.98, floor 0.05 are from `scripts/rl_straight.py` (an untracked file — copy the constants into `config.py`); ε₀ = 0.15 (Pro) is new.

**Haptic language (hub motors via `motor_run_for_time`; pulses ≤400 ms; one pattern per 100 ms; ≤10 writes/s; intra-pattern gaps ≥80 ms; all through the single actuator thread).** Early/late are coded by **pulse count + beep pitch, not left/right motor** (both motors sit in one fist). **H0 (G0) = rows marked ★**; the rest are SHOULD.

| Event | Motors | Beep (Hz) | Light |
|---|---|---|---|
| Ready / tag locked | 150 ms @40% both | 660 | BLUE BREATHE |
| Countdown 3-2-1 | 25 ms @70% ×2, last THUMP 60 ms @100% | 880 on last | |
| ★ Hit perfect | THUMP 60 ms @100% | 1760 | WHITE |
| ★ Hit good | none (a 10 ms / 20% difference from "early" is not felt through a fist) | 1320 | GREEN |
| Hit early | 1 tick 25 ms @50% | 700 | ORANGE |
| Hit late | 2 ticks 25 ms, gap 80 ms | 1000 | ORANGE |
| ★ Fault / miss | 400 ms @60% same direction | 220 | RED |
| New record | 5 × 40 ms, gap 120 ms | triple, rising | PURPLE |

Invariants: no pulse inside `[t_c − E − 0.15, t_c + L + D95]` except the post-impact hit pulse; blank window = actuator's *actual* write-return time −10 ms to pulse end + `BLANK_AFTER_PULSE` (default 0.12 s until P6 measures it; extended while a motor notification reports non-zero speed); "both motors" = `with dm.batch(blocking=False): motor_run_for_time(LEFT); motor_run_for_time(RIGHT)`, never `motor_run_for_time(MOTOR_BOTH)` for pulses; repeated ticks are separate writes ≥80 ms apart, each counting toward the 10 writes/s cap; the actuator limits motor-on time to 25% per rolling 2 s (`test_actuator`), sends `motor_stop(MOTOR_BOTH)` on `motorState` STALLED or NOT_ALLOWED_TO_RUN, key `D` disarms the motors at any time, Q/ESC runs the teardown order; `light_color` has no duration, so each event just sets the next colour; `--no-motor` mutes motors but keeps beep + light; fallback chain beep+light → Mac sound + screen shake.

---

## 7. Software modules

File tree (`~/ME193/P5-Ping-Pong/`, every file <500 lines, `play.py` + `config.py` at project root like P1–P3; `data/`, `recordings/` gitignored; `pp` is a tiny wrapper that always runs `my_env/bin/python` with subcommands `env_check`, `bench_*`, `play`, `soak`, `ready` — **never type bare `python`**):

```
README.md requirements.txt requirements.lock pytest.ini .gitignore config.py config_local.json play.py pp models/pose_landmarker_lite.task
pingpong/  __init__ events clock hub actuator haptics swing spin shot vision pose_features pose oneeuro paddle calibration
           find_apriltag tags physics rules modes policy scoring pd judge recorder mqtt_pub store leaderboard canvas hud audio sources_fake
tools/     env_check bench_hub bench_cam bench_haptics scan_hubs reset_hub make_cards calibrate_swing watch_score republish_best sim report
           (every tool starts with sys.path.insert(0, <project root>))
tests/     fakes.py test_contracts.py test_file_sizes.py test_*.py        docs/PLAN.md docs/JOURNAL.md docs/diagram.md docs/cards/
(PARKED, not created now)  uno_q/ …   (CUT) webboard.py register_player.py
```

**Keys:** SPACE start · 1–3 level · M mode (Survival/Match) · P player · C calibrate · X x-ray · D disarm motors · H swap hand · L leaderboard · Q/ESC quit.

Reuse (copy, don't import across project folders; originals untouched):

| Need | Copy from |
|---|---|
| Pose landmarker factory, geometry, feature code | `~/ME193/P1-Pose-Race/pose_features.py` (L63, L76, L83) — edit model path to `models/`; add hand points 17–22 |
| Camera/pose loop skeleton + rate-limited BLE `Car` | `~/ME193/P1-Pose-Race/pose_car.py` (loop L313–366, `Car` L126) |
| Train/collect pipeline for the spin classifier | `~/ME193/P1-Pose-Race/train_poses.py`, `collect_poses.py` |
| AprilTag detector | `~/ME193/P2-Apriltag-Parking/find_apriltag.py` (`make_detectors` L28, `find_tags` L38) — inline the `make_apriltag` import |
| PD controller | `~/ME193/P2-Apriltag-Parking/PD.py` (`PDController` L75; keep that class only) |
| BLE connect / card helpers / radio diagnosis | `~/ME193/P3-Whistle-Soccer/whistle_car.py` (`lego_card` L47, `Car` L60), `~/ME193/P2-Apriltag-Parking/trike.py` (`bluetooth_error` L182, `connect` L232) |
| MQTT publisher pattern | `~/ME193/P3-Whistle-Soccer/whistle_car.py` `Radio` L134–170 (upgrade: QoS1, retain, LWT, `connect_async`); raw paho — the instructor's `mqttlib` dispatches by exact topic and never fires for wildcards |
| Q-learning constants | `~/ME193/scripts/rl_straight.py` (α .1, γ .9, decay .98, floor .05; the file is untracked, so copy the constants into `config.py`) |
| SQLite store + thread-safe pattern | `~/ratengo/app/python/ratengo/store.py` L36–94 |
| Config style / pure-policy pattern | `~/ME193/P3-Whistle-Soccer/config.py`, `whistle_policy.py` |
| Sound synthesis maths | `~/ME193/P3-Whistle-Soccer/songs.py` `note_hz`, `synth` L23–49 only (float32 for `sounddevice`; no pyaudio) |

`requirements.txt` (P1 pins; comments explain why; **do not upgrade**): `mediapipe==0.10.21`, `numpy<2`, `opencv-python==4.10.0.84`, `opencv-contrib-python==4.10.0.84`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `legoeducation==1.1.1`, `bleak==3.0.2`, `paho-mqtt==2.1.0`, `sounddevice==0.5.6`, `pytest`. Python 3.12 only (mediapipe 0.10.21 has no cp313 wheel (V)); the pip cache already holds ~1 GB so the install takes minutes. Freeze to `requirements.lock` after the first working install.

Commits: `P5-Ping-Pong: <imperative>`; **before every commit run `git add P5-Ping-Pong/ && git diff --cached --stat -- P5-Ping-Pong`**, then `git commit P5-Ping-Pong/ -m …` (a pathspec commit silently skips *new* untracked files unless they were added first). Never `git add -A` at `~/ME193`: unrelated untracked/modified files exist (`Me193/` stale nested clone, `Spectrogram.py`, `scripts/{rl_straight,double_motor_straight,mqtt_send,mqttlib}.py`, `scripts/q_table.npy`, modified `scripts/Mqtt.py` (V, 2026-10-06 20:00)); the earlier staged files were committed tonight in `758c788` (P4 Door-To-Door), so the index is clean. **No Co-Authored-By trailer** — your `~/CLAUDE.md` forbids it unless `.claude/settings.json` sets `attribution.commit`, and no settings file sets it (note: all earlier commits carry a Claude trailer, added by other sessions; tell me if you want it on these). Your `~/CLAUDE.md` says ALWAYS run `npm run build` before committing; `~/ME193` has no `package.json`, so `./pp ready` (pytest + selftests) replaces it. `play.py` and `config.py` stay at the project root like P1–P3 (I read the "no root files" rule as the home directory); say if you want `src/` + `scripts/` instead.

---

## 8. MQTT contract

- Broker `test.mosquitto.org:1883`, plain TCP, anonymous, MQTT 3.1.1, keepalive 30; constant in `config.py`, override `PP_BROKER=host[:port]`. Raw paho 2.1.0, `Client(CallbackAPIVersion.VERSION2, client_id="pp-rafae-"+uuid4().hex[:8])`, `reconnect_delay_set(1,30)`, `connect_async()` + `loop_start()` so the game boots offline; never `wait_for_publish` in the game loop.
- **Official topic** `ME193/Rogers/RafaeShafi` (exact, case-sensitive). **Payload** `f"{float(x):.1f}"` (`"0.0"`, `"17.0"`), never JSON/NaN/negative (paho `str()`s ints, so always format explicitly). **QoS 1, retain True.**
- **Value = best streak this run** = `max(session record, current streak)` (`PP_RECORD_SCOPE=record_session` default; `record_alltime` and `live_streak` exist, unit-tested). A hit = a player return that passed every judge gate and was not a fault; CPU returns never count.
- **When:** first increase (so the first rally ticks 1.0, 2.0, 3.0… live) · every further increase · on reconnect (only if >0.0) · at session end. **No 0.0 publish at startup**, so launching (or a crash-restart) never resets the retained value; any live session that scores a hit *does* overwrite it with its own 1.0, 2.0, …, so after the graded run use only `--no-publish`. `--new-record` (with on-screen confirm) deliberately publishes `0.0`. `--resume` (crash recovery of the graded run only) seeds the session record from the last retained value / `publish_log`. `--no-publish` is the single no-publish flag (`--mqtt off` is an alias). At startup the client subscribes to its own topic and *prints* the retained value ("broker currently holds 30.0") without acting on it.
- **Demo-day protocol:** before the graded take, `mosquitto_sub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi --retained-only -C 1 -W 5`; the graded take is a **fresh session** and the last publishing run; **guest, rehearsal, soak and drill sessions use `--no-publish`**. Clear deliberately with `mosquitto_pub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi -n -r` (**never after the graded run**).
- **LWT** `ME193-pp/RafaeShafi/status` ("offline", QoS1, retained; "online" on connect) — **never a will on the score topic**. Extras live **outside `ME193/`**: `ME193-pp/RafaeShafi/demo/score` (the only place demo/sim values go), optional `…/state` telemetry (≤5 Hz, QoS0).
- **Integrity invariants (unit-tested):** the official topic is published only when `session.source == "live"` (hub IMU + camera + not replay/sim/fake/demo) and `--no-publish` is off; payload regex `^\d+\.0$`; QoS1 + retain; record monotone under a seeded random-sequence test.
- Unverified (U): that the broker accepts a retained QoS1 publish on the real topic (the Wed one-off publish settles it), and how Rogers' tool parses it → the email in §10.
- Verify like the instructor: `mosquitto_sub -h test.mosquitto.org -t 'ME193/Rogers/#' -F '%I %t [%p] retain=%r qos=%q'` and `./pp watch_score`.

---

## 9. AI/ML algorithms (claim only what ships) and the class-topic crosswalk

| Algorithm | Where | Two-sentence explanation (draft for (c)) |
|---|---|---|
| MediaPipe BlazePose landmarker (pre-trained CNN, lite) | `pose.py` | A convolutional network regresses 33 body landmarks per frame and tracks them between frames. I use wrist/index/pinky normalised by shoulder width as the paddle position. |
| AprilTag / ArUco 36h11 | `tags.py` | It thresholds the image, finds square quads and decodes a Hamming-protected bit grid into an ID and corners. I vote over frames for START and LEVEL. |
| One-Euro filter | `oneeuro.py` | A low-pass filter whose cutoff rises with signal speed. The paddle point is smooth when still and nearly lag-free in a swing. |
| Signed-axis swing detector (threshold FSM; axis from SVD) | `swing.py` | It projects gyro onto the learned forward axis, arms on a threshold, tracks the peak and fires on the falling edge with an oscillation guard. This rejects backswings, waving and double triggers. |
| FFT (`numpy.fft.rfft`) | `judge.py` J6 | The dominant frequency of the last second of gyro data separates 3–8 Hz shaking from a one-off swing. A shake locks the paddle instead of scoring. |
| PD controller (from P2) | `policy.py`, `pd.py` | Command = Kp·error + Kd·error rate, saturated at a per-level speed after a reaction delay. Whether the CPU reaches a ball is physical, not a coin flip. |
| Softmax (Boltzmann) policy | `policy.py` | Each of 9 target zones gets a utility and is sampled ∝ exp(utility/temperature). Lower temperature plays sharper at higher levels. |
| StandardScaler + Logistic Regression — **only if spin ships** | `spin.py` | It standardises 12 swing features and learns a linear softmax boundary over flat/top/back from my labelled swings. The probabilities become continuous topspin/backspin. |
| Tabular Q-learning bandit — **only if built** | `policy.py` | It keeps Q(s,a) and updates `Q += α(r + γ·maxQ' − Q)` with ε-greedy exploration. State is your hand position + last zone, action is the target zone, reward is whether you fail. |

Not claimed: RK2/Magnus physics, Elo, MPU fusion, whistle detector. **Honest coverage note:** if spin and the Q-bandit both get cut, the trained-model/RL class topics are covered by the pose landmarker + the PD/softmax policy only, and (c) must say so.

**"Use everything learned in class" crosswalk** (built from the instructor's `me193-robotics/Public stuff` and your P1–P3/scripts; the *Rogers, Lectures I-XII.pdf* in Downloads is a 2024 lecture set from another course, so it was not used):

| Class topic | In P5? | How |
|---|---|---|
| LEGO BLE API: motors, IMU, beep, light | Yes | `hub.py`, `actuator.py` |
| Pose estimation (P1) | Yes | paddle position |
| Trained classifier (P1 LogReg) | If spin ships | `spin.py` via the `train_poses.py` pipeline |
| AprilTags (P2) | Yes | START + LEVEL |
| PD control (P2, `Controls/pd_tracker.py`) | Yes | CPU paddle |
| MQTT (P3, `mqttlib`) | Yes | live score |
| Q-learning (`rl_straight.py`) | If time (Friday item 8) | opponent zone bandit |
| FFT / spectrogram (P3 whistle, `Spectrogram.py`) | Yes (cheap) | `rfft` shake discriminator J6 |
| Convolution / Otsu / morphology (`Kernels`, `morphology.py`, `threshold_slider.py`) | Indirect | AprilTag decode *is* a threshold → contour pipeline; I will say plainly in the write-up that it is not implemented from scratch |
| Camera/mic helper libs (`camlib`, `miclib`) | Partly | `camlib.pick_camera` pattern |
| **Haptic feedback (new, not in class)** | Yes | the "fun/cool" requirement |

If your syllabus lists topics not above, tell me and I will map them.

---

## 10. Day-0 checklist and the email

1. **Due date + time (2 minutes, today).** Assignment page, course Notion, or a classmate. Tell me the answer; the schedule branches on it (§5 Day 0).
2. **Email Prof. Rogers (you send it; I only draft):** *"Quick questions on the ping-pong assignment: (1) what is the exact due date/time and where do I submit (Notion link?); (2) is a bare float string like 12.0 on ME193/Rogers/RafaeShafi correct, and is a retained message fine; (3) by 'record' do you mean my best continuous-hit streak this run; (4) will the demo be live in class or recorded; (5) could you share the exact reflection-question template text; (6) is this Notion page solo (Members), or shared with a partner?"* Proceed on the defaults if unanswered.
3. **Hub.** Wake it, read the card colour + 4-digit serial (keep it a string). Charge; weigh; keep a USB-C cable at the desk.
4. **`./pp env_check` from Terminal.app** (Claude builds it first thing after approval): camera frame, BLE connect/beep/disconnect, MQTT round trip, 30 s hub IMU stream at the play position (Hz + worst gap). If camera access fails, grant Camera to that app in System Settings and restart it. Allow Camera + Bluetooth for that one terminal.
5. **Cards.** `./pp make_cards` → print 0–3 at Actual size on the Brother, ruler-check, mount on foam board; keep a spare set.
6. **Paddle prep.** Arrow tape, optional taped shaft masses, wrist lanyard, kitchen scale.
7. **Video host.** Upload a ~20 MB test clip and confirm it plays from a share link.
8. **Calendar + room.** Mark real free blocks Wed–Sat; confirm a room (~2.5 m depth, front light, no window behind) is free for each.
9. **Permission allowlist (optional, your call).** A small allowlist so night shifts never stall on a prompt; I'll propose it and you approve it.
10. **Git.** The index is clean (the earlier staged files went into tonight's `758c788`); leave the unrelated untracked files alone; every P5 commit uses the §7 recipe.
11. **Class syllabus (2 min).** Skim it against the §9 crosswalk and tell me any topic that is missing.
12. **P4 status.** Tell me whether P4 Door-To-Door is finished/submitted, when it is due, and how much is left (its laptop tracker `live_tracker.py` is still missing per its README). It shares your Wed–Sat hours.

---

## 11. Verification (how we know it works end to end)

- **Unit/property (pytest, <20 s):** swing (≥95% synthetic detection; backswing-only → 0 IMPACT; waving → 0 HITs; no events inside blank windows; peak-time error ≤8 ms; 10× unit-rescale invariance) · judge (each hard gate J1–J3, J5, J6 fails independently, J4 is only logged; touch-then-swing-elsewhere rejected; a swing detected after the plane still counts; a second person entering the frame never moves the paddle; a shoulder width outside ±25% gives PAUSED) · physics (deterministic arrival; flight ≥0.30 s for every spin/ramp) · shot (speed monotone in `w_pk`; perfect hit never faults; levels strictly ordered for every `s_prev`) · policy (Survival never misses over 10,000 seeded decisions; Match miss rate monotone in level/speed/spin) · scoring/rules (record monotone; first-to-7; the three scope values) · mqtt (payload regex, QoS1, retain, no 0.0 at startup, republish only >0, official topic refused unless live and not `--no-publish`, no will on the score topic) · store (fresh build, tie-breaks, live-only filter) · actuator/haptics (blank math from actual write times, ≤10 writes/s, no-pulse invariant, teardown order incl. SIGTERM, `--no-motor`) · `test_contracts` (modules import; dataclasses match `events.py`) · `test_file_sizes` (every `*.py` <500 lines) · tags (synthetic 36h11 frames through `TagVoter`: 4-of-6 vote; START held 0.4 s fires once, a 0.2 s flash never; 1.5 s lockout; unallocated ids rejected; START ignored outside LOBBY; LEVEL changes only between rallies).
- **Hardware-free end to end:** `./pp play --fake --no-publish` reaches 10 consecutive hits at Rookie with mouse + space; `./pp sim --rallies 200 --seed 1 --level all` checks CPU miss-rate ordering across levels; an integration fixture spawns a local `mosquitto -p <random>` for retained delivery/reconnect.
- **Live acceptance:** G0 (Thu), G1 (Fri), final (Sat) lists in §5; soaks (5 min Thu, 15 min Fri); the five failure drills; fresh-clone runs Thu, Fri, Sat.
- **Evidence for the write-up:** `./pp report` prints best streak, accuracy, echo latency, publish count and haptic counts; the video shows the HUD, a phone shot of the haptic, and the broker value live.

---

## 12. Risks (top eight) and fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Schedule slip (3 build days + a submission Saturday, other classes, P4 possibly competing) | 4.5 h/day budget, night shifts with deliverables, **clock ladder (§5)**, time-boxed Friday, submit-minimum Friday night, cut order (§2) |
| Hub IMU too slow / units unknown (U) | Measured tonight and Wed at the play position; thresholds from fixture percentiles; <25 Hz → pose wrist-speed detector, Rookie/Club only |
| Detection + camera lag make late windows unreachable at Pro | MISS at `t_c + L + D95`, back-dated timestamps, 0.9-peak trigger; Pro best-effort, Insane not shipped |
| Mass or hub flying off during swings | Taped + axle-stopped masses, lanyard on every player, 20-swing retention test (P5), forearm-mount fallback |
| Haptics weak, or vibration fakes a swing | Taped masses, blank from actual write times + motor notifications, no pulse inside the swing window, beep/light/Mac sound guaranteed, honest in the write-up |
| BLE writes stall the loop or drop silently; hub sleeps or ghost-connects (~24 s) | Single actuator thread, ≤10 writes/s, SIGTERM/SIGINT handlers, `scan_hubs`/`reset_hub`, connect only to the configured card, battery logged, watchdog from the measured histogram |
| First live contact with the camera costs round trips (Claude cannot open it) | You launch camera runs in Terminal.app, logs tee'd to `recordings/*.log`; per-ball logs make every failure replayable; vision code is the most test-covered with fakes |
| MQTT semantics / public-broker spoofing or restart (U) | Three-value switch, email, no startup 0.0, one pre-Thursday publish-and-clear, `--no-publish` for non-graded sessions, video + watcher log as durable evidence |
| macOS permissions / GUI loop / Continuity Camera / display sleep | One canonical terminal, capture on main thread, Continuity Camera off, `caffeinate -dims` + Focus, walking skeleton Wed |

---

## 13. Open items (none block approval)

- **Folder name and P4:** `~/ME193/P4-Door-To-Door-Service` exists (committed tonight), so this project is `P5-Ping-Pong` — confirm the numbering, and tell me whether P4 still has work due Wed–Sat (it shares your hours and the UNO Q).
- Exact due **date and time** → you find it tonight (§10.1); the email confirms it.
- Which Connection Card is the paddle → read it off the hub tonight.
- Whether to keep the UNO Q scoreboard in place of another item → your call (§0.3).
- UNO Q revival after submission: wire-free scoreboard first (1.5 h), then LDR cover-to-serve and MPU ruler; resistor colour bands and GY-521 header solder state matter only then. The board runs one app at a time, so deploying a ping-pong node stops the P4 minifig apps and RateNgo — do it only after P4 is graded.
- Assumptions: right-handed play (`H` key swaps); webcam ≥720p; a 1.5–2.0 m play spot, front-lit; Claude writes most code, you run benches, play, tune, record and write the explanations; nothing was run against hardware, the broker or the board while planning.

---

**Appendix A — full reference design** (swing-detector pseudocode, all formulas, DB schema, test matrix, UNO Q wiring/protocols, draft answers (a)(b)(c) and the Notion reflection outline): see the persisted path at the top; it is copied to `~/ME193/P5-Ping-Pong/docs/PLAN.md` on approval with its OVERRIDES header. Its §11 (milestones) and §13 (pre-flight) are superseded by §5 and §10 here. Its parked UNO Q material (§3.2, §4.4, Day 6) stays valid for after submission.
