> # OVERRIDES — read before using this document (these win over anything below)
> This is the **reference appendix** (formulas, pseudocode, DB schema, test matrix, draft answers) to the main plan `docs/PLAN-main.md` (v3). It was written for a 7-day schedule before the deadline, the P4 collision and two review rounds. Where it disagrees with the main plan, **the main plan wins**. Specifically:
> - Folder is **`P5-Ping-Pong`** (P4 = Door-To-Door already exists). `pingpong/types.py` is **`pingpong/events.py`**.
> - **Judge gates are J1–J6** (this document's G1–G6). Milestone gates are only **G0/G1/G2**.
> - **Early/late haptics are coded by pulse count + beep pitch, not left/right motor.** A "good" hit has **no motor cue** (beep + light only).
> - **Spin default = 3 classes** (flat/top/back), 12 swings/class, 4 held out; 5-class and the SPIN DIAL tag are PARKED.
> - **Metronome offset is optional (default 0.0)**; camera lag = the Wednesday bench constant.
> - **Level tags 1–3** (Rookie/Club/Pro); Insane is a config row only. Rookie/Club are the live-acceptance levels.
> - **UNO Q / LDR / MPU-6050 / webboard / boardlink / whistle / the Day 6–7 sections / §3.2 wiring / §4.4 board link are PARKED** (kept for after submission; deploying to the board also stops the P4 minifig apps and RateNgo).
> - `--mqtt off` is an alias of **`--no-publish`**; `--resume` exists for crash recovery of the graded run only.
> - Score = **best streak this run**; first publish at the first hit; **no 0.0 at startup**.
> - **§11 (milestones) and §13 (pre-flight) of this document are replaced by §5 and §10 of the main plan.** Guests are registered via `--player <name>` and may appear on the board.
> - Every file <500 lines; no Co-Authored-By trailer (`attribution.commit` unset; `~/CLAUDE.md`); pre-commit check = `./pp ready`.

---

# P5-Ping-Pong final implementation plan: "Tiered Hub-Paddle Pong"

Student: Rafae Shafi. Course: ME193 AI & Robotics (Prof. Rogers). Written 2026-10-06 (Day 0). Day 1 = Wed 2026-10-07 ... Day 7 = Tue 2026-10-13. The real submission target is the END OF DAY 6; Day 7 is rehearsal, in-class demo and buffer. Nothing below has been built or run against hardware.

Evidence labels: (V) verified in research or by a reviewer's local test, (E) estimate or design default that a named bench settles, (U) unverified, with a probe or fallback attached.

---

## 1. Context & goals

**Assignment.** Play virtual ping-pong with the LEGO Double Motor (or UNO Q). It must use pose (is the paddle in the right place), IMU (is the person swinging), AprilTags (player level = ball speed, and game start), and post the score live to `ME193/Rogers/RafaeShafi` as a float equal to the record number of continuous hits. It must also (1) use everything learned so far, (2) do something fun not done in class (haptics, Tier-1), (3) answer (a) policy, (b) limitations, (c) AI/ML algorithms in at most two sentences each, (4) answer the template reflection questions.

**Ambitions ("make it insane").** Per-shot speed and spin, stored profiles plus leaderboard, Survival mode (computer never misses), Full Match with selectable difficulty, IMU plus pose showing the player really hitting the ball.

**Hard constraints (confirmed 2026-10-06).** Due about 2026-10-13. Assignment minimum works by end of Day 3 (gate G0). Speed/spin control, both modes, profiles and leaderboard by end of Day 5 (gate G1). The Mac plus the LEGO Double Motor carry the whole required path. The UNO Q, MPU-6050 and LDR are an additive layer with automatic fallbacks. The "Light Sensor" is an analog LDR; the single resistor is its divider partner. Topic is exactly `ME193/Rogers/RafaeShafi`; extras live outside `ME193/Rogers/`. Budget 4-6 productive hours per day.

**Architecture in one paragraph.** One Python process on the Mac (P1 pin set) runs everything required: a vision thread (webcam, MediaPipe pose), a separate tag thread, the hub IMU over BLE with a dedicated IMU worker running the swing detector, a pure-function game core, a cv2 HUD, SQLite behind a writer thread, and a paho MQTT publisher. The Double Motor is the hand-held paddle: its IMU gives swing timing, peak rate (shot speed) and spin features; its motors, beep and light are the haptic channel. Pose says WHERE the paddle is; the IMU says WHEN and HOW HARD; AprilTags say start and level. The UNO Q "Arena Node" is a desk scoreboard (8x13 matrix), a cover-to-serve LDR input and a short-cable MPU-6050 bench ruler for the hub's undocumented units. A hardware-free `--fake` mode exists from Day 1.

**Gates.** G0 end Day 3 (assignment minimum, backup video). G1 end Day 5 (everything the student asked for). G2/submit-ready end Day 6 morning for write-up and video (tag `v1-submit-ready`), UNO Q extras in the Day 6 afternoon on top. Day 7 is rehearsal and buffer only.

### Conflict and review resolutions (what was trusted, and why)

| # | Issue | Resolution |
|---|---|---|
| 1 | MPU role | Hub is the single timing and speed authority. MPU-6050 is a bench ruler (Day 6, hard 1.0 h cap, first cut after whistle). Runtime MPU events are logged, never used for hits. |
| 2 | MPU rate 1 kHz vs 250 Hz | 250 Hz (SMPLRT_DIV=3), polled at 500 Hz. A 14 byte burst is about 1.6 ms at 100 kHz (V, two reviewers), so 1 kHz is impossible at 100 kHz. |
| 3 | Physics | Analytic Leg, stylised spin. No RK2/Magnus this week. |
| 4 | Renderer | One cv2 window, vision on its own thread. No pygame. |
| 5 | Spin source | LogReg on IMU features primary, tag roll dial override, flat when no model. |
| 6 | Swing detector flaw (blocker) | Magnitude-only FSM fired on the BACKSWING 100% of the time in a reviewer's simulation. Replaced with a signed forward-axis detector plus oscillation guard plus window gating (5.3). Acceptance now checks peak-time and amplitude against argmax, and a backswing-only fixture must give 0 IMPACT. |
| 7 | Latency budget 2x optimistic (major) | Real detection delay is 65-110 ms after the true peak. Fixes: trigger from raw/median-only magnitude at 0.9 x peak, IMU worker blocks on queue, pose window moved to approach-only so the verdict is immediate, MISS declared at t_c + L + D95, haptics honest about lateness, hit-stop visual-only. |
| 8 | G3 vs speed calibration | One threshold: G3 = w_pk >= T_PK = 0.6 x omega_lo. Soft swings count. |
| 9 | Incoming ball speed diluted level (major) | v_in = v_tier x (0.9 + 0.2 s_prev): level dominates and ordering by level is guaranteed (6.1). |
| 10 | Fault dice made streaks luck | Faults are deterministic from s x (1 - q_total) against a per-level threshold; a perfect hit (q_total >= 0.9) can never fault. |
| 11 | MQTT clobber and frozen-looking topic (major) | No 0.0 publish at startup. The graded run is a fresh session; first publish is at the first hit. Optional `--new-record` publishes 0.0 after an on-screen confirm. One switch `PP_RECORD_SCOPE` = record_session (default) / record_alltime / live_streak, all unit-tested. The HUD shows live streak and record, so either reading of "record" is visible. |
| 12 | Day-7 submission | Submit-ready at end of Day 6; Day 7 buffer. Added Day 0 block tonight. |
| 13 | Serialized work | Pure modules are built in parallel with hardware gates (Claude Squad `cs` suggested); mqtt_pub and README answers skeleton move to Day 1-2. |
| 14 | MPU tether plan B | Dropped. Fallback for a slow hub IMU is a pose wrist-speed swing detector (6.4 and 12). |
| 15 | `pip check` clean | Impossible (V, three reproductions: "mediapipe 0.10.21 is not supported on this platform", metadata quirk). Removed from the gate; import line plus a dummy `detect_for_video` is the check. |
| 16 | Copy hazards | `pose_features.py` loads its model from its own folder, `find_apriltag.py` imports `make_apriltag`, `whistle_policy.py` imports `config`. Copy edits listed in 13. |
| 17 | Cards | P2's `make_tag` pad is size//5; a 15 cm tag would print about 21 cm wide. New `tools/make_cards.py` fixes tag 15 cm plus 2 cm white margin (19 cm, fits US Letter). |
| 18 | Git staging | `~/ME193` already has staged unrelated files. Commit with an explicit pathspec (`git commit P5-Ping-Pong/ -m ...`) after checking `git diff --cached --stat`. |
| 19 | Network environment | Using the node puts the Mac on the iPhone hotspot. Benches P1 and P9 run in both configurations; hotspot set to 5 GHz; demo rule in 12. |
| 20 | Board discovery deadlock | Mac POSTs a 1 Hz keepalive; the board sends heartbeat UDP to the subnet broadcast and the Mac learns the board IP from the packet source; `PP_NODE_IP` is an override. |
| 21 | Bridge direction Linux to MCU untested | P10 adds a "curl POST of a number appears on the matrix" check, using `Bridge.provide_safe`; no Arduino API inside a plain `provide`. |
| 22 | Scope trim | Cut: schema migration framework, Elo, TAMPER republish, extra MQTT topics (event, retained leaderboard), Elo anchors. Kept: tiny publish_log, status LWT, web page (cut line 4). |
| 23 | Q-learning scheduling | Moved to Day 5 (before the web page and UNO Q) so "use everything learned" is not hostage to the cut list. |
| 24 | Haptics vs flight time | Motor cues for spin and early/late only when flight time >= 0.5 s (Rookie, Club); at Pro and Insane they degrade to beep and light. |

---

## 2. Requirements trace (assignment line -> feature -> test)

| ID | Requirement | Feature / module | Acceptance test |
|---|---|---|---|
| R1 | Pose: paddle in the right location | `judge.py` G2 on the lag-corrected PoseRing snapshot | Swing with hand more than 2R from the marker rejected 10/10; hand on marker without a swing gives 0 hits; `test_judge` truth table |
| R2 | IMU: is the person swinging | `swing.py` signed detector, gates G3, G6 | 20 real swings give >= 19 IMPACT; 10 soft swings give >= 9; backswing-only fixture 0 IMPACT; hub shaken with arm still gives 0 hits |
| R3 | AprilTag level = ball speed | LEVEL tags 1-4 set v_tier, windows, radius, CPU skill | CPU-serve flight times Rookie vs Insane differ >= 2.0x in the log at fixed simulated swing strength (tools/sim.py); 20/20 latch at 1.8 m |
| R4 | AprilTag game start | START tag 0, held 0.4 s in LOBBY | One START per presentation; 0.2 s flash never triggers; none mid-rally |
| R5 | Live score on `ME193/Rogers/RafaeShafi` as a float | `mqtt_pub.py` QoS1 retain `f"{x:.1f}"` | `mosquitto_sub` shows 1.0, 2.0, ... live; holds after a miss; late joiner sees record; Wi-Fi-off drill republishes |
| R6 | Use everything learned | Pose, trained classifier, AprilTag, PD, MQTT, Q-learning (Day 5), FFT whistle (stretch) | Section 9 table; every claimed algorithm exists in the tagged repo |
| R7 | Fun thing: haptics | `haptics.py`, `actuator.py` | H0 felt Day 3; blind identification of {perfect, early, late, miss} >= 80% over 20 trials at Club (Day 4) |
| R8 | Questions (a)(b)(c) | README `## Questions`, Notion | Section 14, numbers from `tools/report.py`, reconciled against the tagged release |
| R9 | Template questions | Notion reflection block | Four bullets filled with real events, student's own words |
| R10 | Per-shot speed | `shot.py` | 5 soft vs 5 hard swings: median out-speed ratio >= 2.0 |
| R11 | Per-shot spin | `spin.py` plus dial override | Held-out-block accuracy >= 0.80 (else 3 classes); live >= 70% per class |
| R12 | Profiles, leaderboard | `store.py`, `leaderboard.py`, `webboard.py` :8350 | Restart keeps profile and calibration; end-screen top-5 equals page and CLI |
| R13 | Survival | `modes.py`, `policy.py` | Sim: ramp visible at n = 10/20/30; ends at first player fault; CPU never misses 10,000 seeded decisions |
| R14 | Match with difficulty | `modes.py`, `policy.py` miss model | Sim CPU miss rate Rookie > Club > Pro > Insane; live Match to 7 completes at Club |
| R15 | Show the player hitting | HUD: webcam backdrop, skeleton, paddle ghost, x-ray gate checklist | Visual check on video |
| R16 | Hardware-free dev | `sources_fake.py`, `--fake` | `play.py --fake --mqtt off` reaches 10 hits with mouse and space |

---

## 3. Hardware

### 3.1 BOM and roles

| Item | Role | Required? |
|---|---|---|
| Mac (M1 Pro 14", FaceTime camera, speakers, BLE, printer on network (V)) | Everything on the required path; sole MQTT publisher; SQLite and leaderboard host | Yes |
| LEGO Double Motor (BLE) | THE PADDLE: IMU (15 ms notify, about 66 Hz cap (V)) gives swing time, peak rate, spin features; motors, beep, light are haptics; button is a fallback START and calibration "next" | Yes |
| Printed AprilTag cards 36h11, tag 15 cm plus 2 cm white margin on Letter, mounted on foam board or card stock | 0 START, 1-4 LEVEL; later 10-19 PLAYER, 20 SPIN DIAL, 40/41 MODE | Yes (ids 0-4 first). Fallback phone screen (inverted detection on) |
| Paddle prep: tape arrow on hub, optional cardboard blade, shaft masses (secured), wrist lanyard | Grip repeatability, felt haptics, safety | Yes (free) |
| UNO Q "RafaesBoard" | Arena Node: matrix scoreboard, LDR cover-to-serve, MPU bench ruler | Optional |
| MPU-6050 (GY-521), LDR, the one resistor | Bench ruler and cover-to-serve | Optional |
| Dupont jumpers (match GY-521 header gender), mini breadboard or clips | Wiring | Order tonight, needed Day 6 |
| iPhone hotspot (5 GHz, "Maximize Compatibility" off) | Mac and board both join; internet for the broker | Only for the node |
| Built-in Mac speakers (or wired output) | Game audio (Bluetooth audio adds 150-250 ms and shares the radio) | Yes; keep AirPods disconnected |

Parked: UNO Q as paddle or brain, pose/tags on the board, bleak inside the board container, board-side MQTT, 1 kHz MPU streaming, tethered MPU as a play sensor, LEGO Color Sensor (not owned).

### 3.2 Wiring and pinout (Day 6 only; required path has no wires)

All UNO Q I/O is 3.3 V (A0-A5 not 5 V tolerant (V)). Power the board from a 5 V / 3 A USB-C supply. Boot takes 20-30 s.

| Signal | Connection | Notes |
|---|---|---|
| MPU VCC / GND | 3V3 / GND | Never 5 V |
| MPU SDA / SCL | D20 / D21 (`Wire` = I2C2, PB11/PB10 (V)) | Call `Wire.begin()` first, then `Wire.setClock(100000)` (default is 400 kHz; setClock needs begin) |
| MPU AD0 / INT | GND (0x68) / unconnected | |
| LDR | 3V3 to LDR to A0 | |
| Divider resistor | A0 to R to GND | Keep off A4/A5 (that is `Wire2`) |

MPU config (raw registers, no library): 0x6B=0x01; 0x19=0x03; 0x1A=0x01 (gyro BW 188 Hz, accel 184 Hz); 0x1B=0x18 (+-2000 dps, 16.4 LSB/dps); 0x1C=0x18 (+-16 g, 2048 LSB/g); burst-read 14 bytes from 0x3B and check `requestFrom == 14`; poll at 500 Hz (INT unconnected; samples may repeat, it is a peak ruler not a 250 Hz stream). WHO_AM_I accepted set {0x68, 0x70, 0x71, 0x73} everywhere. The ruler cannot calibrate swings above 2000 dps: state "R^2 on swings below 2000 dps". Check the GY-521 VDD with a multimeter or an I2C scan at the 3V3 supply.

LDR: `analogReadResolution(14)` (core default is 10 (V)). The resistor value DOES matter (reviewer correction, V by algebra: V = 3.3 R/(R+R_ldr); a very large or very small R kills the ratio). Read the resistor's color bands on Day 0, record the value in `config.py`. Thresholds come from a 3 s on-screen "cover now / uncover now" calibration (midpoint in log space), with a fallback rule of a drop of more than 2x within 300 ms against a slow baseline. Abort the LDR task if the measured uncovered/covered ratio is below 3x.

Network: Mac POST `http://<BOARD_IP>:8080/state` (Mac to board TCP works already, SSH precedent); board UDP to `<MAC_IP>:9393` (macOS firewall off (V)). Leaderboard page on Mac :8350. Ports 8340/8341/5176/4000 are taken.

### 3.3 Mounting, power, safety

- Hold the hub in the dominant fist like a paddle handle; tape an arrow on the casing so grip orientation repeats. Weight is 80 g or 301 g in conflicting sources: weigh it (P5). If 301 g, keep sessions short.
- Shaft masses (coin, nut, LEGO wheel) must be secured (tape plus a snug axle stop); test 20 hard swings with nothing loosening; fingers stay clear of the shafts; wear the wrist lanyard. If bare shafts are too weak in P6, strap the hub to the forearm and store the offset in the profile.
- Always `disconnect()` in `finally` (a killed process leaves the hub connected for about 24 s). Charge fully. Stay near the Mac; probe P1 is measured AT the play position (see P1).
- Camera: laptop on a table, lid tilted to frame head to hips of a player 1.5-2.0 m away, front-lit, no window behind. Screen legibility at 1.8 m is tested on Day 2 (large-glyph full-screen HUD); an external display or TV is the upgrade if unreadable.
- UNO Q on a stool on the off-hand side, matrix toward the player. Deploying the app stops RateNgo (tell the student).

### 3.4 Bench probes (Day 1, before any threshold tuning; each writes JSONL to `recordings/bench_*.jsonl` and a number into `config.py`)

| Probe | Command / procedure | GO | If not GO |
|---|---|---|---|
| P1 hub rate at the PLAY position | `my_env/bin/python tools/bench_hub.py --test rate --cards green:0997,orange:1129,yellow:994,red:1131,azure:3683 --notify-ms 15 --secs 30`, then 20 swings at 1.8 m with the hub in the fist and the body between hub and Mac; log gaps > 100 ms and the gap histogram; repeat on home Wi-Fi and on the iPhone hotspot with MQTT and camera running | Connects; >= 40 Hz achieved; record Hz, histogram, `CARD_COLOR`, `CARD_SERIAL`, p99.9 gap | 25-40 Hz: NOTIFY_MS 20/30, widen windows x1.3. Below 25 Hz sustained: switch the swing source to the pose wrist-speed detector (hub kept for haptics and gate), run Rookie/Club only, and record the descope in writing by Day 1 noon |
| P2 units | `--test rest6`, `--test turn` | Rest `|a|` CV within 3% over 6 faces gives ACCEL_PER_G; three 360 degree turns give GYRO_PER_DPS within +-5%, and yaw sign | Thresholds fall back to fixture percentiles (rest noise, soft peak, hard peak); everything is expressed via the calibration constant |
| P3 clipping and axes | `--test swing --n 10` max effort | Max raw per axis, plateau count (>= 3 samples at one max sets HUB_FS_RAW); PCA forward axis stored as a candidate u_fwd | Clip flagged on plateau only |
| P4 gestures | logged during P3 swings | Wire a gesture to an action only at 0 false events | Log-only |
| P5 weigh and mount | Kitchen scale; fist vs forearm | Mass and mount chosen | Forearm strap |
| P6 haptics (6 recipes only) | `tools/bench_haptics.py` antiphase vs same-direction, 20/40/80 ms, with/without masses; rate by feel in a noisy room; measure IMU accel RMS during the pulse | >= 1 recipe 3/5 through the grip; BLANK_AFTER_PULSE = RMS-return time + 20 ms; max writes/s (cap 10/s); one batch with 2 motors + beep + light = one write | beep + light + Mac sound primary; honest weakness in write-up |
| P7 batching | Offline fake transport | Batch packs into 1 write | Separate writes, lower cap |
| P8 vision | `bench_vision.py --camera 0 --secs 60` | Pose loop >= 20 fps, p50 <= 25 ms; START found >= 90% of frames at 1.8 m at 960 px width with 30-45 degree tilt; camera lag by wave cross-correlation under the venue lighting (no capture timestamps exist, so "frame age" is not printed) | Larger cards, ROI, 1280 only in LOBBY |
| P9 MQTT (both networks) | `bench_mqtt.py` publish `0.0` QoS1 retain to scratch topic `ME193-pp/RafaeShafi/selftest`; read back with `mosquitto_sub`; check 1883 reachability from the campus network too | Round trip < 2 s, echo RTT median recorded (ICMP 95-149 ms (V)). Official topic untouched until Day 3 | Local `mosquitto -p 18831` plus `PP_BROKER` |
| P10a UNO Q smoke (Day 4, wire-free, 0.5 h, only if G0 is green) | `uno_q/scripts/deploy.sh`; matrix "hello" + LED3; Bridge both directions (`curl -X POST` of a number appears on the matrix via `provide_safe`); Mac to board HTTP and board to Mac UDP | All reach | Park to Day 6 |
| P10b UNO Q wired (Day 6) | WHO_AM_I, `|a|` about 1 g, LDR ratio >= 3x | | Cut 6b/6c |
| P11 audio latency | Click to microphone on built-in speakers | <= 40 ms | Pin output device by name; no Bluetooth audio |

---

## 4. System architecture

### 4.1 Diagram

```
 PLAYER: dominant fist holds Double Motor (paddle); off hand/stand: AprilTag cards
   | webcam                                    | BLE: IMU 15 ms notify in; haptic cmds out
   v                                           v
+--------------------------- MAC: one Python process (P1 pins) ---------------------------+
| [V] VisionWorker: newest frame, t_read stamp, unmirrored 640x360 -> MediaPipe pose       |
|     -> One-Euro -> immutable PoseSnapshot (tuple swap, no shared deque)                  |
| [T] TagWorker: latest frame, 960 px gray, LOBBY only (+ optional ROI dial in rally)      |
| [H] legoeducation loop thread: callback ONLY parses + stamps monotonic_ns -> SimpleQueue |
| [I] IMU worker: blocks on queue -> SwingDetector -> SwingEvent -> main queue             |
| [A] Actuator: sole sender of hub commands, <=10 writes/s, posts (t_write, dur) back      |
| MAIN ~60 Hz: judge -> shot -> Leg physics -> Opponent -> Rules/Modes -> Haptics -> HUD   |
|        |             |                  |                  |                              |
|  [Q] paho thread  [S] StoreWriter    [N] NodeLink       [W] web :8350 (read-only)        |
+--------+-------------+------------------+---------------------------------------------+
         v             |                  | POST :8080 /state 1 Hz keepalive + on change
 test.mosquitto.org    v                  v                    ^ UDP {cv,hb,sw}
 ME193/Rogers/RafaeShafi = "12.0"   data/pingpong.db     UNO Q Arena Node (additive)
 extras: ME193-pp/RafaeShafi/...                          Linux stdlib HTTP :8080 + UDP fwd
                                                          MCU: matrix, LED3, LDR A0, MPU
```

### 4.2 Threads and rules

Only queues, immutable snapshots and locked scalars cross threads. The main thread never calls BLE, MQTT, HTTP or SQLite synchronously (the end-screen top-5 query runs on the StoreWriter thread and returns through a future/queue).
- `cv2.VideoCapture` opened on the main thread first (macOS permission), then handed to [V]. Run everything from the canonical terminal that was granted Camera and Bluetooth. MediaPipe timestamps are strictly increasing monotonic ms.
- [H] callback only parses and `put_nowait`s (sync library calls there raise RuntimeError (V)).
- [I] blocks on the sample queue (no 60 Hz polling latency) and emits events with back-dated peak times.
- [A] owns batch state (device-global and unlocked (V)); `blocking=False` still blocks the caller on the BLE write, so only [A] calls it. bleak 3.0.2 write-without-response has no flow control (V), hence the 10 writes/s cap.
- PoseRing: the vision thread builds an immutable tuple of recent samples each frame and swaps one reference; the judge reads the reference (no deque iteration across threads). Record/streak scalars are lock-guarded; paho `on_connect` reads a snapshot.
- Teardown order: stop and join [A], `cancel_batch()` if a batch is open, `motor_stop(MOTOR_BOTH)`, `disconnect()`, each in its own try/except (a batch swallows any thread's commands and `disconnect()` raises inside one (V)).
- Hub stale: gaps up to the P1 p99.9 + margin (expected 250-300 ms) extend the ball clock by the stall; only silence beyond that PAUSES (never a hit, never a fault) and starts a reconnect loop re-passing 15 ms. Five periods is a log warning only.

### 4.3 One shot, end to end

1. CPU launch: `policy.plan_return()` picks zone, speed, spin; `physics.plan_leg()` returns a Leg with exact arrival `t_c` and point. HUD draws the arrival marker and closing timing ring.
2. [V] updates the pose snapshot (stamped `t_read - delta`); [H] streams IMU.
3. SWING_START (whoosh, paddle ghost) when the signed rate exceeds 0.5 x T_PK; IMPACT carries the back-dated peak `t_i`, `w_pk`, features. IMPACT is known 35-80 ms after the true peak with the 0.9 x peak trigger (E; the budget is replaced by replay of Day 1 fixtures, target <= 120 ms peak to audible hit, so the student is told to swing slightly early).
4. Judge runs at t_detect (immediate): G1 timing on `t_i`, G2 pose on the approach window [t_i-0.30, t_i-0.05] plus the final-position check on the newest sample, G3, G5, G6. Verdict carries per-gate pass/fail for the x-ray HUD. MISS is declared at `t_c + L + D95` (D95 = measured p95 detection lag, default 0.12 s), and the ball keeps rendering past the plane meanwhile.
5. Valid hit: `shot.make()` gives speed, spin, aim and (deterministic) fault; outgoing Leg launches at `max(now, t_c)`.
6. Feedback: haptic and sound at `max(now, t_c - 0.04)` (bench-tuned); blank windows from the actuator's actual write times; hit-stop 60 ms is visual-only (render freeze), never a sleep on the game loop.
7. Score: `ScoreTracker.on_valid_hit()` if not a fault; if max(record, streak) increased, ScorePublisher queues a QoS1 retained float; NodeLink and Store queue their updates.
8. CPU reply (Survival always; Match by miss model). Rally ends on a fault; end screen shows top-5; LOBBY.

### 4.4 Message formats

Internal frozen dataclasses (`pingpong/types.py`): `ImuSample(t_ns, g[3], a[3], src)`, `SwingEvent(kind, t_ns, w_pk, dur_ms, n_reversals, axis_unit, net_rot_unit, a_lin_unit, clipped, feat[12], src)`, `PaddlePose(t_scene_ns, u, v, conf, hand)`, `TagEvent(role, value, t_ns)`, `Verdict(kind, q_pos, e_s, d_min_sw, gates)`, `ShotParams(v_out, T, S, A, aim_a, q_total, fault, label)`, `HapticCmd(name, strength, fire_at_ns)`, `GameEvent(kind, t_ns, data)`.

Board link: Mac to board `POST /state` `{"streak","record","phase","mode","level","reply_to":"<mac_ip>:9393"}` as a 1 Hz keepalive plus on change (at most 5 Hz), 0.3 s timeout, drop-oldest. Board to Mac UDP JSON, cover events sent twice 5 ms apart (dedupe by seq): `{"k":"cv","seq","tb","on"}`, `{"k":"hb","seq","tb","ldr","mpu","hz"}` at 4 Hz (also sent to the subnet directed broadcast so the Mac learns the board IP), `{"k":"sw","seq","tb","wp","dur"}` (bench only). Alive = heartbeat within 2 s. Bridge carries events only (115200 baud, about 11.5 kB/s, 256 B messages (V)). Restarting the board app mid-game must recover within 1-2 s.

### 4.5 Latency budget (all (E) until Day 1 fixtures replace them)

| Stage | Estimate | Handling |
|---|---|---|
| True peak to IMPACT known | 35-80 ms with raw 0.9-peak trigger at 15 ms notify (reviewer sim: 65-110 ms with the old EMA 0.8 rule, 109 ms at the default 100 ms notify) | Back-dated timestamps; D95 sets the MISS deadline |
| Camera to pose | exposure + AVFoundation 70-150 ms + pose 10-25 ms | Ring stamped `t_read - delta`; approach-only window; no wait for post-impact frames |
| IMPACT to audible/visible | 0-16 ms main loop (hit-stop visual-only) | Whoosh at SWING_START masks it |
| Command to felt | queue < 1 ms + BLE 0-30 ms + motor start 5 ms | For an on-time swing the felt pulse lands about 100-200 ms after the peak; acknowledged limitation |
| MQTT | broker RTT 95-190 ms; publish call < 1 ms | Asynchronous |
| Main loop | 16.7 ms; draw 4-8 ms; p95 < 20 ms checked Day 2 | |
| Flight (D = 3 m) | Rookie 0.86 s, Club 0.60, Pro 0.43, Insane 0.32 (floor 0.30 on final T') | Windows absorb jitter |

### 4.6 Degradation ladder

Board silent: Mac-only display; START by tag or key. Broker down: QoS1 queue, republish on reconnect, HUD MQTT red. Pose lost over 1.5 s: PAUSED. Hub stale beyond threshold: PAUSED plus reconnect. Tags unreadable: keys (HUD shows KEY START). Motors weak or dead: beep + light, then Mac sound and screen shake. Hub IMU too slow (P1): pose wrist-speed swing detector.

---

## 5. Software modules

### 5.1 File tree (`~/ME193/P5-Ping-Pong/`, every file under 500 lines, no player data in git)

```
README.md  requirements.txt  requirements.lock  .gitignore
config.py                 # flat constants with units; env PP_BROKER, PP_NODE_IP, PP_RECORD_SCOPE
play.py                   # entry: argparse, wiring, main loop, try/finally teardown
models/pose_landmarker_lite.task
pingpong/
  types.py clock.py hub.py actuator.py haptics.py swing.py spin.py shot.py
  vision.py pose_features.py pose.py oneeuro.py paddle.py calibration.py
  find_apriltag.py tags.py physics.py rules.py modes.py policy.py scoring.py pd.py judge.py
  mqtt_pub.py store.py leaderboard.py webboard.py boardlink.py
  canvas.py hud.py audio.py sources_fake.py whistle_policy.py (stretch)
tools/ bench_hub.py bench_haptics.py bench_vision.py bench_mqtt.py make_cards.py calibrate_swing.py
       register_player.py leaderboard.py watch_score.py sim.py report.py
tests/ fakes.py test_*.py      docs/cards/ docs/diagram.md
uno_q/pingpong-node/ app.yaml(ports: [8080]) python/{main.py,node_server.py}
       sketch/{sketch.ino,sketch.yaml,src/{matrix_render.cpp,ldr_cover.cpp,mpu6050.cpp,event_queue.h}}
uno_q/scripts/deploy.sh  uno_q/Makefile  uno_q/tests/
data/ (gitignored)  recordings/ (gitignored)
```

`play.py` and `config.py` sit in `P5-Ping-Pong/` (P1-P3 convention; `~/CLAUDE.md` "no root files" refers to the home directory). Commit style `P5-Ping-Pong: <imperative>`, always `git commit P5-Ping-Pong/ -m ...` after `git diff --cached --stat`; never `git add -A` at `~/ME193` (staged unrelated files and an untracked nested clone `Me193/` exist (V)). Follow `~/CLAUDE.md` for the Co-Authored-By trailer (only if `.claude/settings.json` sets `attribution.commit`). The `npm run build` rule does not apply; the pre-commit check is `my_env/bin/pytest -q`.

### 5.2 Module responsibilities and reuse (copy, do not import across project folders)

| Module | Responsibility | Copy from |
|---|---|---|
| `config.py` | Every tunable (broker, topic, card, tags, LEVELS, windows, thresholds, bench-measured scales, haptic recipes, node ports, resistor value) | Style of `~/ME193/P3-Whistle-Soccer/config.py` |
| `clock.py` | Injectable monotonic Clock/FakeClock | `pose_car.py` monotonic pattern |
| `hub.py` | HubLink: connect with card, notify 15 ms re-passed on every reconnect, `.connected` check, enqueue-only callback, staleness watchdog | `~/ME193/P3-Whistle-Soccer/whistle_car.py` `lego_card` L47, `Car` L60; `~/ME193/P2-Apriltag-Parking/trike.py` `bluetooth_error` L182, `connect` L232 |
| `actuator.py` | Sole command thread; priority record > point > hit > tick; at most 1 pattern per 100 ms, at most 10 writes/s; pulses <= 400 ms; duty <= 25% per 2 s; posts actual write times; `--no-motor` | `~/ME193/P1-Pose-Race/pose_car.py` `Car` L126 limiter |
| `haptics.py` | Pattern table (6.9), blank math, scheduler invariants | New |
| `swing.py` | Signed SwingDetector (5.3), feature extractor, clip flag; also a pose wrist-speed variant for the degraded mode | Pure-policy pattern of `whistle_policy.py` |
| `spin.py` | StandardScaler + LogisticRegression, per-player joblib bundle | `~/ME193/P1-Pose-Race/train_poses.py`, `collect_poses.py` |
| `shot.py` | Speed map, spin vector, aim, quality, deterministic fault | New |
| `vision.py`, `pose_features.py`, `pose.py`, `oneeuro.py`, `paddle.py` | Capture, pose, hand points 17-22, One-Euro (min_cutoff 1.2, beta 5, d_cutoff 1.0), reach box, lag estimate | `~/ME193/P1-Pose-Race/pose_features.py` (L63, L76, L83), `pose_car.py` loop L313-366 |
| `find_apriltag.py`, `tags.py` | Detector (`make_detectors` L28, `find_tags` L38) plus TagVoter (4 of last 6, START 0.4 s, 1.5 s lockout, unallocated ids rejected, role gating by phase) | `~/ME193/P2-Apriltag-Parking/find_apriltag.py` |
| `physics.py` | Analytic Leg (6.1) | New |
| `judge.py` | HitJudge gates (6.4) | New |
| `policy.py`, `pd.py` | Opponent: PD paddle, softmax, miss model, Q-bandit | `~/ME193/P2-Apriltag-Parking/PD.py` `PDController` L75; `~/ME193/scripts/rl_straight.py` (ALPHA .1, GAMMA .9, EPS decay .98 floor .05) |
| `rules.py`, `modes.py`, `scoring.py` | Game FSM, Survival, Match, streak/record | Shape of `whistle_car.py` `Game` L178 |
| `mqtt_pub.py` | ScorePublisher (section 8) | `whistle_car.py` `Radio` L134-170 upgraded; raw paho |
| `store.py`, `leaderboard.py`, `webboard.py` | SQLite plus writer thread, queries, page | `~/ratengo/app/python/ratengo/store.py` L36-94, `server.py` L26-59 |
| `calibration.py` | Hands-free flow (hub button = next, audio prompts, auto-advance) | `collect_poses.py` countdown UX |
| `boardlink.py` | NodeLink, HTTP tx, UDP rx, alive flag | `~/ratengo/scripts/demo_server.py` |
| `canvas.py`, `hud.py` | cv2 drawing, large-glyph HUD, x-ray panel | `pose_car.py` `draw_hud` L215 |
| `audio.py` | Pre-rendered numpy sounds, one persistent sounddevice stream pinned to built-in speakers, `afplay` fallback | `~/ME193/P3-Whistle-Soccer/songs.py` `note_hz`, `synth` L23-49 |
| `sources_fake.py` | FakeImu, FakePose, FakeTags, FakeHub, FakeMqtt, FakeNode | `~/ratengo/tests/python/conftest.py` L14-31 |
| UNO Q app | Stdlib-only Python + sketch | `~/ratengo/app/python/main.py`, `board.py`, `server.py`; `app/sketch/sketch.ino` L37-135, `sketch.yaml`, `src/matrix_render.cpp`, `src/reading_queue.h`; `scripts/deploy.sh`; `Makefile` |

`requirements.txt` (P1 pins, comments copied; do not upgrade):
```
# mediapipe 1.x aborts on Apple silicon; 0.10.21 forces numpy<2, which forces opencv 4.10.0.84.
# opencv-contrib-python must match opencv-python exactly.
mediapipe==0.10.21
numpy<2
opencv-python==4.10.0.84
opencv-contrib-python==4.10.0.84
scikit-learn==1.9.1
joblib==1.6.0
legoeducation==1.1.1
bleak==3.0.2
paho-mqtt==2.1.0
sounddevice==0.5.6
pytest
```
Freeze to `requirements.lock` after the first working install (the P1 venv has drifted from its file). Python 3.12 only (mediapipe 0.10.21 has no cp313 wheel (V)).

### 5.3 SwingDetector (pure, tested offline; replaces the magnitude-only FSM)

Units: every threshold is stored in dps through `GYRO_PER_DPS` (P2) or derived from Day 1 fixture percentiles (rest noise, soft peak, hard peak). A unit test rescales the same synthetic fixture by 10x and asserts identical behaviour after the calibration constant changes.

Forward axis: from the P3/Day 1 and calibration swings, `u_fwd` = principal axis (SVD) of the gyro vectors at the peak, sign-fixed so the forward pulse is positive; stored in the profile. Signed rate `s = (g - bias) . u_fwd`; magnitude `w = |g - bias|`. Bias re-zeroed when idle (`w < 25 dps` for 0.3 s).

```
Clock: nominal grid period P (probe P1); offset c = min over a sliding window of (arrival_k - P*k);
       on a gap > 3P advance k by round(gap/P).  Samples inside any haptic blank window are ignored.
IDLE : if s > ARM and rising: t0 = t -> FWD   (backswing projects NEGATIVE and never arms)
FWD  : track (t_pk, w_pk) on raw/median-3 magnitude (never EMA); parabolic interpolation on 3 raw samples
       emit SWING_START when s > 0.5*T_PK
       abort if t - t0 > 0.45 s or (s < 0.3*ARM and w_pk < T_PK)
       fire IMPACT(t_pk) when w_pk >= T_PK and (w < 0.9*w_pk or two consecutive falling samples)
            and (t - t0) >= 0.04 s and n_reversals(prev 0.8 s) <= 2 and dur in 60-400 ms
COOL : refractory 0.30 s and w < 100 dps -> IDLE        # the judge adds 0.35 s only after a counted hit/fault
T_PK = 0.6 * omega_lo (player's weakest deliberate swing), ARM = 0.4 * T_PK
```
Oscillation guard: waving back and forth along the same axis produces repeated positive lobes, so a swing is suppressed if the previous 0.8 s holds 3 or more sign reversals of `s` at amplitude >= 0.5 x w_pk. A real swing has one reversal (backswing then forward). IMPACT events are always emitted and logged, but they only become HITs inside a ball window (game-state gating), so practice waving never scores. Features (12 floats): unit(g_peak) 3, unit(net rotation) 3, unit(a_lin at peak) 3, log(w_pk), dur/0.4 s, back ratio. Clip flag: >= 3 consecutive samples at HUB_FS_RAW.

Degraded variant (only if P1 fails): the same FSM on pose wrist speed (lag-corrected) with hub accel magnitude as a gate; Rookie/Club windows only.

Acceptance on fixtures: IMPACT time within 30 ms of argmax |g| over the stroke and w_pk within 15% of the maximum; on clean synthetic swings |t_peak - true peak| <= 8 ms; backswing-only fixture gives 0 IMPACT; 10 soft swings give >= 9 IMPACT; 3 Hz waving gives 0 HITs.

---

## 6. Game design

### 6.1 Geometry and the ball (analytic Leg, no integrator)

World: table 2.74 x 1.525 m, net 0.1525 m. Hand maps from shoulder-width units to the calibrated reach box: a = (u - u_min)/(u_max - u_min), b likewise, with u = (c.x - P.x)/SW (frames fed to MediaPipe unmirrored; mirror only for display). Arrival points on a 3x3 grid at a, b in {0.15, 0.50, 0.85}. R is measured in SW units around the marker.

- Leg: `plan_leg(start, aim, v, spin, fault) -> Leg`. Base flight T = D/v with D = 3.0 m. Bounce at p_b = 0.55. Spin factor f = clip(1 + 0.25 T_spin, 0.75, 1.25). Final arrival time T' = max(0.30, T (p_b + (1 - p_b)/f)) (the floor is applied to the FINAL time, fixing the reviewer's finding that spin could undercut 0.30 s). Sidespin lateral curve 0.25 S sin(pi p) x 0.7625 m plus bounce kick 0.15 S x 0.7625 m. Apex height = min(0.30 m, 0.10 + 0.4 T^2) m so fast balls do not imply absurd accelerations (cosmetic). Fault legs end at the net or past the end line.
- Two speeds. Player to CPU: `v_out` (swing-controlled 3-14 m/s), used by the Match miss model and shown on the HUD. CPU to player: `v_in = v_tier x (0.9 + 0.2 s_prev) x r(n)`, where s_prev is the player's last shot strength (0.5 for the first ball) and r(n) is the Survival ramp (r = 1 in Match). The 0.9-1.1 modulation keeps the levels strictly ordered (Rookie max 3.85 < Club min 4.5, Club max 5.5 < Pro min 6.3, Pro max 7.7 < Insane min 8.55 m/s). T' floor 0.30 s always applies.

### 6.2 Difficulty table ((E) starting values; tuned by `tools/sim.py` and playtest)

| | Rookie (1) | Club (2) | Pro (3) | Insane (4) |
|---|---|---|---|---|
| v_tier m/s (T s at D=3 m) | 3.5 (0.86) | 5.0 (0.60) | 7.0 (0.43) | 9.5 (0.32) |
| Window early/late E/L (s) | 0.30/0.18 | 0.22/0.14 | 0.16/0.10 | 0.12/0.07 |
| Pose radius R (SW) | 0.55 | 0.45 | 0.35 | 0.28 |
| CPU reaction tau (s) | 0.40 | 0.28 | 0.18 | 0.10 |
| CPU paddle speed (m/s) | 1.2 | 2.2 | 3.5 | 5.0 |
| p0 / aim sigma (m) | 0.10/0.25 | 0.06/0.15 | 0.03/0.08 | 0.01/0.04 |
| Softmax temperature | inf | 1.0 | 0.4 | 0.15 |
| Spin variety | 0 | 0.2 | 0.4 | 0.7 |
| v_ref (m/s) / spin ref | 5/0.20 | 7/0.35 | 9/0.50 | 11/0.65 |
| Fault threshold F_TH | 0.60 | 0.50 | 0.42 | 0.35 |

Rookie and Club are the supported levels for G0 and G1 acceptance. Pro and Insane are best effort: Insane stacks 40-90 ms detection, 70-150 ms camera lag and human reaction against a 0.32 s flight and may be unplayable; it is retuned from recordings and honestly reported.

### 6.3 Per-shot speed, spin, aim, risk

- **Speed.** w_pk = peak |gyro - bias| in dps. s = clip((w_pk - omega_lo)/(omega_hi - omega_lo), 0, 1); omega_lo/omega_hi are the medians of 5 soft and 5 full calibration swings (defaults from Day 1 fixtures). v_out = 3.0 + 11.0 s^0.8 m/s; clipped swings count s = 1; T_out = max(0.30, 3.0/v_out); HUD shows km/h = 3.6 v.
- **Spin.** StandardScaler + LogisticRegression (C = 1) on the 12 features over flat/top/back/left/right, 8 swings per class, per player. T = (p_top - p_back) A, S = (p_right - p_left) A, A = 0.4 + 0.6 s. If max probability < 0.5 the shot is flat. Reported accuracy uses the last 2 swings per class as a held-out block (consecutive-swing CV leaks drift); below 0.70 fall back to 3 classes, with no model T = S = 0 and the HUD says SPIN: uncalibrated. Optional SPIN DIAL tag (id 20, G2-only): roll phi in +-45 degrees maps to S in -1..+1 when seen within 0.5 s of impact, read on a ROI at <= 10 Hz; never part of any acceptance gate.
- **Aim.** x_target = clip((a_paddle - 0.5) x 2 x 0.8, -1, 1) x 0.7625 m.
- **Quality.** q_pos = clip(1 - d_min/R, 0, 1); q_time = clip(1 - |e|/E_or_L, 0, 1); q_total = 0.5 q_pos + 0.5 q_time. Grades: perfect if q_total >= 0.9; early/late if e < -0.55 E or e > +0.55 L; otherwise good.
- **Risk (deterministic, explainable).** Fault if s (1 - q_total) > F_TH[level] (table 6.2). Out if s > 0.7, else net. A perfect hit has s(1 - q_total) <= 0.10 and can never fault; a sloppy smash faults. Faulted shots never increment the streak. The seeded RNG is used only for CPU behaviour.

### 6.4 Pose gating and the judge (every gate explains itself on the x-ray HUD)

A swing is a HIT only if:
- **G1 timing.** `t_i` in [t_c - E, t_c + L]. Earlier than t_c - E is ignored (practice swings are never punished); within [t_c - 2E, t_c - E] an early cue fires. No valid hit by t_c + L + D95 is a MISS.
- **G2 pose.** Using the pose snapshot samples with t_scene in [t_i - 0.30, t_i - 0.05] (approach window, available at t_detect; the ring is stamped `t_read - delta`, lookup at `t_i`, lag never subtracted twice): min distance from the paddle point to the ball plane point <= R; visibility >= 0.6 for at least 2 frames or 100 ms of coverage (whichever is lower, since the laptop camera can drop to 15 fps); AND the newest sample at t_detect is within 1.6 R of the marker (closes "touch then swing elsewhere").
- **G3 swing.** w_pk >= T_PK, duration 60-400 ms, oscillation count <= 2.
- **G4 cross-sensor consistency (logged only).** Pose wrist-speed peak within +-150 ms of the IMU peak; logged from Day 3, made hard only if measured false rejects are under 5% over at least 60 swings.
- **G5 refractory.** One hit per ball id; 0.35 s after a counted hit; at most 3 hits per second.
- **G6 shake lock.** High gyro energy for over 1 s with no ball window locks the paddle for 1 s.

Pose lost over 1.5 s or hub stale beyond threshold gives PAUSED, never a hit or fault. Hub gestures stay log-only unless probe P4 shows zero false events.

### 6.5 Calibration (hands-free, key C or hub button; safe defaults exist; about 7 minutes per full player, 60 s guest quick-tune)

1. Wave test 3 s (sharp irregular stabs): cross-correlate wrist speed with |gyro| over -50..+300 ms for paddle hand and lag delta. Treat delta as an effective pose-to-IMU offset (estimate repeatable within +-40 ms, reported).
2. Reach box: four corners x 1 s.
3. Speed and axis: 5 soft + 5 full swings give omega_lo, omega_hi and u_fwd (this IS the guest quick-tune).
4. Metronome offset: 5 beats (mandatory) give the constant timing bias.
5. Spin: 8 swings per class with a held-out block report (skippable).
Prompts are audio plus large-glyph HUD; the hub button advances; timeouts auto-advance.

### 6.6 AprilTag roles (36h11, tag 15 cm, white margin 2 cm, matte, on a stand beside the screen)

| ID | Role | Rule |
|---|---|---|
| 0 | START | >= 4 of last 6 frames and 0.4 s continuous, LOBBY only, 1.5 s lockout |
| 1-4 | LEVEL | latch in LOBBY; change only between rallies |
| 10-19 | PLAYER login (Tier 1) | LOBBY only |
| 20 | SPIN DIAL | optional, rally ROI |
| 40/41 | MODE Survival/Match (Tier 1) | LOBBY only |

Detection runs on TagWorker at 960 px grey in LOBBY (about 63 px per tag at 1.8 m for a 65 degree HFOV (E); 15 cm at 640 px would be only about 42 px). Unknown ids rejected. Keys: SPACE start, 1-4 level, M mode, P player, C calibrate, X x-ray, D disarm motors, H swap hand, L leaderboard, Q/ESC quit.

### 6.7 Modes and rules

- **SURVIVAL**: CPU never misses (paddle snaps to the intercept, return auto-lofted in). Ramp r(n) = min(1 + 0.03 n, 1.8) on speed; spin amplitude min(1, 0.10 + 0.03 n); P(wide) = min(0.8, 0.3 + 0.01 n); every 10th ball is a special (drop, lob, flat screamer). Ends at the first player fault; score = streak.
- **MATCH** (selectable difficulty): first to 7 (`--target 11` win by 2); rally scoring; v1 is receive-only (CPU serves every point; stated as a limitation, optional player-serve upgrade about 1 h after G1). CPU point on any player fault; player point on a CPU miss.
- **Streak/record.** Streak = consecutive valid returns this rally. Record = best streak this session (live source only, any mode), never decreases. Published value follows `PP_RECORD_SCOPE`.

### 6.8 Opponent policy

Perceive the ball after tau_react with noise; forward-simulate the Leg to the intercept; a speed-limited PD paddle (P2 `PDController`, gains 1.0/0.35, derivative low-pass 0.5) moves laterally. Target zone by softmax over U(z) = |x_z - x_p|/W + 0.6 lead - risk(z, v) (+ Q[s, z]); lead = sign(-v_p)(x_z - x_p); risk = 0.05 + 0.10 (v/v_max)^2 + 0.15 [edge]. Match miss model: P_miss = clip(p0 + k_v max(0, v_in - v_ref) + k_w max(0, spin_in - spin_ref) + k_d max(0, d_reach), 0, 0.95), with v_in here the player's `v_out`, d_reach = max(0, |x_land - x_cpu| - v_AI max(0, T_out - tau)), k_v 0.10/0.08/0.06/0.05, k_w 0.30/0.25/0.20/0.15, k_d 1.5/m. Fast, spinny, wide shots win points, so speed and spin control is the strategy.

Q-bandit (Day 5): 9 states (hand lateral bin x last return zone) x 9 zone actions, alpha 0.1, gamma 0.9, epsilon 0.15 (Pro) / 0.05 (Insane) decaying 0.98 to 0.05, reward +1 player fails, +0.3 weak return, -1 CPU fault, saved per player `data/ai_q_<slug>.npy`, pre-trained headless in `tools/sim.py`. Go/no-go wording: it learns a measurable zone preference against the scripted human (not "beats softmax by X points" against a self-written human); claimed in (c) only if shipped.

### 6.9 Haptic patterns (hub motors via `motor_run_for_time`, all pulses <= 400 ms, 1 pattern per 100 ms, <= 10 writes/s, intra-pattern gaps >= 80 ms, through the actuator thread)

| Event | Motors | Beep (Hz) | Light |
|---|---|---|---|
| READY / tag locked | 150 ms @40% both | 660 | BLUE BREATHE |
| Countdown 3-2-1 | 25 ms @70% x2, last THUMP 60 ms @100% | 880 on last | |
| HIT perfect | THUMP 60 ms @100% | 1760 | WHITE solid 150 ms |
| HIT good | 35 ms @70% | 1320 | GREEN |
| HIT early / late (flight >= 0.5 s only) | 2 x 20 ms @50% LEFT / RIGHT motor, gap 80 ms | 990 | ORANGE |
| Spin top / back (flight >= 0.5 s only, spin >= 0.3) | 3 x 20 ms both gap 80 ms / 1 x 150 ms @50% | | |
| Spin side | none (pitch up/down) | pitch | AZURE / MAGENTA |
| FAULT | 400 ms @60% same direction | 220 | RED LONG_BLINK |
| Point won | 2 ticks gap 100 ms + 200 ms buzz | double 880 | GREEN |
| Record tick | none | 2000 | PURPLE |
| New record (rally end) | 5 x 40 ms gap 120 ms | triple rising | colour cycle |
| Every 10th hit | 3 ticks gap 100 ms | triple | |
| Smash warning (optional, T >= 0.75 s) | ramp 25/50/75/100% x 100 ms | | ends >= 0.30 s before t_c - E |

Invariants: no pulse inside [t_c - E - 0.15, t_c + L + D95] except the post-impact hit pulse; at Pro and Insane (flight < 0.5 s) early/late and spin motor cues degrade to beep/light (free-time table per level is generated by a test). Blank window = from the actuator's actual write-return time minus 10 ms to end + 0.12 s, extended while any motor notification reports non-zero speed; the planned window is the backstop. Priority record > point > hit > tick. `--no-motor` mutes motors, keeps beep and light. Fallback chain: hub beep + light, Mac sound + screen shake, node LED3.

---

## 7. Player profiles, leaderboard, DB schema

Storage: stdlib sqlite3 `data/pingpong.db` (gitignored; remember to push code to GitHub before accounts close Dec 10). PRAGMAs `foreign_keys=ON`, `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=3000`. One StoreWriter thread owns writes (queue, commit per rally) and also serves read queries via futures. Schema is created with `CREATE TABLE IF NOT EXISTS` plus a `PRAGMA user_version` check that refuses a newer DB (no migration framework, no backups, no Elo this week). Raw streams go to `recordings/<session>/<rally>.jsonl`. Store raw speeds and spins, never level multipliers.

```sql
CREATE TABLE players(id INTEGER PRIMARY KEY, slug TEXT NOT NULL UNIQUE COLLATE NOCASE, display_name TEXT NOT NULL,
  handedness TEXT NOT NULL DEFAULT 'right' CHECK(handedness IN('right','left')), tag_id INTEGER UNIQUE,
  created_at TEXT NOT NULL, last_seen_at TEXT) STRICT;
CREATE TABLE calibrations(player_id INTEGER NOT NULL REFERENCES players(id),
  kind TEXT NOT NULL CHECK(kind IN('swing','spin','pose_box','lag','haptics','imu_bias')),
  json TEXT NOT NULL CHECK(json_valid(json)), updated_at TEXT NOT NULL, PRIMARY KEY(player_id,kind)) STRICT;
CREATE TABLE sessions(id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES players(id), started_at TEXT NOT NULL,
  ended_at TEXT, source TEXT NOT NULL CHECK(source IN('live','demo','sim','replay','fake')), app_version TEXT,
  rig_json TEXT, record_hits INTEGER NOT NULL DEFAULT 0) STRICT;
CREATE TABLE matches(id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
  mode TEXT NOT NULL CHECK(mode IN('survival','match')), level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 4),
  target_points INTEGER, player_points INTEGER NOT NULL DEFAULT 0, cpu_points INTEGER NOT NULL DEFAULT 0,
  best_streak INTEGER NOT NULL DEFAULT 0, outcome TEXT CHECK(outcome IN('win','loss','ended','abandoned')),
  started_at TEXT NOT NULL, ended_at TEXT) STRICT;
CREATE TABLE rallies(id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL, hits INTEGER NOT NULL, duration_s REAL,
  end_reason TEXT NOT NULL CHECK(end_reason IN('miss','late','early','out','net','timeout','quit','cpu_miss')),
  point_to TEXT CHECK(point_to IN('player','cpu')), max_speed REAL, max_spin REAL, UNIQUE(match_id,seq)) STRICT;
CREATE TABLE hits(id INTEGER PRIMARY KEY, rally_id INTEGER NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,
  n INTEGER NOT NULL, t_ms INTEGER, in_speed REAL, out_speed REAL, topspin REAL, sidespin REAL, spin_class TEXT,
  quality TEXT CHECK(quality IN('perfect','good','early','late')), timing_err_ms REAL, swing_peak_dps REAL,
  d_min_sw REAL, UNIQUE(rally_id,n)) STRICT;
CREATE TABLE publish_log(id INTEGER PRIMARY KEY, session_id INTEGER, at TEXT NOT NULL, topic TEXT NOT NULL,
  payload TEXT NOT NULL, qos INTEGER, retain INTEGER, acked_at TEXT, echo_ms REAL) STRICT;
```
(DDL executed on SQLite 3.51 and 3.53 in research (V); re-run in `tests/test_store.py`.)

**Profiles.** `tools/register_player.py rafae --tag 10 --hand right`; PLAYER card or key P logs in and loads handedness, calibrations and Q file; no card = guest (quick-tune, no spin, not on the leaderboard).

**Leaderboards (live rows only).** Survival: best streak per player per level, earliest wins ties, using the verified `RANK() OVER (ORDER BY best DESC, at ASC)` CTE. Match: wins and win% per level. Per-player stats: average and max out_speed, max spin, accuracy hits/(hits+misses), perfect%, mean timing error, longest rally. For rehearsal only, `--seed-demo` inserts rows tagged `source='demo'` shown with a DEMO badge (never on the official board, never published). On Day 5 recruit 2-3 classmates or roommates for 20 minutes of guest play so the board is not a single name.

**Surfaces.** End-screen top-5 plus rank; `http://localhost:8350` (stdlib `ThreadingHTTPServer`, pure `handle(path, store)`, `/`, `/api/boards`, `/api/player/<slug>`, polled every 2 s; cut line 4); node matrix (streak digits plus progress bar vs record); `tools/report.py` prints best streak, accuracy, echo latency, publish count, haptic counts for the write-up.

---

## 8. MQTT contract

- **Broker** `test.mosquitto.org:1883`, plain TCP, anonymous, MQTT 3.1.1, keepalive 30; constant in `config.py`, override `PP_BROKER=host[:port]`. Raw paho 2.1.0 (instructor `mqttlib` dispatches by exact topic and never fires for wildcards (V)).
- **Official topic** `ME193/Rogers/RafaeShafi` (exact, case-sensitive).
- **Payload** `f"{float(x):.1f}"` e.g. `"0.0"`, `"17.0"`; never JSON, NaN or negative (paho `str()`s ints, so always format explicitly).
- **Value** by `PP_RECORD_SCOPE`: `record_session` (default) = max(session record, current streak); `record_alltime` = same seeded from the profile best; `live_streak` = current streak. All three unit-tested. A hit is a player return that passed every gate and was not a fault; CPU returns never count.
- **When.** (1) On the FIRST increase of the published value (so the first rally ticks 1.0, 2.0, 3.0 ... live); (2) on every further increase; (3) on reconnect, republish the current value only if it is above 0.0; (4) at session end. **No 0.0 publish at startup**, so a rehearsal later cannot clobber a good retained value. The graded run is a fresh session (no resume unless `--resume`, intended for crash recovery only). `--new-record` (with an on-screen confirm) explicitly publishes `0.0`. At startup the client subscribes to its own topic and prints the retained value on the console and HUD ("broker currently holds 30.0") without acting on it.
- **QoS/retain.** QoS 1, retain True (late joiner sees the value; QoS1 queues offline). Clear deliberately with `mosquitto_pub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi -n -r`.
- **Client.** `mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pp-rafae-"+uuid4().hex[:8])`, `reconnect_delay_set(1,30)`, `connect_async()` + `loop_start()` so the game boots offline, never `wait_for_publish` in the game loop (2 s at shutdown only).
- **LWT** `will_set("ME193-pp/RafaeShafi/status","offline",qos=1,retain=True)` plus retained `online` on connect. Never a will on the score topic.
- **Extras (all outside `ME193/`)**: `ME193-pp/RafaeShafi/status`, `.../demo/score` (the only place demo/sim values go), optional `.../state` telemetry (`--telemetry`, at most 5 Hz, QoS0). Cut this week: `event`, retained `leaderboard`, TAMPER republish.
- **Echo.** Subscribe to the own topic at QoS1 for `echo_ms` on the HUD and in `publish_log`. Ignore `retain=1` deliveries at subscribe; match echoes against the set of payloads sent in the last 10 s; a foreign differing live message is only logged (no republish).
- **Integrity invariants (unit-tested).** Official topic only when `session.source == "live"` (imu=hub, pose=cam, assist off, not replay/sim/fake/demo; tag source does not gate); payload matches `^\d+\.0$`; QoS 1 and retain; record monotone under a seeded random-sequence test; no will on the score topic; fake/demo publish nothing there.
- **Verify like the instructor.**
```
mosquitto_sub -h test.mosquitto.org -t 'ME193/Rogers/#' -F '%I %t [%p] retain=%r qos=%q'
mosquitto_sub -h test.mosquitto.org -t ME193/Rogers/RafaeShafi --retained-only -C 1 -W 5
my_env/bin/python tools/watch_score.py
```
- **Unverified (U).** That the broker accepts a retained QoS1 publish on the real topic (research only subscribed; P9 uses a scratch topic, first official publish is Day 3); that the instructor tool parses a bare float and tolerates retain; the meaning of "record". Mitigation: email Prof. Rogers tonight (Day 0) with a Day-3 checkpoint; the three-value switch makes any answer a one-line change.

---

## 9. AI/ML algorithms (claim only what ships; two sentences or fewer)

| Algorithm | Where | How it works |
|---|---|---|
| MediaPipe BlazePose landmarker (pre-trained CNN, lite) | `pose.py` | A convolutional network regresses 33 body landmarks per frame and tracks them between frames. I use wrist, index and pinky normalised by shoulder width as the paddle position. |
| AprilTag / ArUco 36h11 (classical vision) | `tags.py` | It thresholds the image, finds square quads and decodes a Hamming-protected bit grid into an ID and corners. I vote over frames for START and LEVEL. |
| One-Euro filter | `oneeuro.py` | A low-pass filter whose cutoff rises with signal speed. The paddle point is smooth when still and nearly lag-free in a swing. |
| StandardScaler + Logistic Regression | `spin.py` | It standardises 12 swing features and learns a linear softmax boundary over flat/top/back/left/right from my labelled swings. Probabilities become continuous topspin and sidespin. |
| Swing detector (signed-axis threshold FSM, non-learning; axis from SVD/PCA of calibration swings) | `swing.py` | It projects gyro onto the learned forward axis, arms on a threshold, tracks the peak and fires on the falling edge with an oscillation guard. This rejects backswings, waving and double triggers. |
| PD controller (from P2) | `policy.py`, `pd.py` | Command = Kp x error + Kd x error rate, saturated at a per-level speed after a reaction delay. Whether the CPU reaches a ball is physical, not a coin flip. |
| Softmax (Boltzmann) policy | `policy.py` | Each of 9 target zones gets a utility and is sampled with probability proportional to exp(utility/temperature). Low temperature plays sharper at higher levels. |
| Tabular Q-learning bandit (Day 5; claimed only if shipped) | `policy.py` | It keeps Q(s,a) and updates Q += alpha(r + gamma max Q' - Q) with epsilon-greedy exploration. State is hand position plus last zone, action is the target zone, reward is whether I fail. |
| FFT whistle detector (stretch START input; claimed only if shipped) | `whistle_policy.py` | A Hann-windowed FFT finds the strongest tonal peak in 500-3500 Hz gated by SNR and tonality. A confirmed whistle starts the game. |

Not claimed: RK2/Magnus, ridge regression, Elo, MPU fusion.

---

## 10. Testing strategy

**Principle.** Everything that decides anything is pure and clock-injected, so about 90% runs with no hardware; hardware sits behind HubLink, VisionWorker, TagVoter, Actuator and the paho client, each with a fake. `python play.py --fake --mqtt off` runs the full game on a laptop.

**Fakes.** FakeClock; FakeImuStream (half-sine swings with 15 ms bunched jitter, backswing-then-forward pairs, 3 Hz waves, shake, idle noise, haptic bursts, JSONL replay of Day 1 fixtures); FakePose (scripted path, occlusion, flip, lag); FakeHub (records calls); FakeMqttClient; FakeNode (UDP loopback with jitter and loss); FakeBridge/FakeApp from `~/ratengo/tests/python/conftest.py` L14-31.

**Unit and property tests (pytest, under 20 s).**
- `test_swing`: >= 95% synthetic detection; backswing-only gives 0 IMPACT; forward pulse after a larger backswing IMPACTs on the forward peak; zero HITs on 3 Hz waving; zero events inside blank windows; peak-time error <= 8 ms; clip flag; 10x unit rescale invariance.
- `test_judge`: each gate fails independently; touch-then-swing-elsewhere rejected; early ignored; late/no-hit is a MISS at t_c + L + D95; swing at t_c + L - 0.06 (detected after the plane) still counts.
- `test_physics`: deterministic arrival; final T' >= 0.30 s for every spin and ramp; topspin arrives faster after the bounce; sidespin direction.
- `test_shot`: speed monotone in w_pk; fault is a pure function of s(1 - q_total); a perfect hit never faults; level ordering of v_in holds for every s_prev in [0,1].
- `test_policy`: Survival never misses over 10,000 seeded decisions; Match miss rate monotone in level, speed and spin; ramp never gives T below 0.30 s.
- `test_scoring`/`test_rules`: record monotone under 1000 seeded sequences; idempotent hit ids; first-to-7, deuce at 11; the three RECORD_SCOPE values.
- `test_mqtt`: payload regex, QoS1, retain, no 0.0 at startup, first publish at first hit, republish on reconnect only if > 0, offline queue, official topic refused unless live, no will on the score topic, retained-at-subscribe ignored.
- `test_store`: fresh build, leaderboard tie-breaks, live-only filter, newer-DB refusal. `test_actuator`/`test_haptics`: blank-window math from actual write times, priority, spacing >= 100 ms, <= 10 writes/s, no-pulse invariant, free-time-per-level table, teardown order, `--no-motor`. `test_tags`: synthetic aruco images plus voter. `test_web`: `handle()`. `test_oneeuro`. UNO Q: `make -C uno_q test` compiles `matrix_render.cpp` and `ldr_cover.cpp` (`clang++ -std=c++17 -Wall -Wextra -Werror`) and replays traces; pytest for the board app with FakeBridge.

**Integration.** pytest fixture spawns `/opt/homebrew/sbin/mosquitto -p <random>` (V) for retained delivery and reconnect. Optional `PP_LIVE_MQTT=1` live smoke to the scratch topic only. **Walking skeleton (Day 1 exit test):** a 5-minute process with camera thread, imshow, live hub IMU callback and a paho publish all running, no game logic, from the canonical terminal.

**Headless sim.** `python tools/sim.py --rallies 200 --seed 1 --level 3` (source=sim) with a scripted noisy human balances levels and pre-trains the Q-bandit; R3 flight-time ratio is measured here at fixed swing strength. `play.py --imu replay:FILE` replays a rally (source=replay, never published).

**On-hardware acceptance.** Day 1-7 tests in section 11, a 15-minute soak, link-loss drill (pull hub battery), Wi-Fi-off drill, venue rehearsal. Done = Tier 0 pytest green + G0 live acceptance + backup video; Tier 1 = G1 + 30-minute soak; Tier 2 = G2.

---

## 11. Milestones: day-by-day (about 38 h total; hours are student hands-on plus Claude-coding, so the real student time is lower; treat as an upper bound)

If any task passes 2x its estimate, replace it with its documented fallback. Claude Code builds pure modules in parallel (suggest `cs`) while the student runs hardware tasks.

### Day 0, Tue 2026-10-06 (tonight, 1.5 h)
Email Prof. Rogers (one line; section 13.11). Scaffold, venv from the P1 pins, copy edits (13.4). Camera-open plus bleak-scan from the canonical terminal, grant Camera and Bluetooth. Read the hub card colour and serial, charge and weigh the hub. Write and run `tools/make_cards.py`, print ids 0-4 (spare copies) on the Brother printer and mount on foam board. Read the resistor color bands. Order Dupont jumpers and a mini breadboard. Review `git status` at `~/ME193` and decide how to handle the staged files.

### Day 1, Wed 2026-10-07: bench the unknowns, hardware-free slice (6.0 h)
| Task | h |
|---|---|
| 1.1 Finish scaffold, pytest smoke, `.gitignore`, lock file | 0.25 |
| 1.2 Probes P1-P5 (hub rate at play position on both networks, units, clipping, gestures, weigh); fixtures: 20 soft + 20 hard swings, 10 backswing-only, 10 shakes, 30 s waving | 1.5 |
| 1.3 P6 (6 recipes) and P7 | 0.75 |
| 1.4 P8 vision and tag range, P11 audio latency | 0.5 |
| 1.5 P9 retained MQTT on scratch topic, both networks | 0.5 |
| 1.6 Walking skeleton (camera + imshow + hub callback + paho) | 0.75 |
| 1.7 `types`, `clock`, `physics`, `scoring`, `rules`, `sources_fake`, `canvas`/`hud`, `play.py --fake`; Claude also builds `mqtt_pub.py` with tests and the README answers skeleton | 1.75 |

**Acceptance:** `my_env/bin/python -c "import mediapipe, cv2, legoeducation, bleak, paho.mqtt.client, sounddevice, sklearn; assert cv2.aruco.DICT_APRILTAG_36h11"` passes and a dummy `detect_for_video` runs (no `pip check`); every probe number and GO/NO-GO note is in `config.py`; P1 decision (hub swing source vs pose fallback) made by noon; scratch round trip < 2 s; `play.py --fake --mqtt off` reaches 10 consecutive hits at Rookie within 5 minutes; >= 15 pytest tests green. Day 1 hard stop for benches at 5 h; unfinished probes carry defaults forward.

### Day 2, Thu 2026-10-08: sensing to events (5.5 h)
`hub.py` + `actuator.py` + watchdog + teardown (1.0); signed `swing.py` against fixtures, tuned (1.25); `vision.py`, `pose.py`, `oneeuro.py`, `paddle.py`, hands-free calibration with wave lag and hand (1.5); `tags.py` voter on TagWorker (0.5); full-screen large-glyph HUD tested at 1.8 m (0.5); tests (0.5); IMPACT-vs-argmax report and D95 from fixtures (0.25).
**Acceptance:** >= 19/20 real swings give an IMPACT with time within 30 ms and amplitude within 15% of the fixture argmax; >= 9/10 soft swings; backswing-only gives 0; after the first 0.5 s no IMPACT passes G2/G3 during 30 s of 3 Hz waving; 30 s at rest gives 0; wave test picks the correct hand and delta repeatable within +-40 ms over 3 runs; paddle ghost within one hand-width in fast motion; START held 0.4 s fires once and 0.2 s never; level tags 1-4 latch 20/20 at 1.8 m; main-loop p95 < 20 ms with all threads; Ctrl-C tears down BLE and the hub reconnects in < 5 s; HUD readable at 1.8 m.

### Day 3, Fri 2026-10-09: ASSIGNMENT MINIMUM, gate G0 (5.5 h)
`judge.py` G1-G3, G5 hard, G4 logged, x-ray HUD (1.25); Survival-lite CPU with Rookie/Club windows and START/LEVEL tags wired (0.75); live integration and tuning (1.25); first official publish and live `watch_score.py`, echo on HUD (0.5); haptics H0 (hit pulse, miss buzz, countdown, beep and light) with blank windows and `--no-motor` (0.5); source guard and PAUSED on link loss (0.5); README answers draft, 60-90 s insurance video, push with pathspec commit, tag `v0-minimum` (0.75). Per-rally recordings and x-ray polish move to Day 4.
**G0 acceptance (real webcam, hub, printed tags, real broker; Rookie and Club):** (1) START tag gives 3-2-1 then a ball; (2) 10 consecutive in-position swings register 10 hits and `mosquitto_sub -t 'ME193/Rogers/#'` shows 1.0 ... 10.0 live, holds after a miss, a retained late joiner returns the record; (3) hand more than 2R away gives no hit 10/10, hand on the ball without a swing and hub shaken with arm still give none; (4) level 1 vs 4 changes flight time (log); (5) a haptic pulse is felt or beep and light fire; (6) `--fake` and `--imu fake` publish nothing to the official topic; (7) pytest green and a 15-minute soak clean. Also check Prof. Rogers's reply (checkpoint). **Late-evening go/no-go: if 1-4 fail, Day 4 is fix-only and Day 6 extras are cut first.**

### Day 4, Sat 2026-10-10: speed and spin, full haptic language (5.5 h)
`calibrate_swing.py` and `shot.py` speed map, deterministic fault, km/h on HUD (1.25); `spin.py` collection, classifier, arcade spin effects (1.5); full haptic library, priority queue, blind identification test (1.0); `audio.py` whoosh, pitch = speed, timbre = spin (0.5); per-rally recordings, x-ray polish, level retune from recordings (0.75); P10a UNO Q wire-free smoke, only if G0 is green (0.5).
**Acceptance:** 5 soft vs 5 hard swings median out-speed ratio >= 2.0; held-out spin accuracy >= 0.80 (else 3 classes, reported honestly) and >= 70% live per class; the student identifies {perfect, early, late, miss} >= 80% over 20 blind trials at Club (Pro/Insane verified as beep/light cues); 0 false IMPACTs over 100 consecutive in-rally pulses; saturated swings flagged without crash; R3 flight-time ratio >= 2.0 on the CPU-serve sim.

### Day 5, Sun 2026-10-11: modes, difficulty, Q-bandit, profiles, leaderboard, gate G1 (5.5 h)
`policy.py` PD, softmax, miss model, plus `modes.py` Survival ramp and Match to 7 (1.5); Q-bandit with sim pre-train (0.75); `store.py`, profiles, calibrations, publish_log, StoreWriter (0.75); `leaderboard.py`, end-screen top-5, tag/key login and mode (0.75); `webboard.py` :8350 (0.5, cut line 4); `sim.py`, tests, 15-minute soak, classmate guest session (0.75). Buy or borrow any missing jumpers tonight.
**G1 acceptance:** sim of 200 rallies per level shows CPU miss rate monotone Rookie > Club > Pro > Insane; live Survival ramps visibly and ends on the first player fault; a full Match to 7 completes at Club; stop and restart keeps the profile and calibration; page and end screen show the same top-5; Q-bandit shows a measurable zone preference in sim (else cut). Tag `v1-features`.

### Day 6, Mon 2026-10-12: SUBMIT-READY, then UNO Q extras (6.0 h)
Morning (3.0 h): README final with measured numbers from `tools/report.py` (0.75); Notion page with GitHub link, code block, video, diagram, MQTT table, answers (a)(b)(c) and reflection (0.75); 45-minute explain-back session where the student walks through swing detector, judge, shot mapping, spin classifier and opponent, and rewrites (a)(b)(c) and the reflection in their own words (0.75); reconcile (a)(b)(c) against the tagged release (0.25); final backup video with `mosquitto_sub` on screen (0.5). Tag `v1-submit-ready`, push with pathspec, confirm the retained topic equals the last live record. **Submit the Notion link now if the due time allows.**
Afternoon (3.0 h, additive; cut order shown): 6a scoreboard `deploy.sh`, app, sketch, `node.py` (1.5, cut late); 6b LDR cover-to-serve with calibration step (0.5, cut early); 6c MPU bench ruler 30 paired swings, slope and R^2 below 2000 dps, measured MCU loop rate (1.0 hard cap, cut early); P10b wired smoke first. Whistle-to-serve only if everything above is green by hour 2.
**Acceptance:** board mirrors streak and record within 300 ms from a cold start; unplugging it or restarting the board app mid-game only flips a NODE LOST badge and recovers in 1-2 s (frame time spike < 5 ms); cover starts the countdown in <= 0.5 s with 0 false starts in 5 minutes of room-light changes; MPU ruler R^2 >= 0.9 (else reported as a bench-only experiment); pytest green; 15-minute soak with zero crashes. Re-tag and refresh the video only if the extras changed the demo.

### Day 7, Tue 2026-10-13: rehearse, drill, buffer (3.0 h; no new features)
Failure drills (hub asleep, board off, broker down, tags unreadable, pose lost, motors dead, camera glare), each ending with the game continuing or pausing cleanly and a clear HUD message (1.0); full demo rehearsal in the venue if available, local-mosquitto swap via `PP_BROKER` in under 60 s, `--fake` path verified, the final graded fresh-session live run in the network configuration that will actually be used (1.0); fresh clone plus `pip install -r requirements.txt` plus `pytest -q` plus `python play.py --fake` (0.5); buffer and submission if not already done (0.5).
**Instructor demo script:** START tag, level tag change, one Survival record rally with the watcher on screen, one short Match (Club to 7), x-ray HUD and speed/spin gauges, haptic on camera, leaderboard page on the phone, matrix scoreboard.

### Cut lines (drop first to last; each cut keeps the demo working)
1. Whistle-to-serve; 2. MPU bench ruler; 3. LDR cover-to-serve; 4. web leaderboard page (keep end-screen top-5); 5. sonification extras and any commentator; 6. UNO Q matrix scoreboard; 7. spin dial tag and tag-based player/mode login (keep keys); 8. spin classifier: 5 classes to 3 to flat; 9. Match difficulty variants (keep one) and Pro/Insane tuning; 10. Q-bandit (keep softmax; state honestly which class topics are used); 11. haptic language shrinks to the 4 core patterns.
**Never cut:** pose gate, IMU swing gate, tag START and LEVEL, live MQTT record, Tier-1 haptics (H0), `--fake`, written answers, the explain-back session.
**Gate rules:** G0 fails, Day 4 is fix-only. G1 fails, Day 6 afternoon extras are cut and Day 6 morning stays submit-ready work. Hardware tiers are gated by acceptance; pure modules are not gated and are built ahead.

---

## 12. Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Schedule slip (about 38 h, other classes) | High | Gates, cut order, 2x rule, `--fake` from Day 1, submit-ready Day 6, parallel Claude coding |
| Swing detector fires on backswing or waving | Was a blocker | Signed axis, oscillation guard, window gating, fixtures from Day 1, backswing-only test |
| Hub IMU below 66 Hz, units/range unknown (U) | Medium | P1-P3 at the play position; fixture-percentile thresholds; below 25 Hz uses pose wrist-speed detector, Rookie/Club only |
| Detection lag makes the late window unreachable at Pro/Insane | High | MISS at t_c + L + D95, back-dated timestamps, 0.9-peak trigger, Pro/Insane best effort |
| Camera lag 70-150 ms (E) | High | Measured delta, approach-only window, One-Euro, generous Rookie radius |
| Haptics weak or vibrating the IMU | Medium-high | Masses (secured), blank from actual write times plus motor notifications, no pulse in swing window, beep/light/Mac sound guaranteed, honest in write-up |
| BLE writes stall loop; writes dropped silently | Medium | Actuator thread, 10 writes/s cap, batching |
| Hub sleeps or ghost connection (about 24 s) | Medium | Watchdog threshold from histogram, extend ball clock for short stalls, reconnect with 15 ms, teardown order, full charge |
| Network: hotspot cellular dependence and 2.4 GHz coexistence | Medium | Benches on both networks; 5 GHz hotspot; demo rule: if the venue has no cell signal use venue Wi-Fi and accept a dark node; final live run in the real configuration |
| MQTT: instructor parser/semantics (U), clobbering, spoofing | Medium | No startup 0.0, three-value scope switch, email Day 0, publish_log, local broker fallback, recorded `mosquitto_sub` |
| Thread races (ring buffer, record, SQLite from main) | Medium | Immutable snapshot, locks, writer-thread queries |
| Env pin drift | Low | P1 pins, lock file, never upgrade |
| macOS permissions and GUI loop | Medium | Canonical terminal, VideoCapture on main thread, walking skeleton Day 1 |
| Spin classifier does not generalise | Medium | Held-out block, 3-class fallback, flat when uncertain, recalibrate per session |
| UNO Q flakiness (IP drift, hotspot UDP return untested, RAM sketch wiped on reset, boot time, one app at a time, Bridge Linux-to-MCU untested) | Medium | Strictly additive, alive flag, 1 Hz keepalive and broadcast discovery, `provide_safe`, P10a/P10b, deploy stops RateNgo |
| LDR resistor mismatch; MPU tether and header soldering | Medium | Measure R, calibrated thresholds, abort below 3x; 100 kHz, <= 40 cm; 1.0 h cap |
| Mass or hub flying off during swings | Low-medium | Secured masses, lanyard, 20-swing retention test, forearm mount fallback |
| Insane unplayable | Medium | Best effort, retune from recordings, report honestly |
| Hardware or venue unavailable on Day 7 | Medium | Submit-ready Day 6, backup videos Day 3 and Day 6 |
| Leaderboard near-empty | Medium | Classmate guest session Day 5, `--seed-demo` for rehearsal only with DEMO badge |
| Monday 2026-10-12 may be a no-class holiday (unverified secondary web sources) | Unknown | Plan does not rely on it; printing and instructor email done Day 0 |

---

## 13. Pre-flight setup checklist (Day 0, about 1.5 h; exact commands)

1. **Hub card.** Read colour and 4-digit serial from the Connection Card or hub; keep the serial as a string. Charge the hub, wake it with its button, weigh it.
2. **macOS permissions.** System Settings > Privacy > Camera and Bluetooth: allow the ONE canonical terminal; run all Python from it. Test: 10-line camera-open plus bleak scan.
3. **Create the project and venv:**
```
cd ~/ME193 && mkdir -p P5-Ping-Pong/{pingpong,tools,tests,models,docs/cards,uno_q,data,recordings} && cd P5-Ping-Pong
python3.12 -m venv my_env
# write requirements.txt from section 5.2
my_env/bin/pip install -U pip && my_env/bin/pip install -r requirements.txt
my_env/bin/python -c "import mediapipe, cv2, legoeducation, bleak, paho.mqtt.client, sounddevice, sklearn; assert cv2.aruco.DICT_APRILTAG_36h11; print('ok')"
my_env/bin/pip freeze > requirements.lock
```
(`pip check` prints "mediapipe 0.10.21 is not supported on this platform"; that single message is expected and tolerated.)
4. **Copy and edit reusable files (originals untouched):**
```
cp ../P1-Pose-Race/pose_landmarker_lite.task models/
cp ../P1-Pose-Race/pose_features.py pingpong/        # edit: model path -> models/pose_landmarker_lite.task
cp ../P2-Apriltag-Parking/find_apriltag.py pingpong/ # edit: import of make_apriltag FAMILIES -> inline dict
cp ../P2-Apriltag-Parking/make_apriltag.py tools/    # reference only; cards come from tools/make_cards.py
cp ../P2-Apriltag-Parking/PD.py pingpong/pd.py       # keep PDController only
# whistle_policy.py: copy on Day 6 only if the stretch runs (it does `import config`)
```
5. **.gitignore** in `P5-Ping-Pong/`: `data/`, `recordings/`, `calibration*.json`, `__pycache__/`, `*.pyc` (root already ignores `**/my_env/`).
6. **Print cards.** `my_env/bin/python tools/make_cards.py` writes `docs/cards/*.png`: tag 15 cm, 2 cm white margin (19 cm, fits Letter), ids 0-4 now; 10-12, 20, 40-41 on Day 5. Matte paper, mount on foam board. Phone screen at full brightness is the fallback.
7. **Paddle prep.** Tape arrow on the hub, secured shaft masses, wrist lanyard, optional cardboard blade; kitchen scale.
8. **Tools.** `which mosquitto_sub mosquitto_pub` (brew mosquitto 2.1.2 installed (V)); `my_env/bin/pytest --version`.
9. **UNO Q (Day 4/6).** Hotspot on (5 GHz), board powered, `ssh arduino@172.20.10.2` (set `PP_NODE_IP` if the IP differs), `arduino-app-cli app list`. Note deploying stops RateNgo. Confirm the hotspot subnet broadcast address with `ifconfig`.
10. **Parts.** Order female-to-male Dupont jumpers and a mini breadboard (or clips) now (arrive by Day 3-5); check GY-521 headers are soldered and the header gender; read the resistor color bands.
11. **Email Prof. Rogers.** "Is a bare float string like 12.0 on ME193/Rogers/RafaeShafi correct, is retain fine, and does 'record' mean the session-best continuous-hit streak (player returns only)? Will the demo be live or recorded?" Proceed on the defaults if unanswered; reply is a Day-3 checkpoint.
12. **Git hygiene.** `git -C ~/ME193 status` and `git diff --cached --stat`; ask the student whether to commit or unstage the unrelated staged files (P1 `pose_data.csv`, `pose_model.joblib`, `Practice.py`, `scripts/*`, `.gitignore`); commit only with `git commit P5-Ping-Pong/ -m ...`; never `git add -A`.

---

## 14. Draft answers (conditional: reconcile against the tagged release on Day 6; replace bracketed placeholders with measured numbers)

### (a) Describe the policy: how does it make decisions?
My game is a layered policy where each layer is a small explicit rule I can test without hardware. The perception layer turns raw sensors into events. A state machine on the hub's gyro projects rotation onto my learned forward swing axis, arms on a threshold, fires at the peak and rejects backswings and waving. AprilTag IDs must be seen in at least 4 of 6 frames before they count as START or LEVEL, and pose is accepted only when landmarks are visible. The hit judge is a conjunction: the swing's back-dated time must fall inside a timing window around the ball's known arrival, my paddle hand (from pose, shifted by the measured camera lag, and still near the marker at detection) must be within a level-dependent radius of the ball, the swing must be big and clean enough, and no refractory or shake lock may apply [if G4 hardened: and pose and IMU must agree on when it happened]. Each failed gate appears on screen as an x-ray checklist. The shot policy converts the swing into a shot: peak gyro rate sets ball speed, a logistic-regression classifier on rotation features [or an AprilTag dial] sets spin, my hand position sets aim, and a deterministic risk rule (speed x sloppiness against a per-level threshold) decides net or out faults. The opponent chooses a target zone by a softmax over a utility that wrong-foots my tracked hand [with a learned Q bonus if shipped], executes through a delayed, speed-limited PD paddle, and in Match misses with a probability that grows with my shot's speed, spin and reach; in Survival it never misses and only the ramp changes. A rules state machine converts hits and faults to the score, the haptic policy maps each event to a priority-ranked pattern on the hub, and the record of continuous hits is published to MQTT whenever it improves.

### (b) What are the potential limitations of your game?
Webcam plus pose lag reality by roughly 70-150 ms [measured: X ms], and one camera gives no depth, so the pose gate is a 2D approximation I make lenient at low levels. The hub IMU is limited to about 66 Hz over BLE [measured: Y Hz], has undocumented units and range and no timestamps, so peak timing is only good to about 15 ms and detection reports a swing 35-80 ms late; shot speed is a calibrated relative measure, not true racket speed [cross-checked against an MPU-6050 only if the bench ruler shipped]. The motors are weak haptics unless I add inertia, their vibration contaminates the IMU so detection is blanked after each pulse, and the felt pulse lands 100-200 ms after the swing peak. Motor cues for spin and timing are dropped at the two hardest levels. Ball physics is arcade, not validated table tennis, spin is classified from my own swing with no ground truth, and Pro and Insane are best effort. Match is receive-only (the computer serves every point). Pose needs front lighting and my upper body in frame; AprilTags need matte print and a white margin. The score goes to a public unauthenticated broker, so it can be dropped or spoofed, and my "record" (session best of live, unassisted hits) is an interpretation. Calibration and tuning come from one player in one room, so another person needs a quick re-tune. The UNO Q node depends on a phone hotspot and is optional by design.

### (c) AI/ML algorithms (two sentences or fewer each)
See section 9. The final text lists only what ships: BlazePose landmarker, AprilTag/ArUco detector, One-Euro filter, logistic-regression spin classifier, signed-axis swing detector (axis from SVD, otherwise non-learning), PD controller, softmax policy, and only if built tabular Q-learning and the FFT whistle detector.

### Template questions outline (Notion: compelling picture, GitHub link, code block, then reflection block)
Page: title and HUD screenshot mid-rally; GitHub link `github.com/rafaeshafi/Me193` folder P5-Ping-Pong; short code excerpt (swing detector or judge); video link; diagram; MQTT table; answers (a)(b)(c); **Reflection questions**:
- **What we learned.** Measured IMU rate, camera-vs-IMU lag in ms, the signed-axis lesson (a magnitude detector fires on the backswing), why timing comes from the IMU and place from pose.
- **What we are most proud of.** Candidates: x-ray policy HUD, per-shot speed and spin, haptic language, live record ticking on the broker, sim-balanced levels.
- **What took a while.** Candidates: haptics without phantom swings, BLE timing and detection lag, pose lag compensation, UNO Q Bridge quirks. Replace with the real ones.
- **Most spectacular failure.** A real event written by the student on Day 6.
README `## Questions` follows the P3 style (`###` per question, Limitations, `## Note on versions`).

---

## 15. Assumptions and open questions

**Assumptions (each has a bench check or a default):**
1. The Connection Card colour and serial are readable (P1 tries the five known cards otherwise).
2. The Brother printer on the Mac prints the cards (printers are configured (V)); a phone screen is the backup.
3. Right-handed play with the hub in the right fist (`H` key or wave test swaps).
4. Webcam at least 720p; the player can stand 1.5-2.0 m away, front-lit, head to hips in frame (P8 and Day 2 legibility test settle it).
5. The iPhone hotspot is available for the UNO Q; Mac-to-board TCP works (SSH precedent), board-to-Mac UDP is unverified until P10a.
6. Jumpers and a breadboard or clips arrive by Day 5; otherwise 6b/6c are cut.
7. The instructor accepts a bare float string and retained messages; "record" = session best streak (U; email Day 0; three-value switch).
8. A hit counts only the player's valid returns.
9. Gyro units probably decideg/s and full-scale unknown until P2/P3; the design does not need absolute units.
10. Motors may be weak; beep, light and Mac sound still deliver the fun-cool requirement.
11. Python 3.12 works with legoeducation 1.1.1 (V in the P1 venv).
12. The instructor demo runs live with real sensors; `--demo`, `--fake`, `--sim`, `--replay` never publish to the official topic.
13. All lag, window, radius and difficulty numbers are estimates until benches, `tools/sim.py` and playtests settle them.
14. Claude Code writes most code; the student runs benches, plays, tunes, records and writes the explanations.
15. Deploying the UNO Q app stops RateNgo; accepted.
16. Submission is targeted for the end of Day 6; Day 7 is buffer. The exact due hour is not assumed to be late on 2026-10-13.
17. Nothing was run against hardware, the broker or the board while planning.
