# P6 Gait Wearable: design

Date: 2026-10-23. Course: ME 193 project 6. Status: implemented; revised the same day to drop the TFT (see Revision 2).

## Goal

A thigh-worn wearable built from the Arduino UNO Q, an MPU6050 IMU and the 3.2" ILI9341 TFT that tells
standing still, walking, running and biking (including air-biking while lying on the back) apart, live,
on the screen. Two classifiers run side by side and are shown together:

- **Model 1** sees one number: the peak acceleration along the thigh axis.
- **Model 2** sees all six IMU channels through about twelve window features.

No neural nets. Both models are decision trees so the comparison is fair and the explanation is honest:
the only difference between them is what they are allowed to see.

Grading needs (from the assignment): works on three peers, a clear explanation of which model is better
and why, a "wow" moment, and a Notion page with an intro paragraph, demo video, a graphical representation
of the model, the GitHub link and the four reflection answers.

## Why the thigh

The thigh is an established spot for this (the activPAL research sensor sits on the front of the thigh
and classifies sitting, standing, stepping and cycling). Expected signal at the mid-thigh:

| Activity | Thigh-axis accel | Gyro | Notes |
|---|---|---|---|
| Stand | flat, ~1 g | ~0 | |
| Walk | 1.3 to 1.8 g heel-strike peaks at ~1 Hz | 100 to 200 deg/s pitch | thigh swings 20 to 30 degrees |
| Run | 2.5 to 4 g peaks at 1.3 to 1.6 Hz | 300+ deg/s | |
| Bike / air-bike | stays near 1 g, smooth | large, very periodic | lying down flips the gravity axis |

So Model 1 separates stand, walk and run by peak height but confuses biking with standing or a slow
shuffle. Model 2 sees gyro energy and gravity direction and separates all four. That contrast is the
story for the explanation point.

## Hardware

- **Strap:** two VELCRO Brand All-Purpose Elastic Straps, 27 in x 1 in (model 90441, bought). Worn on
  the right mid-thigh, front-lateral, screen facing up when the wearer looks down. Snug. A loose strap
  smears every signal.
- **Plate:** one laser-cut 3 mm acrylic plate, 100 x 144 mm (`plate/plate.py`, parametric, exports the DXF and
  checks its own clearances). TFT across the top on 25 mm M3 standoffs (its rear header housings must clear the
  strap underneath); UNO Q below it turned 90 degrees so the USB-C faces the knee, on 10 mm standoffs; MPU6050
  breakout right of the UNO Q near the bottom on 5 mm standoffs, pins up, long side along the thigh so its X axis
  runs down the thigh. Two 27 x 4 mm slits per strap at the left and right edges: each strap enters through the
  left slit, runs across the FRONT of the plate under the boards (every standoff is kept out of the 25.4 mm strap
  band), exits through the right slit and buckles behind the thigh.
- **Rigid IMU is non-negotiable:** screwed, not taped, so peers' data looks like the training data.
- **Wiring:** TFT as whoop-hr: VCC 5V, LED 3V3, GND, CS D10, DC D9, RST D8, SDI D11, SCK D13, SDO
  unused. MPU6050 on the SDA/SCL header at 3V3 and GND, AD0 to GND (address 0x68). Short jumpers
  zip-tied to the plate.
- **Power:** power bank in the shorts pocket, one USB-C cable up the leg to the UNO Q, with a tape loop
  on the strap as strain relief. No switch: plug in and it boots into the app.
- **Orientation convention:** all recordings on the right thigh with the plate in the same place. The
  code does not hard-code this (Model 2 learns gravity direction), but mixing legs would hurt Model 1.

## Firmware (MCU sketch, `app/sketch/`)

Three fast jobs, no thinking. All logic lives in Python on the Linux side, like whoop-hr.

- **IMU sampling.** Own MPU6050 driver in `sketch/src/mpu6050.{h,cpp}` over Wire (no library; the
  board cannot fetch libraries over the cable). Config: accel +-8 g (running peaks must not clip),
  gyro +-1000 deg/s, internal DLPF 44 Hz, 100 Hz into the chip's own FIFO, which loop() drains in one
  burst each pass, so the sampling clock is the sensor's. Sample = 6 x int16, raw, no timestamp.
- **Batching.** Every 10 samples (100 ms) the sketch sends `Bridge.notify("imu", bytes)`: header
  seq, count, FIFO-reset counter, then 120 bytes of samples (123 bytes, under the ~270-byte Bridge limit
  hit in whoop-hr). The reset counter changes after a FIFO overflow or a reprobe so Python can drop its
  window instead of mixing two streams. If the IMU is missing or fails it reprobes every 2 s and sends
  an empty heartbeat batch every 500 ms so Python shows NO IMU instead of freezing. The `tft` and `tfth`
  providers are registered with `provide_safe` so they run between loop() calls, not on another thread.
- **TFT.** Reuse `ili9341.*` and `tft_gfx.*` from whoop-hr unchanged. New `tft_ui.*` with one layout:
  big activity word (Model 2) in colour across the top; Model 1's word small beneath it, tinted green
  when the models agree and amber when they disagree; cadence and peak g on the right; a 240-column
  scrolling accel waveform across the bottom third. Providers: `tft` takes a short key=value text
  payload (<200 chars), `tfth` takes waveform bytes. Only changed fields are redrawn; the waveform
  scrolls one column per new sample.
- **Guided mode text.** The `tft` payload carries a `prompt` field (for example `WALK 18s`) drawn over
  the waveform during recording. No separate code path.
- **Build:** compile locally with App Lab's arduino-cli (fqbn arduino:zephyr:unoq), seconds instead of
  two minutes on the board.

## Python app on the board (`app/python/`)

Pure Python + numpy. No scikit-learn on the board.

- **`imu.py`:** receives batches, converts to g and deg/s, keeps a 5 s ring buffer. Every 0.5 s cuts
  a 2 s window (two strides at walking pace) and computes features.
- **`features.py`:** one function returns a dict; both models read from it.
  - Model 1 feature: 95th percentile of |thigh-axis accel| in the window (one bump cannot fool it).
  - Model 2 features (~12): mean and std of each accel axis (means encode gravity direction), accel
    magnitude std and peak, gyro magnitude RMS and peak, dominant frequency of accel magnitude by FFT,
    and gyro-energy to accel-energy ratio (the biking fingerprint).
- **`models.py`:** decision trees loaded from JSON (`feature`, `threshold`, `left`, `right`, `label`).
  A ~30-line evaluator. Model 1 depth 3 on its single feature, Model 2 depth 5 on all features.
- **Smoothing:** majority vote over the last 3 windows before display. Cadence from the dominant
  frequency: doubled for walk and run (one thigh sees every other step), straight for bike.
- **`display.py`:** builds the `tft` payload and waveform bytes; sends only on change.
- **`web.py`** (port 8080, published in `app.yaml`): one plain HTML + JS page. Live labels from both
  models, live waveform, recorder controls (Start guided session, Record single activity with a label
  picker, Stop, list of saved sessions). WebSocket at 10 Hz. Reached from the phone over the iPhone
  hotspot (open the Personal Hotspot screen for ~30 s after power-up so the board can join).
- **`recorder.py`:** guided script STAND 20 s, WALK 30 s, RUN 30 s, BIKE 30 s with 5 s unlabelled
  "get ready" gaps. Writes raw 100 Hz samples with a label column to `/app/data/<timestamp>.csv`, one
  file per session. Raw samples, not features, so features can change without re-recording.
  Fallback: when powered and not recording, the app keeps an unlabelled rolling log so a session can be
  labelled by time on the Mac if the hotspot misbehaves.

## Training on the Mac (`train/`)

- **`scripts/pull_data.sh`:** adb-pulls `/app/data/*.csv` from the board over the USB cable.
- **`train/train.py`:** Mac venv with scikit-learn, numpy, matplotlib. Windows the CSVs with the same
  `features.py` the board runs (no train/live mismatch). Fits both trees, prints a held-out confusion
  matrix for each, writes `models/model1.json` and `models/model2.json`. `--peer` holds out whole
  sessions by person to estimate generalisation before the graded demo.
- **`train/plots.py`:** figures for Notion: both tree diagrams with thresholds on the branches, the two
  confusion matrices side by side, and a feature-space scatter (peak g alone vs peak g against gyro
  RMS) that shows biking collapsing onto standing for Model 1.
- **`scripts/deploy.sh`:** copies app + JSON models to the board over adb, stops whoop-hr (one app at a
  time), sets this app as the App Lab default so it starts on power alone. Retrain loop: pull, train,
  deploy, ~2 minutes.

## Demo and Notion

- Peer flow: strap in, plug the bank, stand, walk, jog, lie down and air-bike while watching the screen.
  Disagreement line does the talking during biking. Optional: Start guided session first to collect
  the peer's data for the `--peer` check.
- Notion page: intro paragraph, demo video (user records), the three figures, the GitHub subfolder
  link `https://github.com/rafaeshafi/Me193/tree/main/P6-Gait-Wearable`, the four reflection answers
  (drafted here, finalised by the user). Notion is edited through the user's real Chrome.

## Testing

pytest on the Mac: feature extraction on synthetic walk/run/bike/stand signals; the JSON tree evaluator
against scikit-learn's own predictions on the same windows; window cutting and majority vote; recorder
label timing. The sketch's test is a clean local compile. Final acceptance is on the hardware: the
four activities recognised on the user, then on three peers.

## Repo layout

```
~/ME193/P6-Gait-Wearable/
  app/            app.yaml, python/, sketch/ (App Lab app)
  train/          train.py, plots.py, requirements.txt
  models/         model1.json, model2.json (committed)
  scripts/        deploy.sh, pull_data.sh
  plate/          plate.py (CadQuery) -> plate.dxf
  tests/
  docs/           superpowers/specs, superpowers/plans, notion drafts
  data/           recordings (gitignored)
```

## Out of scope

Hip or ankle placement, Bluetooth or a second display device, per-user calibration, anything beyond
the four classes, on-board training.

## Revision 2 (2026-10-23, user decision): no TFT, the web page is the display

The 3.2" TFT is dropped: it took both power-header pins and most of the plate, and the phone page already shows
everything. What changes:

- **Display = the web page** on port 8080, reached from the phone over the hotspot or from a laptop over
  `adb forward`. It shows the big Model 2 word coloured per class, the Model 1 word with agrees/differs, cadence,
  peak g, a 4 s thigh-axis acceleration trace, the recording prompt and controls. It is also what the demo video
  films.
- **Sketch:** IMU batching only (`imu` notifications, heartbeat, reprobe). No TFT driver, no `tft`/`tfth` providers,
  no mutex.
- **Python:** `display.py` (payload and waveform bytes) is removed; `pipeline.Gait` keeps `on_batch`, `tick`,
  `snapshot`; `main.py` only ticks and logs. Everything else (features, models, vote, recorder, web) is unchanged.
- **Hardware:** UNO Q + MPU6050 on the plate; the sensor takes the 3V3 pin (or the Qwiic port). Plate 100 x 109 mm:
  UNO Q turned 90 degrees at the top (USB-C toward the knee), MPU6050 right of it near its bottom, upper strap
  through the UNO Q's hole-free middle, lower strap in the empty band below the UNO Q. Standoffs: UNO Q 10 mm,
  MPU 5 mm.
- Out of scope now: any on-body display. The UNO Q's LED matrix could later show the activity's initial with the
  P5 `draw` provider; not built.

## Revision 3 (2026-10-26, from the first two recorded sessions): rotation-proof features and a motion gate

Two sessions trained against each other showed the original Model 2 picking up accidents of one session: a slight
forward lean while walking (mean_x) separated standing from walking as cleanly as the gyro did, and the bike leg's
tilt flipped sign between sessions, so raw gravity components did not transfer (77 % leave-one-session-out).

- **Model 2 features** are now rotation-proof about the thigh axis: `tilt` (|gravity on the thigh axis|: 1 g upright,
  ~0.3 g pedalling), `std_x`, `std_perp` (spread across the thigh), `mag_std`, `mag_peak`, `gyro_rms`, `gyro_peak`,
  `dom_freq`, `gyro_ratio`. Raw `mean_y/z` and `std_y/z` are still computed and shown but not used.
- **Model 2 is motion-gated:** the exported tree's root is the fixed split `gyro_rms <= 40 deg/s -> STAND` (a still
  leg reads the gyro bias, ~10; every activity reads 65+), and the learned tree (depth <= 4, light cost-complexity
  pruning) only sorts WALK / RUN / BIKE. A still leg is decided by motion, never by posture. Training warns if the
  gate is not clean on the data.
- Result on the two sessions: Model 2 97.8 % leave-one-session-out (STAND 100, WALK 96, RUN 97, BIKE 98); Model 1
  72 % with WALK mostly called BIKE (1.26 g vs 1.20 g peak). Deployed tree: moving? -> impact > 2.8 g RUN ->
  thigh horizontal BIKE -> else WALK.
- Tests carry their own seed trees (`tests/seed_models.py`); the files in `models/` are the trained ones and a test
  only checks them against the feature contract and the gate.

## Revision 4 (2026-10-26): racing on the spot

A race mode on the page: the runner runs on the spot, the gait drives a runner along a track against ghosts.

- **Speed from the gait.** stride_m = clamp(0.6 + 0.5 x (peak_g - 1), 0.5, 2.2); speed = cadence/60 x stride_m.
  The classifier gates it: STAND, BIKE, NO IMU and `---` give 0, WALK is capped at 7 km/h, RUN is uncapped. The
  speed is smoothed with a 1 s exponential average. (From the recordings: walk 4.9 km/h, jog 11.5 km/h, a 200 spm
  sprint with 4 g peaks 25 km/h.)
- **Race.** Distances 100, 400, 1000 m. Start -> 3 s countdown -> running (distance integrates speed every loop
  pass, ~50 ms) -> finished when the line is crossed (finish time interpolated inside the last step). Abort returns to
  idle without saving. Ghosts at fixed paces: Jogger 9, Runner 13, Sprinter 22 km/h, plus "Your best", a replay of the
  runner's best speed profile at that distance when one exists.
- **Results.** `data/races.json` keeps every finished race: who, distance, time, average km/h, when, and the speed
  profile (one sample per classification tick). The page shows a leaderboard per distance (top 10 by time) and, on
  finishing, the time, average speed, which ghosts were beaten and the rank.
- **Page.** A Race panel above the gait view: distance buttons, Start/Abort, a horizontal track with one dot per
  runner, live speed, distance and time, a countdown overlay, the finish card and the leaderboard; the speed formula
  printed under the track. The gait view keeps working during a race. Web routes: `POST /race/start?distance=&who=`,
  `POST /race/abort`; the race state rides in the existing snapshot/SSE stream.
- No sketch change. Code: `app/python/race.py` (SpeedModel, ResultStore, Race) driven from `Gait.tick`.
