# P6 Gait Wearable: design

Date: 2026-10-23. Course: ME 193 project 6. Status: approved in brainstorming, awaiting implementation plan.

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
- **Plate:** one laser-cut 3 mm acrylic plate, about 110 x 75 mm, carrying everything. UNO Q on its four
  mounting holes with M3 standoffs; TFT on its four 3.0 mm corner holes (pattern 87.4 x 54.4 mm, from
  the whoop-hr measurements); MPU6050 breakout screwed flat to the plate next to the board with its X
  axis pointing down the thigh. Two 27 x 4 mm slots, one near each end, for the straps. The plate is a
  CadQuery script in `plate/` that exports a DXF.
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
  gyro +-1000 deg/s, internal DLPF 44 Hz, 100 Hz sampling from a Zephyr timer. Sample = 6 x int16 +
  ms timestamp.
- **Batching.** Every 100 ms the sketch calls Bridge method `imu` with 10 samples packed as bytes
  (about 160 bytes, under the ~270-byte Bridge limit hit in whoop-hr). If the IMU is missing or fails
  it reprobes every 2 s and sends empty batches so Python shows NO IMU instead of freezing.
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
