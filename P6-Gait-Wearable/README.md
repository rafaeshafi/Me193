# P6: Gait Wearable

A thigh-worn Arduino UNO Q with an MPU6050 motion sensor that tells **standing, walking, running and biking** apart in
real time, and a race game driven by your running on the spot. ME 193, Tufts, Oct 2026.

Two models run side by side on every 2-second window, both plain decision trees (no neural nets):

| | Sees | Result on unseen sessions |
|---|---|---|
| **Model 1** | one number: peak acceleration along the thigh | 72 % (walking and biking look the same) |
| **Model 2** | all six IMU channels (tilt, spread, rotation, swing rate) | 98 % |

Model 2's first question is fixed: *is the leg moving at all?* If not, STAND. The learned tree only sorts the rest.

## What's where

| Path | What |
|---|---|
| `app/sketch/` | MCU sketch: reads the MPU6050 FIFO at 100 Hz, streams batches to Linux |
| `app/python/` | Board app: features, tree evaluator, recorder, race, web page (port 8080) |
| `models/` | The two trained trees and the step-counter net (`stepnet.json`) as JSON |
| `train/` | Training on the Mac: the trees (`train.py`, `plots.py`) and the step counter (`steps.py`, PyTorch) |
| `plate/` | Laser-cut mounting plate (CadQuery → DXF) |
| `scripts/` | `deploy.sh`, `pull_data.sh`, `compile.sh` (all over USB) |
| `docs/animation/` | Manim explainer of the step counter (`render.sh`) |
| `tests/` | 88 pytest tests |

## Run it

```bash
scripts/deploy.sh                       # board on the USB cable; app starts on power from then on
# phone on the same hotspot -> http://<board>:8080      gait view + guided recording
#                              http://<board>:8080/race  race mode
scripts/pull_data.sh && .venv/bin/python train/train.py && .venv/bin/python train/plots.py
scripts/deploy.sh                       # ship the retrained trees
```

Setup once: `python3 -m venv .venv && .venv/bin/pip install -r train/requirements.txt`.
Tests: `.venv/bin/python -m pytest tests -q`.

## Race mode

Speed = cadence × stride, with stride estimated from impact. STAND and BIKE don't move you, WALK is capped at 7 km/h.
Race 100 m, 400 m or 1 km against three fixed runners and the **Champion**: the fastest finish by anyone at that
distance, replayed as a ghost. Beat it and your run becomes the ghost.

## Step counter (StepNet)

A small causal 1-D conv net, trained in PyTorch on all six IMU channels, outputs a step rate; the count is the area
under it. The board runs the exported weights (`models/stepnet.json`) in plain Python and the **Steps** tab shows the
count with its measured uncertainty (`38 ± 18`).

| | |
|---|---|
| Input | ax ay az gx gy gz, 100 → 25 Hz, standardised |
| Net | conv 6→12 (dil 1) → 12→12 (dil 2) → 12→12 (dil 4), ReLU, kernel 5; softplus head → steps/s; 1,849 parameters |
| Training | 9 counted takes (typed count in the file name) + 4 known-zero stretches from the guided sessions; smooth-L1 on the integrated count; rotation / gain / time-stretch / pairing augmentation |
| Leave-one-recording-out | MAE 3.0 steps (walks 4.5, stands 0.1), MAPE 22 %, 95th pct 47 % (the live ±) |
| Three fresh demo walks | 24 → 26.1, 19 → 20.9, 20 → 21.5 (never trained on; what the video shows) |

The ± band is set by a 7-step take (3.7 steps off is 53 %) and by fold noise: with six walks each fold trains on
five. One take's typed 30 disagrees with the 40 swing cycles on the thigh gyro. The band shrinks as counted walks
accumulate. Full table and figures in
`docs/steps_results.md`.

Record counted takes on the Steps tab (count out loud, type the number on Stop), then:

    scripts/pull_data.sh
    .venv/bin/python train/steps.py --report      # LORO, seeds, figures, docs/steps_results.md, models/stepnet.json
    scripts/deploy.sh

How it works, animated: `docs/animation/stepnet_explainer_small.mp4` (`docs/animation/render.sh` rebuilds it).

Design notes and the build log are in `docs/superpowers/`.
