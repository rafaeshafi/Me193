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
| `models/` | The two trained trees as JSON |
| `train/` | Training (`train.py`) and the figures (`plots.py`) on the Mac |
| `plate/` | Laser-cut mounting plate (CadQuery → DXF) |
| `scripts/` | `deploy.sh`, `pull_data.sh`, `compile.sh` (all over USB) |
| `tests/` | 57 pytest tests |

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

Design notes and the build log are in `docs/superpowers/`.
