# P6: Gait Wearable (ME 193)

A thigh-worn wearable built from an Arduino UNO Q, an MPU6050 IMU and a 3.2" ILI9341 TFT. It tells standing, walking,
running and biking (including air-biking on your back) apart, live on the screen, with two models side by side:

- **Model 1** sees one number: the peak acceleration along the thigh axis (95th percentile of |ax| in a 2 s window).
- **Model 2** sees all six IMU channels through twelve window features (gravity direction, spreads, gyro energy,
  dominant frequency, rotation-per-impact).

Both are depth-limited decision trees trained with scikit-learn and exported as JSON; the board evaluates them in
pure Python. No neural nets.

## Layout

- `app/` the Arduino App Lab app: `sketch/` (MPU6050 FIFO at 100 Hz, TFT) and `python/` (features, models, recorder,
  phone page on port 8080).
- `models/` the two exported trees. `train/` the scikit-learn training and the figures. `plate/` the laser-cut plate.
- `scripts/` deploy over USB (`deploy.sh`), pull recordings (`pull_data.sh`), local compile (`compile.sh`).
- `docs/superpowers/specs/` the design; `docs/figures/` the Notion figures.

## Workflow

1. `scripts/deploy.sh` with the board on the cable.
2. Open the Personal Hotspot screen on the phone for ~30 s so the board joins, then browse to the board's address on
   port 8080. Tap **Start guided session**: the screen walks you through STAND, WALK, RUN, BIKE.
3. `scripts/pull_data.sh`, then `.venv/bin/python train/train.py` and `.venv/bin/python train/plots.py`.
4. `scripts/deploy.sh` again to ship the trained models.

Tests: `.venv/bin/python -m pytest tests -q`.
