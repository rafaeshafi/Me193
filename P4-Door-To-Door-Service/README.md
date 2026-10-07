# P4 — Door-to-Door Service

A green LEGO minifig is tracked by a camera on the laptop; its position is published
over MQTT and consumed by two Arduino Apps on the UNO Q that together turn the board
into a car that follows the minifig around like a door-to-door delivery driver.

## Pipeline

```
laptop camera → live_tracker.py (laptop) → MQTT (test.mosquitto.org, topic "minifig/centroid")
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        ▼                                                       ▼
              minifig-dot-display/                                   minifig-motor-follower/
          (shows minifig position as a dot                      (drives two DC motors via
             on the LED matrix)                                  Cytron Maker Drive to follow it)
```

Both apps subscribe to the same MQTT topic and expect the same message shape:

```json
{"found": true, "x_norm": 0.12, "y_norm": -0.4}
```

`x_norm`/`y_norm` are normalized to -1..1 (0 = centered in frame).

## Apps in this folder

| Folder | Pulled from board app | What it does |
| --- | --- | --- |
| `minifig-dot-display/` | Minifig Dot Display 🟩 | Plots the minifig's position as a single dot on the 8x13 LED matrix. |
| `minifig-motor-follower/` | Minifig Motor Follower 🚗 | Drives the car toward the minifig (proportional control with deadband/hysteresis), stops it if the feed goes stale, and can optionally also show the dot on the matrix. |

Each folder is a self-contained Arduino App (`app.yaml`, `python/`, `sketch/`) and can be
opened/checked into Arduino App Lab independently, or run side by side (only one app
runs on the board at a time, so pick whichever you want active).

## Laptop tracker: YOLOv8 minifig detector

The `laptop-minifig-tracker/` folder contains a fine-tuned YOLOv8 model that detects
the green LEGO minifig in the webcam feed and publishes its centroid over MQTT.

**Files:**
- `live_tracker.py` — Main detector; runs live inference and publishes MQTT messages
- `train.py` — Fine-tuning script (trains on your dataset, saves weights to `runs/detect/`)
- `dataset/` — Roboflow YOLOv8 export (~50 labeled minifig images)
- `requirements.txt` — Python dependencies

## Setup & Running

### Quick start (use pre-trained weights):

```bash
cd laptop-minifig-tracker
pip install -r requirements.txt
python live_tracker.py
```

The script will open a preview window showing the bounding box, centroid marker, and confidence.
Press `q` to quit. By default, it publishes to `test.mosquitto.org` (change with `--mqtt-host`).

### Re-train on your own data:

```bash
cd laptop-minifig-tracker

# Export your labeled images from Roboflow as YOLOv8 format and unzip into "dataset/"
# (must contain data.yaml at the root)

python train.py
# Trains for 100 epochs on Apple Silicon (or CPU fallback)
# Saves weights to runs/detect/green_minifig-2/weights/best.pt
```

### Connect to the board:

1. Start the tracker on your laptop:
   ```bash
   python live_tracker.py --mqtt-host <UNO_Q_IP>
   ```
2. Start `minifig-motor-follower` from Arduino App Lab on the UNO Q
3. Move the minifig around—the robot should follow!

## Wiring (motor follower)

Cytron Maker Drive, sign-magnitude (PWM_PWM) control — see
`minifig-motor-follower/README.md` for full pin mapping and notes.
