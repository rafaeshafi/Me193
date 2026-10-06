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

## Missing piece: the laptop tracker

Both apps' Python code and READMEs reference a laptop-side script,
`laptop-minifig-tracker/live_tracker.py`, which does the actual camera-based color
tracking of the green minifig and publishes the `minifig/centroid` MQTT messages.
That script was **not found** on the board or in the `rafaeshafi/Me193` GitHub repo —
it needs to be copied in here separately (e.g. as `laptop-minifig-tracker/live_tracker.py`
alongside these two app folders) before this pipeline can run end to end.

## Running

1. On the laptop: `cd laptop-minifig-tracker && python live_tracker.py`
2. On the board: start `minifig-motor-follower` (and/or `minifig-dot-display`) from
   Arduino App Lab.

## Wiring (motor follower)

Cytron Maker Drive, sign-magnitude (PWM_PWM) control — see
`minifig-motor-follower/README.md` for full pin mapping and notes.
