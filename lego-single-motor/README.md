# lego-single-motor

A quick Python script that spins a **single LEGO Education motor** (from the
Computer Science & AI kit) over Bluetooth Low Energy.

Built for ME 193 – AI & Robotics, "Day 2 / Python Overview".

## What it does

`single_motor.py`:

1. Connects to the first Single Motor it finds over BLE.
2. Turns it one full rotation (360°) at 50% speed.
3. Free-runs it at 30% for ~2 seconds, printing live position and speed.
4. Stops and disconnects.

## Setup

Requires Python 3.11+ and a LEGO Education Single Motor that is powered on and in
Bluetooth range.

### Mac / Linux

```bash
# 1. Create a virtual environment named "my_env"
python3 -m venv my_env

# 2. Activate it
source my_env/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt   # or: pip install legoeducation
```

### Windows

```bat
python -m venv my_env
my_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python single_motor.py
```

On macOS the first run triggers a Bluetooth permission prompt for the terminal —
allow it.

## How the code works

- `import legoeducation as le` — the high-level LEGO Education API. Importing
  **this** library (rather than `spike` / `hub`) is what keeps the code aimed at
  the CS & AI kit instead of SPIKE Prime.
- `motor = le.SingleMotor()` — an object representing one motor.
- `motor.connect()` — scans over BLE and connects to the first Single Motor found.
  Pass `card_color=` / `card_serial=` to target a specific motor by its
  Connection Card. `motor.connected` is then `True`/`False`.
- `motor.motor_run_for_degrees(360, speed=50)` — rotate a fixed amount; this call
  blocks until the move finishes.
- `motor.motor_run(speed=30)` — start spinning continuously; returns immediately.
  Keeps going until `motor.motor_stop()` or the next motor command.
- `motor.motor` holds live telemetry: `.position`, `.speed`, `.power`,
  `.absolutePosition`, `.motorState`.
- `motor.motor_reset_relative_position()` — zero the relative position counter.
- `motor.disconnect()` — drop the BLE connection.

Speeds are a percentage from -100 to 100 (negative = reverse). You can also use
`le.MOTOR_MOVE_DIRECTION_CLOCKWISE` / `..._COUNTERCLOCKWISE` via the `direction=`
argument.

## Why use a virtual environment?

A virtualenv (`my_env/`) is a private, per-project copy of Python and its
packages. It means:

- `pip install legoeducation` and its BLE dependencies land in *this project*,
  not system-wide, so projects can't break each other with conflicting versions.
- The exact set of packages is reproducible — anyone can recreate it from
  `requirements.txt`.
- Your system Python stays clean, and nothing here needs `sudo`.

`my_env/` is listed in `.gitignore` and must **not** be committed — it is large,
machine-specific, and rebuilt from `requirements.txt` in seconds.
