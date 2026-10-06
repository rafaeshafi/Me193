import json
import time

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge, Frame

# Must match the topic published by live_tracker.py on the laptop.
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "minifig/centroid"

DEADBAND = 0.03       # must match the laptop script's DEADBAND
HOLD_BAND = 0.05      # once stopped, must drift back out past this before driving resumes
KP = 45             # proportional gain: PWM duty added per unit of normalized offset beyond DEADBAND
TURN_GAIN = 0         # no per-wheel bias: both motors always get the same duty cycle
MIN_SPEED = 50       # PWM floor: below this a motor just stalls and hums
MAX_SPEED = 60
STALE_TIMEOUT_S = 1.0  # stop the car if no update arrives for this long

_driving = False

ROWS, COLS = 8, 13
BRIGHTNESS = 7
# Must match USE_LED_MATRIX in sketch.ino: the sketch only registers the "draw"
# handler when the matrix is enabled.
SHOW_MATRIX = False

_last_message_time = 0.0


def set_motors(left_direction, left_speed, right_direction, right_speed):
    # Single call carrying both motors' values as one list (the sketch
    # receives it as a vector<int>), rather than two separate scalar-argument
    # calls -- the second of those never reliably reached the MCU.
    Bridge.call("set_motors", [left_direction, left_speed, right_direction, right_speed])


def drive_toward(x_norm):
    global _driving
    # Hysteresis: use the wider HOLD_BAND to resume driving once stopped, not
    # DEADBAND again. Without this, tracker jitter right at DEADBAND flips
    # direction and snaps to full speed on every crossing -- the fast
    # oscillation near center. The car now only restarts once the minifig has
    # actually drifted back out past HOLD_BAND.
    band = DEADBAND if _driving else HOLD_BAND
    if abs(x_norm) <= band:
        _driving = False
        set_motors(0, 0, 0, 0)
        return
    _driving = True

    # Minifig right of center -> forward, left of center -> reverse.
    direction = 1 if x_norm > 0 else -1
    # Proportional term: speed scales linearly with how far off-center the
    # minifig is, so the car slows down as it nears center instead of
    # braking at full speed.
    error = abs(x_norm) - DEADBAND
    base_speed = MIN_SPEED + KP * error
    bias = x_norm * TURN_GAIN
    # Clamp to MIN_SPEED, not 0: a wheel given a tiny duty cycle stalls and
    # hums instead of turning, which looks like only one motor working.
    left_speed = int(max(MIN_SPEED, min(MAX_SPEED, base_speed - bias)))
    right_speed = int(max(MIN_SPEED, min(MAX_SPEED, base_speed + bias)))
    set_motors(direction, left_speed, direction, right_speed)


def draw_dot(x_norm, y_norm):
    """x_norm, y_norm are -1..1 (0 = centered); scale onto the 8x13 grid."""
    if not SHOW_MATRIX:
        return
    nx = min(max((x_norm + 1) / 2, 0.0), 1.0)
    ny = min(max((y_norm + 1) / 2, 0.0), 1.0)
    col = min(int(nx * COLS), COLS - 1)
    row = min(int(ny * ROWS), ROWS - 1)
    frame_array = np.zeros((ROWS, COLS), dtype=np.uint8)
    frame_array[row, col] = BRIGHTNESS
    Bridge.call("draw", Frame(frame_array).to_board_bytes())


def clear_matrix():
    if not SHOW_MATRIX:
        return
    Bridge.call("draw", Frame(np.zeros((ROWS, COLS), dtype=np.uint8)).to_board_bytes())


def on_message(client, userdata, msg):
    global _last_message_time, _driving
    _last_message_time = time.monotonic()

    try:
        data = json.loads(msg.payload.decode())
    except ValueError:
        return

    if not data.get("found"):
        _driving = False
        set_motors(0, 0, 0, 0)
        clear_matrix()
        return

    x_norm = data.get("x_norm", 0.0)
    drive_toward(x_norm)
    draw_dot(x_norm, data.get("y_norm", 0.0))


def on_connect(client, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}:{MQTT_PORT}, subscribing to {MQTT_TOPIC}")
    client.subscribe(MQTT_TOPIC)


def watchdog_loop():
    """Stop the car if the laptop stops sending updates (lost camera, script
    quit, network drop, ...)."""
    global _driving
    if _last_message_time and time.monotonic() - _last_message_time > STALE_TIMEOUT_S:
        _driving = False
        set_motors(0, 0, 0, 0)
        clear_matrix()
    time.sleep(0.2)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

App.run(user_loop=watchdog_loop)
