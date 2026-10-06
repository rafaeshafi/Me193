import json

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge, Frame

# Must match the topic published by live_tracker.py (or part1_color_detector.py)
# on the laptop.
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "minifig/centroid"

ROWS, COLS = 8, 13
BRIGHTNESS = 7


def draw_dot(x_norm, y_norm):
    """x_norm, y_norm are -1..1 (0 = centered); scale onto the 8x13 grid."""
    nx = min(max((x_norm + 1) / 2, 0.0), 1.0)
    ny = min(max((y_norm + 1) / 2, 0.0), 1.0)
    col = min(int(nx * COLS), COLS - 1)
    row = min(int(ny * ROWS), ROWS - 1)
    frame_array = np.zeros((ROWS, COLS), dtype=np.uint8)
    frame_array[row, col] = BRIGHTNESS
    Bridge.call("draw", Frame(frame_array).to_board_bytes())


def clear_matrix():
    Bridge.call("draw", Frame(np.zeros((ROWS, COLS), dtype=np.uint8)).to_board_bytes())


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
    except ValueError:
        return

    if data.get("found"):
        draw_dot(data.get("x_norm", 0.0), data.get("y_norm", 0.0))
    else:
        clear_matrix()


def on_connect(client, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}:{MQTT_PORT}, subscribing to {MQTT_TOPIC}")
    client.subscribe(MQTT_TOPIC)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

App.run()
