"""The real outside world for env_check and the bench tools (BLE, camera, broker).

Imported lazily by tools so unit tests and --selftest never touch hardware.
Each method matches pingpong.sources_fake.FakeEnv.
"""

import threading
import time
import uuid

import config
from pingpong.clock import Clock


RETRY_MIN_S, RETRY_MAX_S = 0.5, 2.0


def _connect_patiently(client):
    """Connect in the background and keep trying: the public test broker closes a connection attempt without an
    answer now and then (three of the first four in one probe), and one dropped attempt must not read as 'blocked'."""
    client.reconnect_delay_set(RETRY_MIN_S, RETRY_MAX_S)
    client.connect_async(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE_S)
    client.loop_start()


class RealEnv:
    threaded = True            # camera, IMU parser and actuator each get a thread

    def __init__(self):
        self.clock = Clock()
        self.on_idle = None        # a tool's window: called while sleep() waits, so macOS keeps drawing it

    def sleep(self, seconds):
        if self.on_idle is None:
            time.sleep(seconds)
            return
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            self.on_idle()
            time.sleep(min(0.02, left))

    def make_hub(self, notify_ms, card):
        import legoeducation as le

        from pingpong.hub import HubLink

        return HubLink(le.DoubleMotor(), notify_ms=notify_ms, clock=self.clock, card=card)

    def open_camera(self, index):
        import cv2

        return cv2.VideoCapture(index)

    def make_landmarker(self, model="lite"):
        from pingpong.pose_features import make_landmarker

        return make_landmarker(model=model)

    def make_tag_detector(self):
        from pingpong.tags import TagDetector

        return TagDetector()

    def make_mqtt_client(self):
        from pingpong import mqtt_link

        return mqtt_link.make_paho_client()

    def make_audio(self):
        from pingpong.audio import Audio

        return Audio()                           # the built-in speakers through sounddevice

    def mqtt_roundtrip(self, topic, timeout_s=10.0):
        """Publish a unique token to `topic` and wait for the broker to echo it; -> RTT ms or None."""
        import paho.mqtt.client as mqtt

        token = uuid.uuid4().hex
        got = threading.Event()
        sent = {}

        def on_connect(client, userdata, flags, reason_code, properties=None):
            client.subscribe(topic, qos=1)

        def on_subscribe(client, userdata, mid, reason_codes, properties=None):
            sent["t0"] = time.monotonic()
            client.publish(topic, token, qos=1, retain=False)

        def on_message(client, userdata, msg):
            if msg.payload.decode(errors="replace") == token:
                sent["rtt_ms"] = (time.monotonic() - sent["t0"]) * 1000.0
                got.set()

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pp-check-" + uuid.uuid4().hex[:8])
        client.on_connect, client.on_subscribe, client.on_message = on_connect, on_subscribe, on_message
        _connect_patiently(client)
        try:
            return sent.get("rtt_ms") if got.wait(timeout_s) else None
        finally:
            client.loop_stop()
            client.disconnect()

    def official_roundtrip(self, timeout_s=10.0):
        """Publish a retained "0.0" to the OFFICIAL score topic, see it come back, then clear it.

        Settles whether the broker accepts a retained QoS 1 publish on the real topic.  Only ever called
        after the player said yes (bench_cam --official-check): it briefly changes the retained value.
        """
        import paho.mqtt.client as mqtt

        topic, got, sent = config.SCORE_TOPIC, threading.Event(), {}

        def on_connect(client, userdata, flags, reason_code, properties=None):
            client.subscribe(topic, qos=1)

        def on_subscribe(client, userdata, mid, reason_codes, properties=None):
            sent["t0"] = time.monotonic()
            client.publish(topic, "0.0", qos=1, retain=True)

        def on_message(client, userdata, msg):
            if msg.payload == b"0.0" and "rtt_ms" not in sent:
                sent["rtt_ms"] = (time.monotonic() - sent["t0"]) * 1000.0
                got.set()

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pp-official-" + uuid.uuid4().hex[:8])
        client.on_connect, client.on_subscribe, client.on_message = on_connect, on_subscribe, on_message
        _connect_patiently(client)
        try:
            if not got.wait(timeout_s):
                return None
            info = client.publish(topic, b"", qos=1, retain=True)          # an empty retained payload clears it
            info.wait_for_publish(timeout=timeout_s)
            return sent["rtt_ms"]
        finally:
            client.loop_stop()
            client.disconnect()
