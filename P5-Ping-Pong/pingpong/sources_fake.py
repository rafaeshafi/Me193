"""Fakes for hardware-free development, tests and tool --selftest runs.

FakeDoubleMotor feeds the REAL legoeducation notification parser (it builds
genuine IMU/gesture packets), so HubLink is exercised end to end without BLE.
"""

import contextlib
from types import SimpleNamespace

from legoeducation import rpc_message as rm


class FakeDoubleMotor:
    """Stands in for the legoeducation DoubleMotor: records every command, can fail on demand."""

    def __init__(self, fail_connect=False):
        self.calls = []             # (method name, kwargs) in call order
        self.connected = False
        self.fail_connect = fail_connect
        self.fail_on = set()        # method names that should raise
        self.info_device = None
        self._callback = None

    def _record(self, name, **kwargs):
        if name in self.fail_on:
            raise RuntimeError(f"fake failure in {name}")
        self.calls.append((name, kwargs))

    # --- the slice of the DoubleMotor API the project uses ---------------------
    def set_notification_callback(self, callback):
        self._callback = callback

    def connect(self, **kwargs):
        self._record("connect", **kwargs)
        self.connected = not self.fail_connect

    def disconnect(self):
        self._record("disconnect")
        self.connected = False

    def motor_stop(self, *, motor=None, blocking=True):
        self._record("motor_stop", motor=motor, blocking=blocking)

    def motor_run_for_time(self, time_ms, **kwargs):
        self._record("motor_run_for_time", time_ms=time_ms, **kwargs)

    def beep(self, **kwargs):
        self._record("beep", **kwargs)

    def light_color(self, color, **kwargs):
        self._record("light_color", color=color, **kwargs)

    def begin_batch(self):
        self._record("begin_batch")

    def end_batch(self, blocking=True):
        self._record("end_batch", blocking=blocking)

    def cancel_batch(self):
        self._record("cancel_batch")

    @contextlib.contextmanager
    def batch(self, blocking=True):
        self.begin_batch()
        try:
            yield
        finally:
            self.end_batch(blocking=blocking)

    # --- test helpers -----------------------------------------------------------
    def set_battery(self, pct):
        self.info_device = SimpleNamespace(batteryLevel=pct)

    def emit(self, imu=None, gesture=None):
        """Deliver one notification packet to the registered callback.

        imu = (ax, ay, az, gx, gy, gz) raw counts; gesture = le.MOTION_GESTURE_*.
        """
        data = b""
        if imu is not None:
            ax, ay, az, gx, gy, gz = imu
            data += bytes(rm.ImuDeviceNotification(
                orientation=0, yawFace=0, yaw=0, pitch=0, roll=0,
                accelerometerX=ax, accelerometerY=ay, accelerometerZ=az,
                gyroscopeX=gx, gyroscopeY=gy, gyroscopeZ=gz).Serialize())
        if gesture is not None:
            data += bytes(rm.ImuGestureNotification(gesture).Serialize())
        packet = rm.DeviceNotification(deviceDataByteLength=len(data), deviceData=data)
        self._callback(packet)


class FakeCamera:
    """cv2.VideoCapture-shaped: kind is "ok" | "black" | "closed"."""

    def __init__(self, kind="ok"):
        self.kind = kind

    def isOpened(self):
        return self.kind != "closed"

    def read(self):
        import numpy as np

        if self.kind == "closed":
            return False, None
        level = 0 if self.kind == "black" else 120
        return True, np.full((360, 640, 3), level, dtype=np.uint8)

    def release(self):
        pass


class FakeEnv:
    """Everything env_check/bench tools need from the outside world, simulated.

    sleep() advances a FakeClock and, while the fake hub is connected, delivers
    IMU notifications at `hz` -- so rate statistics come out exactly.
    """

    def __init__(self, hz=66.0, hub_found=True, camera="ok", mqtt_ok=True, mqtt_rtt_ms=95.0):
        from pingpong.clock import FakeClock

        self.clock = FakeClock(start_ns=1_000_000_000)
        self.hz = hz
        self.camera = camera
        self.mqtt_ok = mqtt_ok
        self.mqtt_rtt_ms = mqtt_rtt_ms
        self.hub_device = FakeDoubleMotor(fail_connect=not hub_found)
        self._next_emit_ns = None   # absolute schedule, so the rate is exact across sleep() calls
        self.scenario = None        # optional callable(now_ns) -> (ax, ay, az, gx, gy, gz) raw counts

    def make_hub(self, notify_ms, card):
        from pingpong.hub import HubLink

        return HubLink(self.hub_device, notify_ms=notify_ms, clock=self.clock, card=card)

    def sleep(self, seconds):
        target_ns = self.clock.now_ns() + round(seconds * 1e9)
        if self.hub_device.connected and self.hz > 0:
            period_ns = round(1e9 / self.hz)
            if self._next_emit_ns is None:
                self._next_emit_ns = self.clock.now_ns() + period_ns
            while self._next_emit_ns <= target_ns:
                self.clock.advance_s((self._next_emit_ns - self.clock.now_ns()) / 1e9)
                imu = self.scenario(self.clock.now_ns()) if self.scenario else (0, 0, 1000, 0, 0, 0)
                self.hub_device.emit(imu=imu)
                self._next_emit_ns += period_ns
        self.clock.advance_s((target_ns - self.clock.now_ns()) / 1e9)

    def open_camera(self, index):
        return FakeCamera(self.camera)

    def mqtt_roundtrip(self, topic, timeout_s=5.0):
        return self.mqtt_rtt_ms if self.mqtt_ok else None


class FakeMqttClient:
    """paho-shaped: records publishes and connection configuration."""

    def __init__(self):
        self.published = []         # {"topic", "payload", "qos", "retain"}
        self.config_calls = []      # will_set / reconnect_delay_set / connect_async / loop_start
        self.on_connect = None
        self.connected = False

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append({"topic": topic, "payload": payload, "qos": qos, "retain": retain})

    def will_set(self, topic, payload=None, qos=0, retain=False):
        self.config_calls.append({"call": "will_set", "will_topic": topic, "payload": payload,
                                  "qos": qos, "retain": retain})

    def reconnect_delay_set(self, min_delay=1, max_delay=120):
        self.config_calls.append({"call": "reconnect_delay_set", "min": min_delay, "max": max_delay})

    def connect_async(self, host, port=1883, keepalive=60):
        self.config_calls.append({"call": "connect_async", "host": host, "port": port, "keepalive": keepalive})

    def loop_start(self):
        self.config_calls.append({"call": "loop_start"})

    def simulate_connect(self):
        """What paho does when the broker accepts us: call on_connect(client, userdata, flags, rc, props)."""
        self.connected = True
        if self.on_connect:
            self.on_connect(self, None, {}, 0, None)
