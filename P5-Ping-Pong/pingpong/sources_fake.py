"""Fakes for hardware-free development, tests and tool --selftest runs.

FakeDoubleMotor feeds the REAL legoeducation notification parser (it builds
genuine IMU/gesture packets), so HubLink is exercised end to end without BLE.
"""

import contextlib
import inspect
from types import SimpleNamespace

from legoeducation import rpc_message as rm


def _conforms(name, *args, **kwargs):
    """TypeError unless the REAL DoubleMotor.<name> would accept this call (a fake that accepts anything hides typos)."""
    import legoeducation as le

    inspect.signature(getattr(le.DoubleMotor, name)).bind(None, *args, **kwargs)


class FakeDoubleMotor:
    """Stands in for the legoeducation DoubleMotor: records every command, can fail on demand.

    Every call is checked against the real library's signature, so a misnamed argument fails in
    the tests instead of on the hardware.
    """

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
        _conforms("connect", **kwargs)
        self._record("connect", **kwargs)
        self.connected = not self.fail_connect

    def disconnect(self):
        self._record("disconnect")
        self.connected = False

    def motor_stop(self, *, motor=None, blocking=True):
        _conforms("motor_stop", motor=motor, blocking=blocking)
        self._record("motor_stop", motor=motor, blocking=blocking)

    def motor_run_for_time(self, time_ms, **kwargs):
        _conforms("motor_run_for_time", time_ms, **kwargs)
        self._record("motor_run_for_time", time_ms=time_ms, **kwargs)

    def beep(self, *args, **kwargs):
        _conforms("beep", *args, **kwargs)
        self._record("beep", **kwargs)

    def light_color(self, color, **kwargs):
        _conforms("light_color", color, **kwargs)
        self._record("light_color", color=color, **kwargs)

    def begin_batch(self):
        self._record("begin_batch")

    def end_batch(self, *args, **kwargs):
        _conforms("end_batch", *args, **kwargs)
        self._record("end_batch", blocking=kwargs.get("blocking", True))

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


def body_landmarks(u, v, *, hand="right", vis=0.95, width=640, height=360, shoulder_x=(0.6, 0.4),
                   shoulder_y=0.4):
    """33 pose-landmark stand-ins for which pose.paddle_uv() returns exactly (u, v)."""
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    lm[11] = SimpleNamespace(x=shoulder_x[0], y=shoulder_y, visibility=vis)
    lm[12] = SimpleNamespace(x=shoulder_x[1], y=shoulder_y, visibility=vis)
    sw_px = abs(shoulder_x[0] - shoulder_x[1]) * width
    wrist = 16 if hand == "right" else 15
    lm[wrist] = SimpleNamespace(x=(shoulder_x[0] + shoulder_x[1]) / 2 - u * sw_px / width,
                                y=shoulder_y - v * sw_px / height, visibility=vis)
    return lm


class FakeLandmarker:
    """MediaPipe-shaped: the body is placed by hand_fn(t_ns) -> (u, v), or None for nobody in frame.

    lag_s: the frame shows the scene as it was lag_s ago, like a real camera pipeline.
    """

    def __init__(self, hand_fn, clock, *, lag_s=0.0, hand="right"):
        self.hand_fn, self.clock, self.hand = hand_fn, clock, hand
        self.lag_ns = round(lag_s * 1e9)

    def detect_for_video(self, image, ts_ms):
        uv = self.hand_fn(self.clock.now_ns() - self.lag_ns)
        if uv is None:
            return SimpleNamespace(pose_landmarks=[])
        return SimpleNamespace(pose_landmarks=[body_landmarks(uv[0], uv[1], hand=self.hand)])


class FakeTagDetector:
    """TagDetector-shaped: the cards in view at the clock's current time come from ids_fn(t_ns)."""

    def __init__(self, ids_fn, clock):
        self.ids_fn, self.clock = ids_fn, clock

    def detect(self, frame):
        from pingpong.tags import Tag

        return [Tag(i, ((0, 0),) * 4, (0.0, 0.0), 0.0) for i in sorted(self.ids_fn(self.clock.now_ns()))]


class FakeStream:
    def __init__(self, callback):
        self.callback, self.started, self.closed = callback, False, 0

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed += 1


class FakeSound:
    """sounddevice-shaped: OutputStream(...) gives a stream whose callback a test can pull from."""

    def __init__(self, fail=False):
        self.fail, self.stream, self.options = fail, None, {}

    def OutputStream(self, **kwargs):
        if self.fail:
            raise OSError("no output device")
        self.options = kwargs
        self.stream = FakeStream(kwargs["callback"])
        return self.stream


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
        self.cameras = None         # optional {index: kind}: several cameras, e.g. an iPhone via Continuity Camera
        self.mqtt_ok = mqtt_ok
        self.mqtt_rtt_ms = mqtt_rtt_ms
        self.hub_device = FakeDoubleMotor(fail_connect=not hub_found)
        self._next_emit_ns = None   # absolute schedule, so the rate is exact across sleep() calls
        self.scenario = None        # optional callable(now_ns) -> (ax, ay, az, gx, gy, gz) raw counts
        self.threaded = False       # play.py's live mode is threaded on real hardware, synchronous here
        self.mqtt_client = None
        self.official_calls = 0     # how often something touched the OFFICIAL score topic

    @staticmethod
    def to_image(frame):
        return frame

    def make_landmarker(self):
        return FakeLandmarker(lambda t_ns: (0.0, -0.4), self.clock)

    def make_tag_detector(self):
        return FakeTagDetector(lambda t_ns: set(), self.clock)

    def make_audio(self):
        from pingpong.audio import Audio

        return Audio(backend=FakeSound())

    def make_mqtt_client(self):
        self.mqtt_client = FakeMqttClient()
        return self.mqtt_client

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
        if self.cameras is not None:                       # {index: "ok" | "black" | "closed"}; other indices do not exist
            return FakeCamera(self.cameras.get(index, "closed"))
        return FakeCamera(self.camera)

    def mqtt_roundtrip(self, topic, timeout_s=5.0):
        return self.mqtt_rtt_ms if self.mqtt_ok else None

    def official_roundtrip(self, timeout_s=5.0):
        self.official_calls += 1
        return self.mqtt_rtt_ms if self.mqtt_ok else None


class _Published:
    """What paho's publish() returns: something you can wait on."""

    def wait_for_publish(self, timeout=None):
        return None


class FakeMqttClient:
    """paho-shaped: records publishes and connection configuration."""

    def __init__(self):
        self.published = []         # {"topic", "payload", "qos", "retain"}
        self.config_calls = []      # will_set / reconnect_delay_set / connect_async / loop_start
        self.log = []               # chronological ("publish", topic, payload) / ("disconnect",) / ("loop_stop",)
        self.subscriptions = []     # (topic, qos)
        self.on_connect = None
        self.on_message = None
        self.connected = False

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append({"topic": topic, "payload": payload, "qos": qos, "retain": retain})
        self.log.append(("publish", topic, payload))
        return _Published()

    def is_connected(self):
        return self.connected

    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))

    def deliver(self, topic, payload, retain=False, qos=1):
        """What the broker does when a message arrives for one of our subscriptions."""
        data = payload if isinstance(payload, bytes) else str(payload).encode()
        self.on_message(self, None, SimpleNamespace(topic=topic, payload=data, retain=retain, qos=qos))

    def disconnect(self):
        self.connected = False
        self.log.append(("disconnect",))

    def loop_stop(self):
        self.log.append(("loop_stop",))

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
