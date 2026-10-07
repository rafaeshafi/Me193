"""HubLink: the Double Motor's BLE notifications -> thread-safe queues.

Rules this module exists to enforce (all verified in the legoeducation source):
  * The notification callback runs ON the library's event-loop thread and sync
    library calls inside it raise RuntimeError, so it only parses, stamps and
    enqueues -- never touches the device.
  * connect() defaults to 100 ms notifications (10 Hz): the delay must be passed
    on every connect AND reconnect.
  * connect() never raises when the hub is not found; check .connected.
  * The hub sends no timestamps, so samples are stamped on arrival.
  * Commands (beep, haptics) are NOT sent from here: the actuator thread owns
    them, because even blocking=False blocks its caller on the BLE write.
"""

import queue

import legoeducation as le

from pingpong.clock import Clock
from pingpong.events import GestureEvent, ImuSample

COLORS = {
    "green": le.LEGO_COLOR_GREEN, "blue": le.LEGO_COLOR_BLUE, "red": le.LEGO_COLOR_RED,
    "orange": le.LEGO_COLOR_ORANGE, "yellow": le.LEGO_COLOR_YELLOW, "azure": le.LEGO_COLOR_AZURE,
    "purple": le.LEGO_COLOR_PURPLE, "magenta": le.LEGO_COLOR_MAGENTA,
}


def card_kwargs(color, serial):
    """Connection-Card filter for connect(): colour name + 4-digit serial STRING."""
    key = str(color).strip().lower()
    if key not in COLORS:
        raise ValueError(f"unknown card colour {color!r}; choose from {sorted(COLORS)}")
    text = str(serial).strip()
    if not text.isdigit() or len(text) > 4:
        raise ValueError(f"card serial must be 1-4 digits, got {serial!r}")
    return {"card_color": COLORS[key], "card_serial": text.zfill(4)}


class HubLink:
    def __init__(self, device, *, notify_ms=15, clock=None, card=None):
        self.dev = device
        self.notify_ms = notify_ms
        self.clock = clock or Clock()
        self.card = dict(card or {})
        self.imu = queue.SimpleQueue()        # ImuSample, raw counts
        self.gestures = queue.SimpleQueue()   # GestureEvent (log-only by plan)
        self.last_rx_ns = None
        self.n_samples = 0
        self._closed = False

    @classmethod
    def create(cls, color, serial, *, notify_ms=15, clock=None):
        return cls(le.DoubleMotor(), notify_ms=notify_ms, clock=clock, card=card_kwargs(color, serial))

    # --- connection --------------------------------------------------------------
    def connect(self):
        self.dev.set_notification_callback(self._on_notify)
        self.dev.connect(**self.card, device_notification_delay=self.notify_ms)
        if not self.dev.connected:
            raise ConnectionError(
                "Double Motor not found: wake it with its button, check the card colour/serial "
                "(./pp scan_hubs), and make sure no other program holds the connection")

    def reconnect(self):
        self.connect()   # the library no-ops if still connected; the delay is re-passed

    @property
    def connected(self):
        return bool(self.dev.connected)

    # --- notification path (library loop thread: O(1), no device calls) ----------------
    def _on_notify(self, data):
        t_ns = self.clock.now_ns()
        self.last_rx_ns = t_ns
        for item in le.device_notification_parser(data):
            if isinstance(item, le.ImuDeviceNotification):
                self.n_samples += 1
                self.imu.put_nowait(ImuSample(
                    t_ns=t_ns,
                    g=(item.gyroscopeX, item.gyroscopeY, item.gyroscopeZ),
                    a=(item.accelerometerX, item.accelerometerY, item.accelerometerZ)))
            elif isinstance(item, le.ImuGestureNotification):
                if item.gesture != le.MOTION_GESTURE_NO_GESTURE:
                    self.gestures.put_nowait(GestureEvent(t_ns=t_ns, gesture=item.gesture))

    # --- health --------------------------------------------------------------------
    def stale_ms(self):
        if self.last_rx_ns is None:
            return None
        return (self.clock.now_ns() - self.last_rx_ns) / 1e6

    def is_stale(self, threshold_ms):
        age = self.stale_ms()
        return age is None or age > threshold_ms

    def battery_pct(self):
        """Battery percent, or None until the hub has reported (the library starts at 0, which means "unknown")."""
        info = getattr(self.dev, "info_device", None)
        level = getattr(info, "batteryLevel", None) if info is not None else None
        return level or None

    # --- teardown (a killed process leaves the hub "connected" ~24 s) ------------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        for step in (lambda: self.dev.motor_stop(motor=le.MOTOR_BOTH, blocking=False),
                     self.dev.disconnect):
            try:
                step()
            except Exception:
                pass
