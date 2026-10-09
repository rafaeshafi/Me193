"""ThreadedFakeRig: the live pipeline in REAL time with REAL threads, on fake hardware.

fakerig.FakeRig steps one simulated clock on one thread; this is the same game wired the way `./pp play` runs it on
the Mac.  A "BLE" thread calls the hub's notification callback at 66 Hz (the legoeducation event-loop thread does
exactly that), the camera thread blocks in read() at 30 fps inside the real VisionWorker, the IMU worker and the
actuator are the real threads, and the game thread pumps the LiveRig at ~60 Hz.  What it can show that the
single-threaded rig cannot: a deadlock, a race on a shared structure, an exception a worker swallowed, a thread
that outlives close().  Every thread's uncaught exception and every message a worker logs are collected.
"""

import threading
import time

from pingpong import live, posegyro
from pingpong.clock import Clock
from pingpong.fakerig import GPD, S, ScriptedPlayer, VibratingMotor
from pingpong.hub import HubLink
from pingpong.profile import Calibration
from pingpong.sources_fake import FakeCamera, FakeLandmarker, FakeMqttClient, FakeTagDetector

PUMP_S = 1.0 / 60.0


class PacedCamera:
    """FakeCamera whose read() blocks until the next frame is due, like a capture device does."""

    def __init__(self, fps):
        self.inner, self.period, self._due = FakeCamera(), 1.0 / fps, None

    def isOpened(self):
        return self.inner.isOpened()

    def read(self):
        now = time.monotonic()
        self._due = now if self._due is None else max(self._due, now - self.period)       # never owe a burst of frames
        if self._due > now:
            time.sleep(self._due - now)
        self._due += self.period
        return self.inner.read()

    def release(self):
        self.inner.release()


class ThreadedFakeRig:
    def __init__(self, *, level=1, mode="survival", target=7, seed=1, w_pk=600.0, stale_ms=300.0, swing_source="imu",
                 hz=66.0, fps=30.0, lag_s=0.10, cards=None, vibration=False, no_motor=False, record_dir=None):
        self.clock = Clock()
        self.origin_ns = self.clock.now_ns()
        camera = swing_source == "pose"
        calibration = Calibration.default(swing_source)
        self.player = ScriptedPlayer(calibration.box, w_pk=w_pk, cards=[(0.6, 3.2, 0)] if cards is None else cards,
                                     pose_motion=camera)
        self.player.origin_ns = self.origin_ns
        self.dev = VibratingMotor(self.player, self.clock, vibration)
        self.client = FakeMqttClient()
        self.hz = hz
        self.hub_blackouts, self.pose_blackouts = [], []
        self.pause_reasons_seen = set()
        self.thread_errors, self.logs = [], []
        hub = HubLink(self.dev, notify_ms=15, clock=self.clock)
        hub.connect()
        self.rig = live.assemble(
            hub=hub, capture=PacedCamera(fps),
            landmarker=FakeLandmarker(self._hand_if_visible, self.clock, lag_s=lag_s, hand=calibration.hand),
            calibration=calibration, clock=self.clock, tag_detector=FakeTagDetector(self.player.card_ids, self.clock),
            mqtt_client=self.client, level=level, mode=mode, target=target, seed=seed, source="live",
            no_motor=no_motor, threaded=True, lag_s=lag_s, gyro_per_dps=GPD, accel_per_g=1000.0, fs_raw=32767,
            stale_ms=stale_ms, to_image=lambda frame: frame, record_dir=record_dir, player="fake",
            pose_gyro=posegyro.PoseGyro() if camera else None, log=self.logs.append)
        self.session, self.game = self.rig.session, self.rig.session.game
        self._stop, self._started, self._closed = threading.Event(), False, False
        self._ble = threading.Thread(target=self._notify_loop, name="fake-ble", daemon=True)
        self._old_hook = None

    # --- the hardware's own threads --------------------------------------------------------------------------------
    def _within(self, windows, t_ns):
        rel = (t_ns - self.origin_ns) / S
        return any(a <= rel < b for a, b in windows)

    def _hand_if_visible(self, t_ns):
        return None if self._within(self.pose_blackouts, t_ns) else self.player.hand_uv(t_ns)

    def _notify_loop(self):
        period, due = 1.0 / self.hz, time.monotonic()
        while not self._stop.is_set():
            now_ns = self.clock.now_ns()
            if not self._within(self.hub_blackouts, now_ns):               # a silent hub delivers nothing at all
                self.dev.emit(imu=self.player.imu(now_ns))
            due += period
            delay = due - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                due = time.monotonic()

    # --- the game thread (the caller's) --------------------------------------------------------------------------------
    def start(self):
        self._started = True
        self.origin_ns = self.clock.now_ns()                                # the START card is timed from here
        self.player.origin_ns = self.origin_ns
        self._old_hook = threading.excepthook
        threading.excepthook = lambda args: self.thread_errors.append(args)
        self.rig.start()
        self._ble.start()
        return self

    def now_s(self):
        return (self.clock.now_ns() - self.origin_ns) / S

    def step(self):
        self.player.watch(self.game, self.clock.now_ns())
        self.rig.pump()
        if self.game.paused:
            self.pause_reasons_seen |= self.game.pause_reasons
        time.sleep(PUMP_S)

    def run(self, *, until=None, seconds=None, max_s=60.0):
        """Pump for `seconds` real seconds, or until `until()` (AssertionError after max_s real seconds)."""
        end = time.monotonic() + (max_s if seconds is None else seconds)
        while time.monotonic() < end:
            self.step()
            if until is not None and until():
                return True
        if until is not None:
            raise AssertionError(f"not reached within {max_s:.0f} s: phase={self.game.phase} "
                                 f"streak={self.game.tracker.streak} paused={sorted(self.game.pause_reasons)}")
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.rig.close()
        finally:
            self._stop.set()
            if self._ble.is_alive():
                self._ble.join(timeout=2.0)
            if self._old_hook is not None:
                threading.excepthook = self._old_hook
