"""ImuWorker: hub IMU samples -> SwingDetector -> SwingEvents, on its own thread.

It BLOCKS on the sample queue (no 60 Hz polling latency), so an IMPACT is emitted as
soon as the falling edge arrives.  The detector is not thread-safe, so every touch of it
(feed, blank, parameter swap) happens under one lock: blank windows come from the
actuator thread, new calibration parameters from the main thread.
"""

import queue
import threading

from pingpong.swing import SwingDetector


class ImuWorker:
    def __init__(self, samples, detector):
        self.samples = samples                       # queue.SimpleQueue of ImuSample (HubLink.imu)
        self.detector = detector
        self._events = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="imu", daemon=True)

    # --- control (any thread) ----------------------------------------------------------------
    def blank(self, start_ns, end_ns):
        with self._lock:
            self.detector.blank(start_ns, end_ns)

    def set_params(self, params):
        with self._lock:
            self.detector = SwingDetector(params)

    def poll(self):
        events = []
        while not self._events.empty():
            events.append(self._events.get_nowait())
        return events

    # --- processing -----------------------------------------------------------------------------
    def _feed(self, sample):
        with self._lock:
            for event in self.detector.feed(sample):
                self._events.put_nowait(event)

    def step(self):
        """Drain whatever is queued right now (synchronous mode for tests / the fake rig)."""
        n = 0
        while True:
            try:
                sample = self.samples.get_nowait()
            except queue.Empty:
                return n
            self._feed(sample)
            n += 1

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                sample = self.samples.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._feed(sample)
            except Exception as exc:                 # a bad sample must not kill the detector thread
                print(f"imu worker: {exc}")

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
