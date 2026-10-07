"""ImuWorker: hub IMU samples -> SwingDetector (+ ShakeMonitor) -> SwingEvents / shake locks, on its own thread.

It BLOCKS on the sample queue (no 60 Hz polling latency), so an IMPACT is emitted as
soon as the falling edge arrives.  The detector is not thread-safe, so every touch of it
(feed, blank, parameter swap) happens under one lock: blank windows come from the
actuator thread, new calibration parameters from the main thread.
"""

import queue
import threading

from pingpong.swing import SwingDetector


class ImuWorker:
    def __init__(self, samples, detector, shake=None, recorder=None, log=print):
        self.samples = samples                       # queue.SimpleQueue of ImuSample (HubLink.imu)
        self.detector = detector
        self.shake = shake                           # optional ShakeMonitor (judge gate J6)
        self.recorder = recorder                     # optional Recorder: gets every raw sample
        self.log = log
        self._events = queue.SimpleQueue()
        self._locks = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="imu", daemon=True)

    # --- control (any thread) ----------------------------------------------------------------
    def blank(self, start_ns, end_ns):
        with self._lock:
            self.detector.blank(start_ns, end_ns)
            if self.shake is not None:
                self.shake.blank(start_ns, end_ns)

    def set_params(self, params):
        with self._lock:
            self.detector = SwingDetector(params)

    def trace(self, seconds):
        with self._lock:
            return self.detector.trace(seconds)

    def poll(self):
        events = []
        while not self._events.empty():
            events.append(self._events.get_nowait())
        return events

    def poll_locks(self):
        """Times (ns) the paddle should be locked until, from the shake monitor."""
        locks = []
        while not self._locks.empty():
            locks.append(self._locks.get_nowait())
        return locks

    # --- processing -----------------------------------------------------------------------------
    def _feed(self, sample):
        if self.recorder is not None:
            self.recorder.imu(sample)                # raw truth, including blanked samples
        with self._lock:
            for event in self.detector.feed(sample):
                self._events.put_nowait(event)
            if self.shake is not None:
                until = self.shake.feed(sample)
                if until is not None:
                    self._locks.put_nowait(until)

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
                self.log(f"imu worker: {exc}")

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
