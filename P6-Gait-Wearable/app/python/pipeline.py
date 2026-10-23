"""Gait: IMU batches in, both verdicts out. Owns the ring buffer, the two trees, the votes and the recorder."""
import threading
import time

from display import payload as tft_payload, wave_byte
from features import X, compute
from imu import RATE, Ring, decode
from models import Vote

WINDOW_S = 2.0
HOP_S = 0.5
IMU_TIMEOUT_S = 1.5
WAVE_DECIMATE = 2  # 100 Hz samples -> 50 Hz sweep columns
# dom_freq is the thigh swing rate: a stride has two steps, a pedal cycle is one revolution
CADENCE_MULT = {"WALK": 2.0, "RUN": 2.0, "BIKE": 1.0}


class Gait:
    def __init__(self, model1, model2, recorder, rate=RATE, clock=time.monotonic):
        self.m1, self.m2, self.recorder, self.rate, self._clock = model1, model2, recorder, rate, clock
        self.ring = Ring(5.0, rate)
        self.vote1, self.vote2 = Vote(3), Vote(3)
        self._lock = threading.Lock()
        self._wave = bytearray()
        self._last_sample_t = None
        self._resets = None
        self._next_tick = 0.0
        self.state = {"act": "---", "m1": "---", "cad": None, "pk": None, "features": {}, "dropped": 0, "imu": False}

    # --- from the Bridge thread -------------------------------------------------------------------------------
    def on_batch(self, data):
        try:
            seq, resets, samples = decode(data)
        except ValueError:
            return
        now = self._clock()
        with self._lock:
            if resets != self._resets:  # the sensor stream restarted: older samples are not contiguous with these
                self.ring.clear()
                self._resets = resets
            if samples:
                self._last_sample_t = now
                self.ring.push(seq, samples)
                self.recorder.on_samples(now, samples, self.rate)
                self._wave.extend(wave_byte(s[X]) for s in samples[::WAVE_DECIMATE])
            elif self._last_sample_t is None:
                self._last_sample_t = -1.0  # heard the sketch, it has no IMU

    def take_wave(self):
        with self._lock:
            out = bytes(self._wave)
            self._wave.clear()
        return out

    # --- from the main loop -------------------------------------------------------------------------------------
    def tick(self, now=None):
        """Every HOP_S: classify the newest window. Returns True when it ran (the payload may have changed)."""
        now = self._clock() if now is None else now
        if now < self._next_tick:
            return False
        self._next_tick = now + HOP_S
        with self._lock:
            heard = self._last_sample_t is not None
            imu_ok = heard and self._last_sample_t >= 0 and now - self._last_sample_t < IMU_TIMEOUT_S
            win = self.ring.window(WINDOW_S) if imu_ok else None
            dropped = self.ring.dropped
        if not imu_ok:
            self.state.update(act="NO IMU" if heard else "---", m1="---", cad=None, pk=None, imu=False)
            return True
        if win is None:
            self.state.update(act="---", m1="---", cad=None, pk=None, imu=True)
            return True
        f = compute(win, self.rate)
        a1 = self.vote1.push(self.m1.predict(f))
        a2 = self.vote2.push(self.m2.predict(f))
        cad = None if a2 == "STAND" else int(round(f["dom_freq"] * 60 * CADENCE_MULT.get(a2, 1.0)))
        self.state.update(act=a2, m1=a1, cad=cad, pk=f["peak_x"], features=f, dropped=dropped, imu=True)
        return True

    def payload(self):
        s = self.state
        return tft_payload(s["act"], s["m1"], s["cad"], s["pk"], prompt=self.recorder.prompt(),
                       st="REC" if self.recorder.active else "")

    def snapshot(self):
        """Everything the phone page shows, as plain JSON-able types."""
        with self._lock:
            recent = self.ring.window(1.5) or []
        s = self.state
        return {"act": s["act"], "m1": s["m1"], "cad": s["cad"], "pk": s["pk"], "imu": s["imu"], "dropped": s["dropped"],
                "features": {k: round(v, 4) for k, v in s["features"].items()},
                "wave": [round(r[X], 3) for r in recent],
                "recorder": self.recorder.status(), "sessions": self.recorder.sessions()}
