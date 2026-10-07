"""Game audio: synthesized sounds, a tiny mixer, and the game-event -> sound mapping.

Nothing is loaded from disk: each sound is a few lines of numpy (a pop whose pitch tells you how good the
hit was, a low buzz for a miss, an arpeggio for a point or a record, ticks for the countdown).  A callback
stream pulls from a mixer, so overlapping sounds add up instead of cutting each other off.  The built-in
Mac speakers are the output (Bluetooth audio adds 150-250 ms).  A machine with no output device just has
no sound: the game never depends on it.
"""

import threading

import numpy as np

from pingpong import feedback

RATE = 44_100
PEAK = 0.8


def _norm(y):
    return (PEAK * y / np.abs(y).max()).astype(np.float32)


def _tone(freq, ms, tau_ms, *, glide_to=None, harmonics=(1.0,)):
    n = int(RATE * ms / 1000)
    t = np.arange(n) / RATE
    f = np.full(n, float(freq)) if glide_to is None else np.linspace(freq, glide_to, n)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    y = sum(h * np.sin((k + 1) * phase) for k, h in enumerate(harmonics))
    return _norm(y * np.minimum(1.0, t / 0.003) * np.exp(-t / (tau_ms / 1000)))


def _arpeggio(freqs, ms_each, tau_ms):
    return _norm(np.concatenate([_tone(f, ms_each, tau_ms).astype(float) for f in freqs]))


SOUNDS = {
    "tick": _tone(1000, 60, 18),
    "go": _tone(1500, 160, 70),
    "hit_good": _tone(880, 90, 28),
    "hit_perfect": _tone(1200, 110, 40, glide_to=1800, harmonics=(1.0, 0.4)),
    "hit_off": _tone(660, 90, 28),
    "miss": _tone(180, 380, 160, harmonics=(1.0, 0.5, 0.33)),
    "point": _arpeggio((523, 659, 784, 1047), 80, 60),
    "record": _arpeggio((659, 784, 988, 1319, 1568), 70, 60),
}
PATTERN_SOUND = {"hit_perfect": "hit_perfect", "hit_good": "hit_good", "hit_early": "hit_off", "hit_late": "hit_off",
                 "fault": "miss", "point_won": "point", "record": "record"}


class Mixer:
    """Sums the sounds that are playing; finished ones drop out; the output is clipped to +-1."""

    def __init__(self):
        self._active, self._lock = [], threading.Lock()

    @property
    def active(self):
        return len(self._active)

    def add(self, samples):
        with self._lock:
            self._active.append([samples, 0])

    def read(self, frames):
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            for item in self._active:
                chunk = item[0][item[1]:item[1] + frames]
                out[:len(chunk)] += chunk
                item[1] += frames
            self._active = [item for item in self._active if item[1] < len(item[0])]
        return np.clip(out, -1.0, 1.0)


def _default_backend():
    import sounddevice

    return sounddevice


class Audio:
    def __init__(self, backend=None, log=print):
        self.backend, self.log = backend, log
        self.mixer, self.muted, self.enabled, self.played = Mixer(), False, False, []
        self._stream = None

    def start(self):
        try:
            backend = self.backend or _default_backend()
            self._stream = backend.OutputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=256,
                                                latency="low", callback=self._callback)
            self._stream.start()
            self.enabled = True
        except Exception as exc:                      # no output device, no permission, no sounddevice
            self._stream, self.enabled = None, False
            self.log(f"audio disabled: {exc}")

    def _callback(self, outdata, frames, time, status):
        outdata[:, 0] = self.mixer.read(frames)

    def play(self, name):
        if not self.enabled or self.muted:
            return
        self.played = (self.played + [name])[-200:]
        self.mixer.add(SOUNDS[name])

    def play_events(self, events, level):
        for event in events:
            name = PATTERN_SOUND.get(feedback.pattern_for(event, level))
            if name:
                self.play(name)

    def stop(self):
        stream, self._stream, self.enabled = self._stream, None, False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
