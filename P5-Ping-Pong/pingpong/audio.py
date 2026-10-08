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
from pingpong import music as music_module

RATE = 44_100
PEAK = 0.8
MUSIC_GAIN = 0.5                 # the music sits under the effects


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
    "bounce": _tone(520, 45, 14),
    "miss": _tone(180, 380, 160, harmonics=(1.0, 0.5, 0.33)),
    "point": _arpeggio((523, 659, 784, 1047), 80, 60),
    "record": _arpeggio((659, 784, 988, 1319, 1568), 70, 60),
}
PATTERN_SOUND = {"hit_perfect": "hit_perfect", "hit_good": "hit_good", "hit_early": "hit_off", "hit_late": "hit_off",
                 "fault": "miss", "point_won": "point", "record": "record"}


class Mixer:
    """Sums the sounds that are playing; finished ones drop out; the output is clipped to +-1.

    One more channel carries the music: a long clip that fades in, goes round (or plays once), and fades out when it is stopped or
    another tune is asked for; the new one comes in as the old one goes."""

    def __init__(self):
        self._active, self._lock = [], threading.Lock()
        self._music = self._next = None

    @property
    def active(self):
        return len(self._active)

    @property
    def music_active(self):
        return self._music is not None

    def add(self, samples):
        with self._lock:
            self._active.append([samples, 0])

    def set_music(self, samples, *, fade_s=0.6, loop=True):
        with self._lock:
            if self._music is None:
                self._music = {"samples": samples, "pos": 0, "gain": 0.0, "target": 1.0, "step": 1.0 / max(1.0, fade_s * RATE), "loop": loop}
            else:
                self._music.update(target=0.0, step=1.0 / max(1.0, fade_s * RATE))
                self._next = (samples, fade_s, loop)

    def stop_music(self, *, fade_s=0.6):
        with self._lock:
            self._next = None
            if self._music is not None:
                self._music.update(target=0.0, step=1.0 / max(1.0, fade_s * RATE))

    def _music_chunk(self, frames):
        m = self._music
        if m is None:
            return 0.0
        samples, pos = m["samples"], m["pos"]
        index = pos + np.arange(frames)
        if m["loop"]:
            index = index % len(samples)
            chunk = samples[index]
        else:
            chunk = np.zeros(frames, dtype=np.float32)
            live = index < len(samples)
            chunk[live] = samples[index[live]]
        direction = np.sign(m["target"] - m["gain"])
        ramp = m["gain"] + direction * m["step"] * np.arange(1, frames + 1)
        ramp = np.minimum(ramp, m["target"]) if direction > 0 else np.maximum(ramp, m["target"]) if direction < 0 else np.full(frames, m["gain"])
        m["gain"], m["pos"] = float(ramp[-1]), (pos + frames) % len(samples) if m["loop"] else pos + frames
        finished = (m["target"] <= 0.0 and m["gain"] <= 0.0) or (not m["loop"] and m["pos"] >= len(samples))
        if finished:
            self._music = None
            if self._next is not None:
                tune, fade_s, loop = self._next
                self._next = None
                self._music = {"samples": tune, "pos": 0, "gain": 0.0, "target": 1.0, "step": 1.0 / max(1.0, fade_s * RATE), "loop": loop}
        return chunk * ramp * MUSIC_GAIN

    def read(self, frames):
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            for item in self._active:
                chunk = item[0][item[1]:item[1] + frames]
                out[:len(chunk)] += chunk
                item[1] += frames
            self._active = [item for item in self._active if item[1] < len(item[0])]
            out += self._music_chunk(frames)
        return np.clip(out, -1.0, 1.0)


def _default_backend():
    import sounddevice

    return sounddevice


def pick_output_device(devices):
    """The index of the built-in speakers ("MacBook Pro Speakers", ...), or None to use the system default.

    Bluetooth headphones add 150-250 ms (the cue would arrive after the swing) and share the radio with the hub,
    so the sounds go to the speakers even when headphones happen to be the system's default output.
    """
    for index, device in enumerate(devices):
        if device.get("max_output_channels", 0) > 0 and "speakers" in str(device.get("name", "")).lower():
            return index
    return None


class Audio:
    def __init__(self, backend=None, log=print, music=True):
        self.backend, self.log = backend, log
        self.mixer, self.enabled, self.played = Mixer(), False, []
        self.library = {**SOUNDS, **music_module.STINGERS}
        self.music_on, self._muted, self._wanted, self._playing = music, False, None, None
        self._stream = None

    @property
    def muted(self):
        return self._muted

    @muted.setter
    def muted(self, value):
        self._muted = bool(value)
        self._apply_music()

    @property
    def music_playing(self):
        """The tune that is playing (None: none, or the sound is off or muted)."""
        return self._playing

    def play_music(self, name):
        """Ask for a tune by name ("intro", "menu"), or None for quiet; a muted or music-less game keeps the request for later."""
        if name is not None and name not in music_module.TRACKS:
            return
        self._wanted = name
        self._apply_music()

    def _apply_music(self):
        playing = self._wanted if self.enabled and self.music_on and not self._muted else None
        if playing == self._playing:
            return
        self._playing = playing
        if playing is None:
            self.mixer.stop_music()
        else:
            self.mixer.set_music(music_module.TRACKS[playing](), fade_s=0.05 if playing == "intro" else 0.6, loop=playing != "intro")

    def start(self):
        try:
            backend = self.backend or _default_backend()
            options = {}
            listing = getattr(backend, "query_devices", None)
            device = pick_output_device(listing()) if listing is not None else None
            if device is not None:
                options["device"] = device
                self.log(f"sounds on {listing()[device]['name']}")
            self._stream = backend.OutputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=256,
                                                latency="low", callback=self._callback, **options)
            self._stream.start()
            self.enabled = True
            self._apply_music()
        except Exception as exc:                      # no output device, no permission, no sounddevice
            self._stream, self.enabled = None, False
            self.log(f"audio disabled: {exc}")

    def _callback(self, outdata, frames, time, status):
        outdata[:, 0] = self.mixer.read(frames)

    def play(self, name):
        if not self.enabled or self.muted:
            return
        self.played = (self.played + [name])[-200:]
        self.mixer.add(self.library[name])

    def sound_for(self, event, level):
        """The name of the sound an event makes (None for a silent one)."""
        if event.kind == "match_over":
            return "fanfare_win" if event.data.get("winner") == "player" else "fanfare_lose"
        if event.kind == "game_over":
            return "fanfare_lose"
        return PATTERN_SOUND.get(feedback.pattern_for(event, level))

    def play_events(self, events, level):
        for event in events:
            name = self.sound_for(event, level)
            if name:
                self.play(name)

    def stop(self):
        self._wanted = None
        stream, self._stream, self.enabled = self._stream, None, False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
