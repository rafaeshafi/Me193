"""Play the death / victory songs through the laptop speaker with PyAudio.

A song in config.py is either a .wav path or a list of (note, beats) pairs.
Preview one without the car:  python songs.py death   (or victory)
"""

import re
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import pyaudio

import config

RATE = 44100
_NOTE = re.compile(r"^([A-G])([#b]?)(-?\d)$")
_SEMITONE = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}


def note_hz(name):
    """'A4' -> 440.0, 'C#5' -> 554.4, 'R' -> 0 (rest)."""
    if name.upper() == "R":
        return 0.0
    m = _NOTE.match(name)
    if not m:
        raise ValueError(f"Bad note {name!r}: use names like C4, F#5, Bb3, or R")
    letter, accidental, octave = m.groups()
    n = _SEMITONE[letter] + {"#": 1, "b": -1, "": 0}[accidental] + 12 * (int(octave) - 4)
    return 440.0 * 2 ** (n / 12)


def synth(notes, tempo_bpm):
    """Render (note, beats) pairs to float32 samples, with a soft attack/release."""
    beat = 60.0 / tempo_bpm
    parts = []
    for name, beats in notes:
        n = int(RATE * beats * beat)
        t = np.arange(n) / RATE
        tone = 0.4 * np.sin(2 * np.pi * note_hz(name) * t)
        ramp = min(n // 2, int(0.01 * RATE))
        env = np.ones(n)
        if ramp:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        parts.append(tone * env)
    return np.concatenate(parts).astype(np.float32).tobytes(), pyaudio.paFloat32, 1, RATE


def load_wav(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    with wave.open(str(path), "rb") as w:
        return (w.readframes(w.getnframes()), pyaudio.get_format_from_width(w.getsampwidth()),
                w.getnchannels(), w.getframerate())


def render(song):
    if isinstance(song, (str, Path)):
        return load_wav(song)
    return synth(song, config.TEMPO_BPM)


class Player:
    """Plays one song at a time on a background thread.

    `playing` is True while a song is sounding, so the whistle detector can
    ignore the microphone and not hear the car's own speaker.
    """

    def __init__(self, pa):
        self.pa = pa
        self.playing = False
        self._thread = None

    def play(self, song):
        if self.playing:
            return
        data, fmt, channels, rate = render(song)
        self.playing = True
        self._thread = threading.Thread(target=self._run, args=(data, fmt, channels, rate),
                                        daemon=True)
        self._thread.start()

    def _run(self, data, fmt, channels, rate):
        try:
            out = self.pa.open(format=fmt, channels=channels, rate=rate, output=True)
            out.write(data)
            out.stop_stream()
            out.close()
        finally:
            self.playing = False

    def wait(self):
        if self._thread is not None:
            self._thread.join()


if __name__ == "__main__":
    songs = {"death": config.DEATH_SONG, "victory": config.VICTORY_SONG}
    which = sys.argv[1] if len(sys.argv) > 1 else "victory"
    if which not in songs:
        raise SystemExit(f"usage: python songs.py [{'|'.join(songs)}]")
    pa = pyaudio.PyAudio()
    player = Player(pa)
    player.play(songs[which])
    player.wait()
    pa.terminate()
