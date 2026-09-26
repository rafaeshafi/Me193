"""Whistle pitch detection and the driving policy. No hardware in here.

detect_pitch() turns one block of microphone samples into a whistle pitch, or
None when the block does not look like a whistle. Policy.step() turns that
stream of pitches into left/right motor speeds and a GOAL flag.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

import config

SAMPLE_RATE = 44100
BLOCK = 2048                 # samples per analysis block, ~46 ms
BAND = (500.0, 3500.0)       # whistles live here; everything else is ignored

MIN_RMS = config.MIN_LOUDNESS      # absolute loudness gate (full scale 1.0)
MIN_SNR_DB = config.MIN_ABOVE_ROOM_DB  # peak must beat the room's level at that frequency
MIN_TONALITY_DB = 20.0       # peak must beat the median of the band: one narrow line
MAX_OUTSIDE_DB = 6.0         # nothing outside the band may be louder than peak + this
                             # (voices and hum put their strongest line below 500 Hz)
SILENCE_TIMEOUT = 0.3        # s without a valid whistle before the car stops
SMOOTHING = 5                # median over this many recent pitch estimates

FREQS = np.fft.rfftfreq(BLOCK, 1 / SAMPLE_RATE)
IN_BAND = (FREQS >= BAND[0]) & (FREQS <= BAND[1])
OUT_BAND = (FREQS >= 80) & ~IN_BAND   # below 80 Hz is mostly handling rumble
_WINDOW = np.hanning(BLOCK)


def spectrum_db(block):
    """Magnitude spectrum of one block in dB (Hann-windowed)."""
    mags = np.abs(np.fft.rfft(block * _WINDOW)) / (BLOCK / 4)
    return 20 * np.log10(mags + 1e-10)


def noise_floor(blocks):
    """Per-frequency room level, averaged over a few seconds of ambient blocks."""
    return np.mean([spectrum_db(b) for b in blocks], axis=0)


@dataclass
class Detection:
    pitch: float | None      # Hz, or None if this block is not a whistle
    peak_hz: float           # strongest in-band frequency, whistle or not
    rms: float
    snr_db: float
    tonality_db: float
    reason: str              # why it was rejected, or "whistle"


def detect_pitch(block, floor_db):
    """Find the whistle in one block, applying every noise mask in turn."""
    db = spectrum_db(block)
    band_idx = np.flatnonzero(IN_BAND)
    i = band_idx[np.argmax(db[band_idx])]
    rms = float(np.sqrt(np.mean(block ** 2)))
    snr = float(db[i] - floor_db[i])
    tonality = float(db[i] - np.median(db[band_idx]))
    outside = float(db[OUT_BAND].max() - db[i])

    # Parabolic interpolation between neighbouring bins: ~21 Hz bins become ~2 Hz.
    a, b, c = db[i - 1], db[i], db[i + 1]
    offset = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    peak_hz = float(FREQS[i] + offset * (FREQS[1] - FREQS[0]))

    if rms < MIN_RMS:
        reason = "too quiet"
    elif snr < MIN_SNR_DB:
        reason = "not above room noise"
    elif tonality < MIN_TONALITY_DB:
        reason = "not tonal (noise)"
    elif outside > MAX_OUTSIDE_DB:
        reason = "louder sound outside band (voice/hum)"
    else:
        reason = "whistle"
    pitch = peak_hz if reason == "whistle" else None
    return Detection(pitch, peak_hz, rms, snr, tonality, reason)


@dataclass
class Bands:
    f_min: float
    f_center: float
    f_max: float
    f_goal: float
    dead_band: float


@dataclass
class Decision:
    left: float
    right: float
    steer: float             # -1 hard right .. 0 straight .. +1 hard left
    label: str
    goal: bool = False
    goal_progress: float = 0.0  # 0..1 while the goal whistle is being held


class Policy:
    """Pitch stream -> tank-drive speeds.

    Middle whistle drives straight; higher turns left, lower turns right, in
    proportion to how far from the middle. Silence stops the car. A whistle
    held at the very top claims a goal.
    """

    def __init__(self, bands, base_speed, turn_gain=1.0, goal_hold=0.75):
        self.bands = bands
        self.base = base_speed
        self.turn_gain = turn_gain
        self.goal_hold = goal_hold
        self.history = deque(maxlen=SMOOTHING)
        self.last_whistle = -1e9
        self.goal_since = None
        self.straight = True  # for hysteresis at the dead-band edge

    def step(self, pitch, now=None):
        now = time.monotonic() if now is None else now
        b = self.bands

        if pitch is None:
            self.goal_since = None
            if now - self.last_whistle > SILENCE_TIMEOUT:
                self.history.clear()
                self.straight = True
                return Decision(0, 0, 0, "STOP - no whistle")
            if not self.history:
                return Decision(0, 0, 0, "STOP - no whistle")
            # A brief dropout: keep doing what the last whistle said.
        else:
            self.last_whistle = now
            if pitch >= b.f_goal:
                # Goal band: stand still while the hold timer runs, so the car
                # does not lurch left on the way up to the goal note.
                self.history.clear()
                self.goal_since = self.goal_since or now
                held = now - self.goal_since
                if held >= self.goal_hold:
                    return Decision(0, 0, 0, "GOAL!", goal=True, goal_progress=1.0)
                return Decision(0, 0, 0, f"goal whistle... {held:.1f}s",
                                goal_progress=held / self.goal_hold)
            self.goal_since = None
            self.history.append(pitch)

        f = float(np.median(self.history))
        offset = f - b.f_center
        # Hysteresis: leaving "straight" needs a bit more than the dead band,
        # so a whistle sitting right on the edge does not twitch the car.
        edge = b.dead_band * (1.25 if self.straight else 1.0)
        if abs(offset) <= edge:
            self.straight = True
            return Decision(self.base, self.base, 0.0, f"STRAIGHT {f:.0f} Hz")
        self.straight = False

        if offset > 0:
            span = max(b.f_max - (b.f_center + b.dead_band), 1.0)
            steer = float(np.clip((offset - b.dead_band) / span, 0.0, 1.0))
        else:
            span = max((b.f_center - b.dead_band) - b.f_min, 1.0)
            steer = -float(np.clip((-offset - b.dead_band) / span, 0.0, 1.0))

        # Tank drive: slow the inside wheel. steer > 0 is left, so the left
        # wheel is the inside one.
        inner = self.base * (1 - self.turn_gain * abs(steer))
        left, right = (inner, self.base) if steer > 0 else (self.base, inner)
        side = "LEFT" if steer > 0 else "RIGHT"
        return Decision(left, right, steer, f"{side} {abs(steer):.0%}  {f:.0f} Hz")
