"""The music of the island, made of nothing but numpy: a relaxed marimba-and-bass loop for the menus and a short piece for the intro
whose last, biggest chord lands on the moment the camera reaches the court, plus the little jingles of the menus.

The tunes are original and kept in C major on a pentatonic line over the chords of the bar, so they cannot clash.  Four bars of the
intro last exactly as long as the flight (BAR_S = ARRIVE_S / 4).  Instruments: a marimba (a sine with a few fast-dying overtones), a
soft plucked bass, a pad that swells, a shaker, and the wash of the sea.  Nothing is loaded from disk.
"""

from functools import lru_cache

import numpy as np

RATE = 44_100
BAR_S = 2.25                           # four beats; four bars = the nine seconds the camera takes to arrive (intro.ARRIVE_S)
BEAT_S = BAR_S / 4
INTRO_S = 10.0
SCALE = (60, 62, 64, 67, 69)           # C D E G A: the major pentatonic every tune is made of
PEAK = 0.8

# the melody on the marimba: (start in beats from the top, MIDI note, length in beats, how hard)
INTRO_NOTES = (
    (0, 69, 1, .40), (1, 72, 1, .40), (2, 76, 1, .42), (3, 81, 1, .45),                                    # bar 1, A minor: A C E A
    (4, 72, 1, .45), (5, 77, 1, .47), (6, 81, 1, .50), (7, 84, 1, .52),                                    # bar 2, F: C F A C
    (8, 76, 1, .52), (9, 79, 1, .54), (10, 84, 1, .56), (11, 88, 1, .58),                                  # bar 3, C: E G C E
    (12, 74, .5, .56), (12.5, 76, .5, .58), (13, 79, .5, .60), (13.5, 81, .5, .62), (14, 83, .5, .64),    # bar 4, G: a run up to the court
    (14.5, 86, .5, .66), (15, 88, .5, .68), (15.5, 91, .5, .70),
    (16, 84, 4, .95), (16, 88, 4, .90), (16, 91, 4, .90), (16, 96, 4, .80),                                # the landing: C major, all of it
)
MENU_NOTES = (
    (0, 76, 1, .50), (1.5, 79, .5, .45), (2, 81, 1, .50), (3, 79, 1, .45),                                 # bar 1, C
    (4, 76, 1.5, .50), (5.5, 72, .5, .42), (6, 74, 1, .46), (7, 76, 1, .48),                               # bar 2, A minor
    (8, 81, 1, .50), (9, 84, 1, .52), (10, 81, 1, .48), (11, 79, .5, .44), (11.5, 81, .5, .46),            # bar 3, F
    (12, 79, 2, .50), (14, 74, 1, .44), (15, 76, 1, .46),                                                  # bar 4, G
    (16, 84, 1, .52), (17, 81, 1, .48), (18, 79, 1, .48), (19, 76, 1, .46),                                # bar 5, C (the second time round, higher)
    (20, 81, 1.5, .50), (21.5, 76, .5, .42), (22, 79, 1, .46), (23, 81, 1, .48),                           # bar 6, A minor
    (24, 84, 1, .52), (25, 81, .5, .46), (25.5, 79, .5, .44), (26, 77, 1, .48), (27, 81, 1, .50),          # bar 7, F
    (28, 86, 1, .54), (29, 83, 1, .50), (30, 79, 1.5, .50),                                                # bar 8, G
)
# the chord of each bar: (bass note, the chord's notes)
C_, AM, F_, G_ = (36, (60, 64, 67)), (33, (57, 60, 64)), (29, (53, 57, 60)), (31, (55, 59, 62))
INTRO_CHORDS = (AM, F_, C_, G_, C_)
MENU_CHORDS = (C_, AM, F_, G_) * 2


def hz(note):
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _rng(seed):
    return np.random.default_rng(seed)


# --- the instruments ------------------------------------------------------------------------------------------------------------------
def _marimba(note, seconds, vel):
    n, f = round(seconds * RATE), hz(note)
    t = np.arange(n) / RATE
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.55) + 0.28 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.16)
         + 0.07 * np.sin(2 * np.pi * 10 * f * t) * np.exp(-t / 0.05))
    return vel * y * np.minimum(1.0, t / 0.002)


def _bass(note, seconds, vel=0.5):
    n, f = round(seconds * RATE), hz(note)
    t = np.arange(n) / RATE
    y = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)) * np.exp(-t / 0.45)
    return vel * y * np.minimum(1.0, t / 0.006)


def _pad(notes, seconds, vel, attack=0.5):
    n = round(seconds * RATE)
    t = np.arange(n) / RATE
    swell = np.minimum(1.0, t / attack) * np.minimum(1.0, (seconds - t) / 0.4)
    return vel * swell * sum(np.sin(2 * np.pi * hz(note) * t) + 0.3 * np.sin(4 * np.pi * hz(note) * t) for note in notes) / len(notes)


def _shaker(seconds, vel, seed):
    n = round(seconds * RATE)
    t = np.arange(n) / RATE
    noise = _rng(seed).standard_normal(n)
    return vel * np.diff(noise, prepend=0.0) * np.exp(-t / 0.025)             # a high-passed puff


def _wash(seconds, vel, seed):
    """The sea: noise smoothed to a low roar, breathing slowly."""
    n = round(seconds * RATE)
    noise = _rng(seed).standard_normal(n)
    roar = np.convolve(noise, np.ones(220) / 220, mode="same")
    t = np.arange(n) / RATE
    return vel * roar * (0.65 + 0.35 * np.sin(2 * np.pi * 0.22 * t)) * 9.0


def _put(track, clip, start_s, wrap):
    i = round(start_s * RATE)
    if i >= len(track):                                                         # it would start after the end
        return
    end = i + len(clip)
    if end <= len(track):
        track[i:end] += clip
    elif wrap:                                                                  # a loop: the ring-out of the last notes joins its start
        track[i:] += clip[:len(track) - i]
        track[:end - len(track)] += clip[len(track) - i:]
    else:
        track[i:] += clip[:len(track) - i]


def _finish(track, fade_in_s=0.0):
    if fade_in_s:
        k = round(fade_in_s * RATE)
        track[:k] *= np.linspace(0.0, 1.0, k)
    out = (PEAK * track / np.abs(track).max()).astype(np.float32)
    out.setflags(write=False)
    return out


def _backing(track, chords, wrap, *, swell=False, shaker_from_bar=0, vel=1.0):
    """The bass, the pad and the shaker under the tune."""
    for bar, (root, notes) in enumerate(chords):
        t0 = bar * BAR_S
        _put(track, _bass(root, 1.6 * BEAT_S, 0.55 * vel), t0, wrap)
        _put(track, _bass(root + 7, 1.0 * BEAT_S, 0.38 * vel), t0 + 2.5 * BEAT_S, wrap)
        _put(track, _pad(notes, BAR_S + 0.3, (0.10 if not swell else 0.10 + 0.03 * bar) * vel, attack=0.5), t0, wrap)
        if bar >= shaker_from_bar:
            for k in range(8):
                _put(track, _shaker(0.09, 0.07 * vel * (0.6 if k % 2 == 0 else 1.0), seed=bar * 8 + k), t0 + k * BEAT_S / 2, wrap)


def _melody(track, notes, wrap):
    for start, note, length, vel in notes:
        _put(track, _marimba(note, min(2.2, 1.6 + length * BEAT_S), vel), start * BEAT_S, wrap)


@lru_cache(maxsize=4)
def render(name):
    """A whole track as a read-only float32 array: "intro" (10 s, once) or "menu" (eight bars, to be looped)."""
    if name == "menu":
        track = np.zeros(round(8 * BAR_S * RATE))
        _backing(track, MENU_CHORDS, True, shaker_from_bar=1, vel=0.9)
        _melody(track, MENU_NOTES, True)
        return _finish(track)
    track = np.zeros(round(INTRO_S * RATE))
    track += _wash(INTRO_S, 0.05, seed=3) * np.concatenate([np.linspace(0.3, 1.0, round(9.0 * RATE)), np.linspace(1.0, 0.0, len(track) - round(9.0 * RATE))])
    _backing(track, INTRO_CHORDS, False, swell=True, shaker_from_bar=1, vel=1.0)
    _put(track, _pad((48, 52, 55, 60, 64, 67), 1.6, 0.30, attack=0.05), 4 * BAR_S, False)          # the landing swells
    _put(track, _bass(24, 1.0, 0.9), 4 * BAR_S, False)
    _melody(track, INTRO_NOTES, False)
    return _finish(track, fade_in_s=0.6)


TRACKS = {"intro": lambda: render("intro"), "menu": lambda: render("menu")}


# --- the jingles ----------------------------------------------------------------------------------------------------------------------
def _notes(seq, vel=0.8):
    """Notes one after another: seq of (MIDI note, seconds it is given before the next)."""
    total = sum(gap for _, gap in seq) + 0.9
    out = np.zeros(round(total * RATE))
    at = 0.0
    for note, gap in seq:
        _put(out, _marimba(note, 0.8, vel), at, False)
        at += gap
    return out


def _stinger(clip):
    out = (PEAK * clip / np.abs(clip).max()).astype(np.float32)
    out.setflags(write=False)
    return out


def _vs():
    n = round(0.85 * RATE)
    t = np.arange(n) / RATE
    drum = np.sin(2 * np.pi * np.cumsum(np.linspace(190.0, 55.0, n)) / RATE) * np.exp(-t / 0.14)
    crash = np.diff(_rng(9).standard_normal(n), prepend=0.0) * np.exp(-t / 0.22) * 0.35
    chord = _marimba(72, 0.85, 0.5) + _marimba(76, 0.85, 0.5) + _marimba(79, 0.85, 0.5)
    return drum + crash + chord * 0.5


def _win():
    clip = _notes(((72, .09), (76, .09), (79, .09), (84, .16)), 0.8)
    for note in (84, 88, 91):
        _put(clip, _marimba(note, 1.0, 0.55), 0.45, False)
    return clip


STINGERS = {
    "menu_tick": _stinger(_notes(((88, 0.0),), 0.5)[:round(0.12 * RATE)]),
    "menu_select": _stinger(_notes(((79, .07), (84, .0)), 0.8)[:round(0.45 * RATE)]),
    "menu_back": _stinger(_notes(((69, .08), (64, .0)), 0.7)[:round(0.4 * RATE)]),
    "vs": _stinger(_vs()),
    "fanfare_win": _stinger(_win()[:round(1.5 * RATE)]),
    "fanfare_lose": _stinger(_notes(((64, .26), (60, .26), (57, .0)), 0.7)[:round(1.4 * RATE)]),
}
