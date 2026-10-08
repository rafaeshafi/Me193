"""The music of the island, made of nothing but numpy: a dance loop for the menus (a kick on every beat, claps on two and four, hats, a
pumping bass, stabs and a bright arpeggio under a marimba hook) and a piece for the intro that builds up, takes a breath and drops its
biggest hit on the moment the camera reaches the court, plus the little jingles of the menus.

The tunes are original and kept in C major on a pentatonic line over the chords of the bar, so they cannot clash.  They run at 128 beats
a minute; the intro is a rising pickup (GROOVE_S) and four bars of groove, so that the hit comes down exactly on the camera's landing
(LAND_S = intro.ARRIVE_S).  Instruments: a kick, a clap and a snare, hats, a sawtooth bass, chord stabs, plucks and a marimba, a pad,
risers, a crash and a sub boom, and the wash of the sea.  The bass, the stabs and the pad duck on every kick, which is what makes
a dance track pump.  Nothing is loaded from disk.
"""

from functools import lru_cache

import numpy as np

RATE = 44_100
BPM = 128
BEAT_S = 60.0 / BPM
BAR_S = 4 * BEAT_S                     # 1.875 s
LAND_S = 9.0                           # the camera reaches the court (intro.ARRIVE_S): the drop is here
GROOVE_S = LAND_S - 4 * BAR_S          # 1.5 s of build-up before the first bar of the groove
INTRO_S = 10.0
SCALE = (60, 62, 64, 67, 69)           # C D E G A: the major pentatonic every tune is made of
PEAK = 0.8
DRIVE = 1.8                            # how hard the mix is pushed into the soft clipper: louder and punchier, never clipping

# the melody (the hook on the marimba and a pluck an octave up): (start in beats from the top, MIDI note, length in beats, how hard)
INTRO_NOTES = (
    (0, 69, .5, .40), (0.5, 72, .5, .40), (1, 76, .5, .42), (1.5, 81, .5, .45), (2, 76, .5, .42), (2.5, 72, .5, .40), (3, 69, 1, .45),     # bar 1, A minor
    (4, 72, .5, .45), (4.5, 77, .5, .47), (5, 81, .5, .50), (5.5, 84, .5, .52), (6, 81, .5, .50), (6.5, 77, .5, .47), (7, 72, 1, .50),    # bar 2, F
    (8, 76, .5, .52), (8.5, 79, .5, .54), (9, 84, .5, .56), (9.5, 88, .5, .58), (10, 84, .5, .56), (10.5, 79, .5, .54), (11, 76, 1, .56), # bar 3, C
    (12, 74, .25, .56), (12.25, 76, .25, .58), (12.5, 79, .25, .60), (12.75, 81, .25, .62), (13, 83, .25, .64), (13.25, 86, .25, .66),    # bar 4, G: a run up to the court
    (13.5, 88, .25, .68), (13.75, 91, .25, .70), (14, 86, .25, .66), (14.25, 88, .25, .68), (14.5, 91, .25, .70), (14.75, 93, .25, .72),
    (16, 84, 4, .95), (16, 88, 4, .90), (16, 91, 4, .90), (16, 96, 4, .80),                                # the landing: C major, all of it
)
MENU_NOTES = (
    (0, 79, .75, .55), (1, 76, .5, .50), (1.5, 79, .5, .52), (2.5, 81, .5, .54), (3, 79, 1, .52),            # bar 1, C
    (4, 76, .75, .52), (5, 72, .5, .48), (5.5, 76, .5, .50), (6.5, 79, .5, .52), (7, 76, 1, .50),            # bar 2, A minor
    (8, 81, .75, .54), (9, 72, .5, .48), (9.5, 81, .5, .52), (10.5, 84, .5, .56), (11, 81, 1, .52),          # bar 3, F
    (12, 79, .75, .54), (13, 74, .5, .48), (13.5, 79, .5, .52), (14.5, 76, .5, .52), (15, 74, 1, .50),       # bar 4, G
    (16, 88, .75, .58), (17, 84, .5, .52), (17.5, 88, .5, .56), (18.5, 91, .5, .58), (19, 88, 1, .55),       # bar 5, C (the second time round, higher)
    (20, 84, .75, .55), (21, 79, .5, .50), (21.5, 84, .5, .54), (22.5, 88, .5, .56), (23, 84, 1, .52),       # bar 6, A minor
    (24, 89, .75, .58), (25, 81, .5, .52), (25.5, 89, .5, .56), (26.5, 91, .5, .58), (27, 88, 1, .55),       # bar 7, F
    (28, 79, .25, .52), (28.25, 81, .25, .54), (28.5, 84, .25, .56), (28.75, 86, .25, .58), (29, 88, .25, .60), (29.25, 91, .25, .62),
    (29.5, 93, .25, .64), (29.75, 96, .25, .66), (30, 91, .5, .62), (30.5, 88, .5, .58), (31, 86, .5, .56), (31.5, 91, .5, .60),            # bar 8, G: a run to the top
)
# the chord of each bar: (bass note, the chord's notes)
C_, AM, F_, G_ = (48, (60, 64, 67)), (45, (57, 60, 64)), (41, (53, 57, 60)), (43, (55, 59, 62))      # (the bass an octave up: a laptop's speakers cannot play 45 Hz)
INTRO_CHORDS = (AM, F_, C_, G_, C_)
MENU_CHORDS = (C_, AM, F_, G_) * 2


def hz(note):
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _rng(seed):
    return np.random.default_rng(seed)


def _time(seconds):
    return np.arange(round(seconds * RATE)) / RATE


def _band(x, lo=None, hi=None):
    """x with only the frequencies from lo to hi left in it."""
    spectrum, freqs = np.fft.rfft(x), np.fft.rfftfreq(len(x), 1.0 / RATE)
    if lo is not None:
        spectrum[freqs < lo] = 0.0
    if hi is not None:
        spectrum[freqs > hi] = 0.0
    return np.fft.irfft(spectrum, len(x))


def _frozen(y):
    y.setflags(write=False)
    return y


# --- the instruments ------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _marimba(note, seconds, vel):
    n, f = round(seconds * RATE), hz(note)
    t = np.arange(n) / RATE
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.55) + 0.28 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.16)
         + 0.07 * np.sin(2 * np.pi * 10 * f * t) * np.exp(-t / 0.05))
    return _frozen(vel * y * np.minimum(1.0, t / 0.002))


@lru_cache(maxsize=128)
def _pluck(note, seconds, vel, bright=1.0):
    """A bright synth pluck: a sawtooth whose upper harmonics die away faster than its body, like a filter closing."""
    t, f = _time(seconds), hz(note)
    y = sum(np.sin(2 * np.pi * k * f * t) / k * np.exp(-t * (3.0 + 7.0 * k / bright)) for k in range(1, 9) if k * f < 9000)
    return _frozen(vel * y * np.minimum(1.0, t / 0.002) * np.minimum(1.0, (seconds - t) / 0.01))


@lru_cache(maxsize=32)
def _bass(note, seconds, vel):
    """A sawtooth bass with a sine under it: the plucked body of every 8th note."""
    t, f = _time(seconds), hz(note)
    y = 0.9 * np.sin(2 * np.pi * f * t) + sum(np.sin(2 * np.pi * k * f * t) / k * np.exp(-t * (1.5 + 2.0 * k)) for k in range(1, 7))
    return _frozen(vel * np.tanh(1.4 * y) * np.minimum(1.0, t / 0.004) * np.minimum(1.0, (seconds - t) / 0.012))


@lru_cache(maxsize=32)
def _stab(notes, seconds, vel):
    """A short chord stab: the notes together as sawtooths that close like a filter."""
    t = _time(seconds)
    y = sum(np.sin(2 * np.pi * k * hz(note) * t) / k * np.exp(-t * (6.0 + 5.0 * k)) for note in notes for k in range(1, 9) if k * hz(note) < 9000)
    return _frozen(vel * y / len(notes) * np.minimum(1.0, t / 0.003))


def _pad(notes, seconds, vel, attack=0.5):
    n = round(seconds * RATE)
    t = np.arange(n) / RATE
    swell = np.minimum(1.0, t / attack) * np.minimum(1.0, (seconds - t) / 0.4)
    return vel * swell * sum(np.sin(2 * np.pi * hz(note) * t) + 0.3 * np.sin(4 * np.pi * hz(note) * t) for note in notes) / len(notes)


@lru_cache(maxsize=4)
def _kick(vel):
    """A sine that falls from a punch to a low thump, with a click on its front."""
    t = _time(0.42)
    phase = 2 * np.pi * np.cumsum(52.0 + 190.0 * np.exp(-t / 0.03)) / RATE
    y = np.tanh(1.8 * np.sin(phase)) * np.exp(-t / 0.17) + 0.22 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.004)      # (saturated: a laptop hears its punch)
    return _frozen(vel * y * np.minimum(1.0, t / 0.0015))


@lru_cache(maxsize=4)
def _clap(vel):
    """Three quick bursts of noise and the tail after them."""
    t = _time(0.32)
    noise = _band(_rng(11).standard_normal(len(t)), 1100, 5200)
    env = 0.9 * np.exp(-np.maximum(t - 0.03, 0.0) / 0.075) * (t >= 0.03)
    for start in (0.0, 0.011, 0.022):
        env += np.exp(-np.maximum(t - start, 0.0) / 0.007) * (t >= start)
    return _frozen(vel * noise / np.abs(noise).max() * env * np.minimum(1.0, t / 0.001))


@lru_cache(maxsize=8)
def _snare(vel):
    t = _time(0.28)
    noise = _band(_rng(12).standard_normal(len(t)), 1500, 8000)
    y = 0.8 * noise / np.abs(noise).max() * np.exp(-t / 0.075) + 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06)
    return _frozen(vel * y * np.minimum(1.0, t / 0.001))


@lru_cache(maxsize=8)
def _hat(open_, vel):
    t = _time(0.30 if open_ else 0.07)
    noise = _band(_rng(13 + open_).standard_normal(len(t)), 7000, 15000)
    return _frozen(vel * noise / np.abs(noise).max() * np.exp(-t / (0.11 if open_ else 0.016)) * np.minimum(1.0, t / 0.0008))


@lru_cache(maxsize=4)
def _crash(seconds, vel):
    t = _time(seconds)
    noise = _band(_rng(14).standard_normal(len(t)), 3500, 16000)
    return _frozen(vel * noise / np.abs(noise).max() * np.exp(-t / (seconds / 3.0)) * np.minimum(1.0, t / 0.002))


@lru_cache(maxsize=4)
def _boom(vel):
    """The sub drop under the landing: a sine that sinks from a punch to the floor and rings."""
    t = _time(1.6)
    phase = 2 * np.pi * np.cumsum(34.0 + 70.0 * np.exp(-t / 0.25)) / RATE
    return _frozen(vel * np.sin(phase) * np.exp(-t / 0.7) * np.minimum(1.0, t / 0.002))


@lru_cache(maxsize=4)
def _riser(seconds, vel, seed):
    """Noise that climbs from a rumble to a hiss while it swells, with a tone rising under it."""
    n = round(seconds * RATE)
    x = np.arange(n) / RATE / seconds
    noise = _rng(seed).standard_normal(n)
    layers = [_band(noise, 150, 700), _band(noise, 700, 2500), _band(noise, 2500, 8000), _band(noise, 8000, 16000)]
    y = sum(layer / np.abs(layer).max() * np.clip(1.0 - np.abs(x * 3.0 - k), 0.0, 1.0) for k, layer in enumerate(layers))
    tone = np.sin(2 * np.pi * np.cumsum(220.0 * 2.0 ** (x * 3.5)) / RATE) * 0.25
    return _frozen(vel * (y + tone) * x ** 2.2)


def _wash(seconds, vel, seed):
    """The sea: noise smoothed to a low roar, breathing slowly."""
    n = round(seconds * RATE)
    noise = _rng(seed).standard_normal(n)
    roar = np.convolve(noise, np.ones(220) / 220, mode="same")
    t = np.arange(n) / RATE
    return vel * roar * (0.65 + 0.35 * np.sin(2 * np.pi * 0.22 * t)) * 9.0


# --- putting them together ----------------------------------------------------------------------------------------------------------------
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


def _pump(length, kicks, wrap, depth=0.7, release=0.13):
    """The gain that every kick pushes the rest of the music down by and lets back up: the pump of a dance track."""
    curve = 1.0 - depth * np.exp(-np.arange(round(0.7 * RATE)) / RATE / release)
    env = np.ones(length)
    for start in kicks:
        i = round(start * RATE)
        if i >= length:
            continue
        end = i + len(curve)
        env[i:min(end, length)] = np.minimum(env[i:min(end, length)], curve[:min(end, length) - i])
        if wrap and end > length:
            env[:end - length] = np.minimum(env[:end - length], curve[length - i:])
    return env


def _finish(track, fade_in_s=0.0, fade_out_s=0.0):
    """Push the mix into a soft clipper (louder and punchier, no hard clipping), fade its ends, set its peak."""
    track = np.tanh(DRIVE * track / np.abs(track).max())
    if fade_in_s:
        k = round(fade_in_s * RATE)
        track[:k] *= np.linspace(0.0, 1.0, k)
    if fade_out_s:
        k = round(fade_out_s * RATE)
        track[-k:] *= np.linspace(1.0, 0.0, k)
    out = (PEAK * track / np.abs(track).max()).astype(np.float32)
    out.setflags(write=False)
    return out


def _melody(track, notes, wrap, offset_s=0.0, vel=1.0):
    for start, note, length, hard in notes:
        t0 = offset_s + start * BEAT_S
        _put(track, _marimba(note, min(2.2, 0.6 + length * BEAT_S), hard * vel), t0, wrap)
        if note <= 88:                                                          # (a pluck above this is a needle)
            _put(track, _pluck(note + 12, min(1.0, 0.25 + length * BEAT_S), 0.22 * hard * vel, 1.0), t0, wrap)


def _bar(parts, kicks, bar_start, chord, wrap, *, kick=True, hats=2, claps=True, bass=True, stabs=True, arp=False, pad=0.10, vel=1.0):
    """One bar of groove: what the drums, the bass, the stabs and the arpeggio play, as far as `hats` (0 none, 1 off beats, 2 every half
    beat, 4 every sixteenth) and the flags say."""
    drums, low, mids, high = parts
    root, notes = chord
    if kick:
        for beat in range(4):
            _put(drums, _kick(1.0 * vel), bar_start + beat * BEAT_S, wrap)
            kicks.append(bar_start + beat * BEAT_S)
    if claps:
        for beat in (1, 3):
            _put(drums, _clap(0.55 * vel), bar_start + beat * BEAT_S, wrap)
    if hats:
        steps = 16 if hats == 4 else 8
        per_beat = steps // 4
        for k in range(steps):                                                  # the "and" of every beat is the loud one
            loudness = 0.20 if k % per_beat == per_beat // 2 else 0.10 if k % per_beat == 0 else 0.07
            _put(drums, _hat(False, loudness * vel), bar_start + k * BAR_S / steps, wrap)
        _put(drums, _hat(True, 0.16 * vel), bar_start + 3.5 * BEAT_S, wrap)
    if bass:
        for k in range(8):                                                      # a bass note every half beat: the pump leaves the off beats
            _put(low, _bass(root + (12 if k in (3, 7) else 0), BEAT_S * 0.5 * 0.92, 0.42 * vel), bar_start + k * BEAT_S / 2, wrap)
    if stabs:
        for beat in (0.0, 1.5, 3.0):                                            # three, three and two half beats: the dance floor's syncopation
            _put(mids, _stab(notes + (notes[0] + 12,), 0.22, 1.1 * vel), bar_start + beat * BEAT_S, wrap)
    if arp:
        a, b, c = notes
        pattern = (a, b, c, a + 12, b + 12, a + 12, c, b)                       # up through the chord and back, twice a bar
        for k in range(16):
            _put(high, _pluck(pattern[k % 8] + 12, 0.17, 0.42 * vel, 0.7), bar_start + k * BEAT_S / 4, wrap)
    if pad:
        _put(mids, _pad(notes, BAR_S + 0.3, pad * vel, attack=0.5), bar_start, wrap)


def _mix(parts, kicks, wrap):
    drums, low, mids, high = parts
    length = len(drums)
    pump = _pump(length, kicks, wrap)
    return drums * 1.0 + low * pump * 1.0 + mids * (0.35 + 0.65 * pump) + high * (0.55 + 0.45 * pump)


@lru_cache(maxsize=4)
def render(name):
    """A whole track as a read-only float32 array: "intro" (10 s, once) or "menu" (eight bars, to be looped)."""
    if name == "menu":
        n = round(8 * BAR_S * RATE)
        parts, kicks = tuple(np.zeros(n) for _ in range(4)), []
        for bar, chord in enumerate(MENU_CHORDS):
            _bar(parts, kicks, bar * BAR_S, chord, True, hats=4 if bar >= 4 else 2, arp=bar % 4 >= 2)
            if bar % 4 == 3:                                                    # the fill into the next four bars: a snare roll that speeds up
                for k, vel in enumerate((0.35, 0.45, 0.6, 0.8)):
                    _put(parts[0], _snare(vel), bar * BAR_S + 3.0 * BEAT_S + k * BEAT_S / 4, True)
        _put(parts[0], _crash(1.8, 0.5), 0.0, True)
        _put(parts[0], _crash(1.8, 0.4), 4 * BAR_S, True)
        _melody(parts[3], MENU_NOTES, True)
        return _finish(_mix(parts, kicks, True))
    n = round(INTRO_S * RATE)
    parts, kicks = tuple(np.zeros(n) for _ in range(4)), []
    sea = _wash(INTRO_S, 0.05, seed=3) * np.concatenate([np.linspace(0.3, 1.0, round(9.0 * RATE)), np.linspace(1.0, 0.0, n - round(9.0 * RATE))])
    _put(parts[2], _riser(GROOVE_S, 0.9, seed=5), 0.0, False)                    # the pickup: a rise that opens the way in
    for bar, chord in enumerate(INTRO_CHORDS[:4]):                              # the drums come in one after another
        _bar(parts, kicks, GROOVE_S + bar * BAR_S, chord, False, hats=0 if bar == 0 else 2, claps=bar >= 2, stabs=bar >= 2, arp=bar >= 1,
             pad=0.10 + 0.03 * bar, vel=0.8 + 0.07 * bar)
    for k, vel in enumerate((0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)):         # a snare roll over the last half bar
        _put(parts[0], _snare(vel), LAND_S - 4 * BEAT_S / 2 + k * BEAT_S / 4, False)
    _put(parts[2], _riser(2.0, 0.5, seed=6), LAND_S - 2.0, False)
    _melody(parts[3], INTRO_NOTES[:-4], False, offset_s=GROOVE_S)
    mix = _mix(parts, kicks, False) + sea
    breath = np.ones(n)                                                          # a hundred milliseconds of near silence before the hit
    breath[round((LAND_S - 0.105) * RATE):round(LAND_S * RATE)] = 0.02
    mix *= np.convolve(breath, np.ones(round(0.006 * RATE)) / round(0.006 * RATE), mode="same")
    mix[round(LAND_S * RATE):] = sea[round(LAND_S * RATE):]
    land = np.zeros(n)
    _put(land, _kick(1.35), LAND_S, False)
    _put(land, _boom(1.1), LAND_S, False)
    _put(land, _crash(2.2, 0.8), LAND_S, False)
    _put(land, _stab((48, 52, 55, 60, 64, 67, 72, 76, 79), 1.2, 0.9), LAND_S, False)
    _put(land, _pad((48, 52, 55, 60, 64, 67), 1.6, 0.30, attack=0.05), LAND_S, False)                      # the landing swells
    _melody(land, INTRO_NOTES[-4:], False, offset_s=GROOVE_S)
    return _finish(mix + land * 1.5, fade_in_s=0.4, fade_out_s=0.25)


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
