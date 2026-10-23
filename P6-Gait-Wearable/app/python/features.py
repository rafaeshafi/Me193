"""Window features for both models, from a list of (ax, ay, az, gx, gy, gz) samples in g and deg/s.

Pure Python so the same file runs on the board (no numpy there) and in training on the Mac. X is the thigh axis.
Model 1 is allowed exactly one feature, peak_x. Model 2 reads MODEL2_FEATURES.
"""
import math

RATE = 100
X, Y, Z, GX, GY, GZ = range(6)
MODEL1_FEATURES = ["peak_x"]
MODEL2_FEATURES = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_std", "mag_peak",
                   "gyro_rms", "gyro_peak", "dom_freq", "gyro_ratio"]
FREQ_LO, FREQ_HI, FREQ_STEP = 0.5, 4.0, 0.1


def _mean(v):
    return sum(v) / len(v)


def _std(v):
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v))


def _percentile(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dominant_frequency(signal, rate=RATE):
    """Frequency in [FREQ_LO, FREQ_HI] Hz with the most power: a direct DFT of the mean-removed, Hann-windowed signal
    at FREQ_STEP spacing. About 36 x 200 multiply-adds for a 2 s window, a few ms in CPython."""
    n = len(signal)
    m = _mean(signal)
    w = [(signal[i] - m) * (0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]
    best_f, best_p = FREQ_LO, -1.0
    for k in range(int(round((FREQ_HI - FREQ_LO) / FREQ_STEP)) + 1):
        f = FREQ_LO + k * FREQ_STEP
        re = im = 0.0
        step = 2 * math.pi * f / rate
        for i, x in enumerate(w):
            re += x * math.cos(step * i)
            im -= x * math.sin(step * i)
        p = re * re + im * im
        if p > best_p + 1e-12:
            best_f, best_p = f, p
    return round(best_f, 1)


def compute(window, rate=RATE):
    """-> dict of every feature for one window (floats)."""
    ax = [s[X] for s in window]
    ay = [s[Y] for s in window]
    az = [s[Z] for s in window]
    mag = [math.sqrt(s[X] ** 2 + s[Y] ** 2 + s[Z] ** 2) for s in window]
    gyro = [math.sqrt(s[GX] ** 2 + s[GY] ** 2 + s[GZ] ** 2) for s in window]
    mag_std = _std(mag)
    gyro_rms = _rms(gyro)
    return {
        "peak_x": _percentile([abs(a) for a in ax], 0.95),
        "mean_x": _mean(ax), "mean_y": _mean(ay), "mean_z": _mean(az),
        "std_x": _std(ax), "std_y": _std(ay), "std_z": _std(az),
        "mag_std": mag_std,
        "mag_peak": max(mag),
        "gyro_rms": gyro_rms,
        "gyro_peak": max(gyro),
        "dom_freq": dominant_frequency(mag, rate),
        # rotation per unit of impact: large when the thigh swings without hitting anything, i.e. pedalling
        "gyro_ratio": gyro_rms / (1000.0 * mag_std + 1.0),
    }
