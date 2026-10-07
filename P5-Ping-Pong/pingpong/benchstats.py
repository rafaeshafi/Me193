"""Pure analysis behind env_check and the bench tools (no hardware here)."""

import numpy as np

_GAP_BINS_MS = ((0, 20, "<20 ms"), (20, 40, "20-40 ms"), (40, 100, "40-100 ms"),
                (100, 250, "100-250 ms"), (250, float("inf"), ">=250 ms"))


def rate_stats(t_ns):
    """Arrival statistics of a notification stream (timestamps in ns)."""
    n = len(t_ns)
    out = {"n": n, "hz": 0.0, "mean_gap_ms": 0.0, "worst_gap_ms": 0.0, "p99_gap_ms": 0.0,
           "p999_gap_ms": 0.0, "gaps_over_100ms": 0, "histogram": {}}
    if n < 2:
        return out
    gaps = np.diff(np.asarray(t_ns, dtype=np.int64)) / 1e6
    span_s = (t_ns[-1] - t_ns[0]) / 1e9
    out.update(
        hz=(n - 1) / span_s if span_s > 0 else 0.0,
        mean_gap_ms=float(gaps.mean()),
        worst_gap_ms=float(gaps.max()),
        p99_gap_ms=float(np.percentile(gaps, 99)),
        p999_gap_ms=float(np.percentile(gaps, 99.9)),
        gaps_over_100ms=int((gaps > 100).sum()),
        histogram={label: int(((gaps >= lo) & (gaps < hi)).sum()) for lo, hi, label in _GAP_BINS_MS},
    )
    return out


def rate_verdict(hz):
    """The plan's go/no-go on hub IMU rate: >=40 GO, 25-40 WARN, <25 NO-GO."""
    if hz >= 40.0:
        return "GO"
    if hz >= 25.0:
        return "WARN"
    return "NO-GO"


def accel_scale_from_faces(faces):
    """Counts per g from six rest faces; every face should read |a| = 1 g."""
    norms = np.array([np.linalg.norm(f) for f in faces], dtype=float)
    mean = float(norms.mean())
    cv = float(norms.std() / mean) if mean else float("inf")
    return {"accel_per_g": mean, "cv": cv, "ok": cv <= 0.03}


def gyro_scale_from_turn(samples, true_degrees):
    """Counts per (deg/s) from a known rotation: integral of raw rate / angle."""
    t = np.array([s[0] for s in samples], dtype=float)
    raw = np.array([s[1] for s in samples], dtype=float)
    integral = float(np.sum((raw[1:] + raw[:-1]) * 0.5 * np.diff(t)))
    return abs(integral) / true_degrees


def xcorr_lag_s(a, b, fs, max_lag_s=0.5):
    """Seconds by which signal b lags signal a (positive = b is later)."""
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    max_k = int(max_lag_s * fs)
    best_k, best = 0, -np.inf
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            c = float(np.dot(a[: len(a) - k], b[k:]))
        else:
            c = float(np.dot(a[-k:], b[: len(b) + k]))
        if c > best:
            best, best_k = c, k
    return best_k / fs
