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


def _smooth(x, n=3):
    return np.convolve(x, np.ones(n) / n, mode="same") if len(x) >= n else x


def wave_lag_s(pose_t_ns, pose_uv, imu_t_ns, imu_g, *, fs=100.0, max_lag_s=0.35):
    """Camera-vs-IMU lag from a waving take -> (lag_s, quality).

    The wrist's speed in the camera and the gyro's magnitude both peak when the hand is moving
    fastest; the camera reports each peak later than the IMU does.  lag_s is how much later (the
    constant CAMERA_LAG_S subtracts from camera stamps), searched only over 0..max_lag_s because the
    camera cannot see the future.  quality is the correlation at that lag (0..1): below ~0.35 the
    wave was too small or too regular to trust the number.
    """
    pt, it = np.asarray(pose_t_ns, dtype=float) / 1e9, np.asarray(imu_t_ns, dtype=float) / 1e9
    uv, g = np.asarray(pose_uv, dtype=float), np.asarray(imu_g, dtype=float)
    if len(pt) < 30 or len(it) < 60:
        raise ValueError("not enough samples: wave the hub for the whole window")
    t0, t1 = max(pt[0], it[0]), min(pt[-1], it[-1]) - max_lag_s
    if t1 - t0 < 3.0:
        raise ValueError("the pose and IMU streams overlap for under 3 s: wave for the whole window")
    pose_speed = _smooth(np.hypot(np.gradient(uv[:, 0], pt), np.gradient(uv[:, 1], pt)))
    gyro_mag = _smooth(np.linalg.norm(g, axis=1), 5)
    grid = np.arange(t0, t1, 1.0 / fs)
    reference = np.interp(grid, it, gyro_mag)
    best_lag, best_r = 0.0, -2.0
    for lag in np.arange(0.0, max_lag_s + 1e-9, 1.0 / fs):
        shifted = np.interp(grid + lag, pt, pose_speed)
        if reference.std() < 1e-9 or shifted.std() < 1e-9:
            continue
        r = float(np.corrcoef(reference, shifted)[0, 1])
        if r > best_r:
            best_r, best_lag = r, float(lag)
    return best_lag, max(0.0, best_r)
