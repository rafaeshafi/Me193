"""Idealised thigh IMU signals for tests: (ax, ay, az, gx, gy, gz) in g and deg/s, X along the thigh."""
import math


def synth(kind, seconds=2.0, rate=100, phase=0.0):
    out = []
    for i in range(int(seconds * rate)):
        t = i / rate + phase
        if kind == "STAND":
            out.append((1.0 + 0.005 * math.sin(37 * t), 0.02 * math.sin(7 * t), 0.003 * math.sin(11 * t),
                        1.0 + 0.3 * math.sin(23 * t), -1.0, 0.5))  # ~5 mg accel noise, ~1 deg/s gyro bias plus a little jitter
        elif kind == "WALK":   # 1.8 steps/s, heel strikes ~1.6 g, thigh pitching at half the step rate
            out.append((1.0 + 0.6 * math.sin(2 * math.pi * 1.8 * t), 0.1, 0.05,
                        150 * math.sin(2 * math.pi * 0.9 * t), 10.0, 5.0))
        elif kind == "RUN":    # 2.8 steps/s, sharp 3.5 g peaks
            s = math.sin(2 * math.pi * 2.8 * t)
            out.append((1.0 + 2.5 * (s if s > 0 else 0.0), 0.3 * s, 0.1,
                        350 * math.sin(2 * math.pi * 1.4 * t), 20.0, 10.0))
        elif kind == "BIKE":   # lying back: gravity on Y, smooth 1.2 Hz pedalling, big gyro
            out.append((0.3 * math.sin(2 * math.pi * 1.2 * t), 0.95, 0.1,
                        200 * math.sin(2 * math.pi * 1.2 * t), 15.0, 5.0))
        else:
            raise ValueError(kind)
    return out


def steps_take(n_steps, cadence_hz=1.8, rate=100, still_s=1.0, kind="WALK"):
    """A take with exactly n_steps steps at cadence_hz (one accel peak per step, thigh gyro at half the step rate),
    `still_s` of standing before and after. kind STAND/BIKE gives a 0-step take of the given length in steps/cadence."""
    out = []
    moving_s = n_steps / cadence_hz
    if kind in ("STAND", "BIKE"):
        return synth(kind, seconds=moving_s + 2 * still_s, rate=rate)
    out += synth("STAND", seconds=still_s, rate=rate)
    for i in range(int(round(moving_s * rate))):
        t = i / rate
        amp = 0.6 if kind == "WALK" else 2.5
        s = math.sin(2 * math.pi * cadence_hz * t)
        out.append((1.0 + amp * (s if kind == "RUN" and s > 0 else s if kind == "WALK" else 0.0), 0.1, 0.05,
                    150 * math.sin(2 * math.pi * cadence_hz / 2 * t), 10.0, 5.0))
    out += synth("STAND", seconds=still_s, rate=rate)
    return out
