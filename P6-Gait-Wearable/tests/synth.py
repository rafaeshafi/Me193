"""Idealised thigh IMU signals for tests: (ax, ay, az, gx, gy, gz) in g and deg/s, X along the thigh."""
import math


def synth(kind, seconds=2.0, rate=100, phase=0.0):
    out = []
    for i in range(int(seconds * rate)):
        t = i / rate + phase
        if kind == "STAND":
            out.append((1.0, 0.02 * math.sin(7 * t), 0.0, 1.0, -1.0, 0.5))
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
