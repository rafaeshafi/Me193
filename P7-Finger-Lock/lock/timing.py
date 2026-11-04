"""Per-stage timings (capture, preprocess, inference, loop) kept the same way on the Mac and the board, so the
'UNO Q vs computer' table comes from one function."""
import platform
import time


class Stats:
    def __init__(self, keep=2000):
        self.keep, self.samples, self.t_start = keep, {}, time.monotonic()

    def add(self, name, ms):
        s = self.samples.setdefault(name, [])
        s.append(float(ms))
        if len(s) > self.keep:
            del s[: len(s) - self.keep]

    @staticmethod
    def _pct(values, q):
        if not values:
            return None
        v = sorted(values)
        return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]

    def summary(self):
        out = {}
        for name, s in self.samples.items():
            out[name] = {"n": len(s), "mean_ms": round(sum(s) / len(s), 2) if s else None,
                         "p50_ms": round(self._pct(s, 0.5), 2) if s else None, "p95_ms": round(self._pct(s, 0.95), 2) if s else None,
                         "max_ms": round(max(s), 2) if s else None}
        loop = self.samples.get("loop")
        out["fps"] = round(1000.0 / (sum(loop) / len(loop)), 1) if loop else None
        out["machine"] = {"node": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
                          "system": platform.system()}
        return out


class Timer:
    """with Timer(stats, "inference"): ..."""

    def __init__(self, stats, name):
        self.stats, self.name = stats, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.stats.add(self.name, (time.perf_counter() - self.t0) * 1000.0)
