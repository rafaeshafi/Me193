"""StepNet on the board: the step-rate net in pure Python, and the streaming counter that integrates it.

Input is the 100 Hz six-channel stream (ax ay az in g, gx gy gz in deg/s) decimated by 4 (block average) to 25 Hz and
standardised per channel with the mean/std stored in the weights file. The net is a stack of causal dilated 1-D
convolutions (left zero padding (kernel-1)*dilation, so an output at t sees only inputs at or before t) with ReLU,
and a kernel-1 Softplus head giving a non-negative step rate in steps per second. count = sum(rate) * DT.

Streaming: StepCounter keeps the last `pad` decimated samples (pad = total left padding of the stack) and runs the
whole net over context + new samples, keeping only the new outputs. That equals an offline pass over the full history
exactly, because every output depends on at most pad+1 inputs. Weights: models/stepnet.json, written by
train/stepnet_torch.py; train/steps.py imports `decimate` from here so training and the board decimate identically.
"""
import json
import math
from collections import deque

CHANNELS = 6
DECIMATE = 4
RATE_HZ = 25
DT = 1.0 / RATE_HZ
ACTS = ("relu", "softplus")
WAVE_S = 4.0


def decimate(samples, remainder=()):
    """Average blocks of DECIMATE consecutive samples. -> (decimated 6-tuples, leftover samples (< DECIMATE))."""
    buf = list(remainder) + list(samples)
    n = len(buf) // DECIMATE
    out = []
    for i in range(n):
        blk = buf[i * DECIMATE:(i + 1) * DECIMATE]
        out.append(tuple(sum(s[c] for s in blk) / DECIMATE for c in range(CHANNELS)))
    return out, buf[n * DECIMATE:]


def _softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def conv_causal(x, w, b, dilation, act):
    """x: [cin][T] -> [cout][T]. w: [cout][cin][K] with w[..][K-1] the newest tap; left zero padding (K-1)*dilation."""
    cin, T = len(x), len(x[0]) if x else 0
    K = len(w[0][0])
    pad = (K - 1) * dilation
    xp = [[0.0] * pad + list(xc) for xc in x]
    taps = [k * dilation for k in range(K)]
    out = []
    for o in range(len(w)):
        wo, bo, row = w[o], b[o], []
        for t in range(T):
            acc = bo
            for c in range(cin):
                xc, wc = xp[c], wo[c]
                for k in range(K):
                    acc += wc[k] * xc[t + taps[k]]
            if act == "relu":
                row.append(acc if acc > 0.0 else 0.0)
            else:
                row.append(_softplus(acc))
        out.append(row)
    return out


class StepNet:
    def __init__(self, spec):
        try:
            self.mean, self.std = [float(v) for v in spec["mean"]], [float(v) for v in spec["std"]]
            self.layers = [(l["w"], l["b"], int(l["dilation"]), l["act"]) for l in spec["layers"]]
            self.meta = spec.get("meta", {}) or {}
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("bad StepNet spec: %s" % e)
        if len(self.mean) != CHANNELS or len(self.std) != CHANNELS or any(s <= 0 for s in self.std):
            raise ValueError("mean/std must have %d entries and std > 0" % CHANNELS)
        if int(spec.get("decimate", DECIMATE)) != DECIMATE or abs(float(spec.get("dt", DT)) - DT) > 1e-9:
            raise ValueError("spec decimation does not match the board (%d, %.3f)" % (DECIMATE, DT))
        cin = CHANNELS
        self.pad = 0
        for w, b, d, act in self.layers:
            if act not in ACTS or d < 1 or not w or any(len(wo) != cin for wo in w) or len(b) != len(w):
                raise ValueError("layer shapes do not chain (%d in)" % cin)
            K = len(w[0][0])
            if any(len(wc) != K for wo in w for wc in wo):
                raise ValueError("ragged kernel")
            self.pad += (K - 1) * d
            cin = len(w)
        if cin != 1 or self.layers[-1][3] != "softplus":
            raise ValueError("the head must be one softplus channel")
        self.pm_pct95 = float((self.meta.get("loro") or {}).get("pm_pct95", 100.0))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def forward(self, x):
        """x: [T] decimated raw 6-tuples -> [T] step rates (steps/s); the history before x[0] is taken as silence."""
        if not x:
            return []
        h = [[(s[c] - self.mean[c]) / self.std[c] for s in x] for c in range(CHANNELS)]
        for w, b, d, act in self.layers:
            h = conv_causal(h, w, b, d, act)
        return h[0]


class StepCounter:
    """Feeds 100 Hz samples to a StepNet in hops and integrates the rate. on_samples() and process() are called under
    the owner's lock (Gait); the snapshot is plain values."""

    def __init__(self, net):
        self.net = net
        self._pending, self._rem, self._ctx = [], [], []
        self.count, self.rate = 0.0, 0.0
        self.wave = deque(maxlen=int(WAVE_S * RATE_HZ))
        self.wave_all = []  # every rate ever produced (tests compare it with the offline pass); bounded by reset()

    def on_samples(self, samples):
        self._pending.extend(samples)

    def process(self):
        """Run the net over the pending samples. Returns the new rates (possibly empty)."""
        if not self._pending:
            return []
        dec, self._rem = decimate(self._pending, self._rem)
        self._pending = []
        if not dec:
            return []
        rates = self.net.forward(self._ctx + dec)[len(self._ctx):]
        self._ctx = (self._ctx + dec)[-self.net.pad:] if self.net.pad else []
        self.count += sum(rates) * DT
        self.rate = rates[-1]
        self.wave.extend(rates)
        if len(self.wave_all) < 100000:
            self.wave_all.extend(rates)
        return rates

    def reset_context(self):
        """The sensor stream restarted: the next samples are not contiguous with the past. The count stays."""
        self._pending, self._rem, self._ctx = [], [], []

    def reset(self):
        self.count, self.rate = 0.0, 0.0
        self.wave.clear()
        self.wave_all = []

    def snapshot(self):
        n = int(self.count)
        pm = max(1, int(math.ceil(n * self.pm_pct95 / 100.0))) if n >= 1 else 0
        return {"count": n, "exact": round(self.count, 2), "rate": round(self.rate, 3), "spm": int(round(self.rate * 60)),
                "pm": pm, "pm_pct": self.pm_pct95, "wave": [round(r, 3) for r in self.wave]}

    @property
    def pm_pct95(self):
        return self.net.pm_pct95
