# Step Counter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The P6 gait wearable counts steps live with a small neural net ("StepNet") trained in PyTorch on all six IMU channels, shows the count with an uncertainty on a new Steps tab, and ships with an uncertainty analysis, a Manim explainer animation and write-up drafts.

**Architecture:** The board (stdlib Python) decimates the 100 Hz stream to 25 Hz, standardises it and runs a causal dilated 1-D conv net from `models/stepnet.json`; the step rate it outputs is integrated into the count. The recorder gains "counted takes" whose true count travels in the file name. On the Mac, `train/steps.py` builds the dataset from those names, trains the same net in PyTorch with count-preserving augmentation, evaluates it leave-one-recording-out, exports JSON and figures. Spec: `docs/superpowers/specs/2026-11-01-step-counter-design.md`.

**Tech Stack:** Python 3.12 stdlib on the board (Arduino UNO Q, App Lab); PyTorch + numpy + matplotlib in `.venv` on the Mac; pytest; Manim Community for the animation; adb for deploy.

**Conventions (read first):**
- Work in `~/ME193/P6-Gait-Wearable` (git root is `~/ME193`). Commit after every task, message prefixed `P6-Gait-Wearable:`, **no Co-Authored-By trailer**. Never push.
- Tests: `.venv/bin/python -m pytest tests -q` (57 pass today). Train deps: `.venv/bin/pip install -r train/requirements.txt`.
- Board code in `app/python` is stdlib only. Keep every file under 500 lines. Read a file before editing it.
- `data/` is git-ignored and holds the real recordings; never commit it.
- Tasks 6 and 11 need the user at the bench (board on the USB cable, phone on the hotspot). Everything else runs on the Mac.

**File map**

| File | Status | Responsibility |
|---|---|---|
| `app/python/stepnet.py` | create | decimation, normalisation, causal conv forward, `StepNet` (load/forward), `StepCounter` (streaming, integral, reset, snapshot) |
| `models/stepnet.json` | create (seed), later overwritten by training | weights + normalisation + meta |
| `app/python/recorder.py` | modify | counted takes, `parse_counted`, `kind`/`elapsed_s` in status |
| `app/python/web.py` | modify | `/steps`, `/record?steps=1`, `/stop?steps=N`, `/steps/reset` |
| `app/python/pipeline.py` | modify | `stepnet=` argument, `Gait.steps`, snapshot `steps` |
| `app/python/main.py` | modify | load `stepnet.json`, tolerate its absence |
| `app/python/page.html` | modify | Steps tab |
| `scripts/deploy.sh` | modify | ship `stepnet.json` |
| `train/stepnet_torch.py` | create | `StepNetTorch` + JSON export |
| `train/steps.py` | create | dataset, augmentation, fit, LORO, seeds, metrics, report, export |
| `train/steps_plots.py` | create | the four figures |
| `train/requirements.txt` | modify | add torch |
| `tests/seed_models.py`, `tests/synth.py` | modify | `seed_stepnet()`, `steps_take()` |
| `tests/test_stepnet.py`, `tests/test_steps_train.py` | create | see tasks |
| `tests/test_recorder.py`, `tests/test_web.py`, `tests/test_pipeline.py` | modify | new cases |
| `docs/animation/stepnet_explainer.py`, `render.sh`, `requirements.txt` | create | Manim explainer |
| `docs/steps_results.md`, `docs/figures/steps_*.png` | generated | results |
| `README.md`, `docs/notion-steps.md` | modify / create | write-up |

---

### Task 1: StepNet forward pass and streaming counter (`app/python/stepnet.py`)

**Files:**
- Create: `app/python/stepnet.py`
- Modify: `tests/seed_models.py` (append `seed_stepnet`)
- Create: `tests/test_stepnet.py`

- [ ] **Step 1: Add the seed weight generator to `tests/seed_models.py`**

Add `import random` at the top of `tests/seed_models.py` (after the docstring) and append to the end:

```python
def seed_stepnet(seed=0, channels=12, kernel=5, dilations=(1, 2, 4), pm_pct95=100.0):
    """A StepNet spec with small deterministic random weights: the right shape, no skill (tests and the first deploy)."""
    rng = random.Random(seed)
    layers, cin = [], 6
    for d in dilations:
        layers.append({"w": [[[rng.uniform(-0.3, 0.3) for _ in range(kernel)] for _ in range(cin)] for _ in range(channels)],
                       "b": [rng.uniform(-0.1, 0.1) for _ in range(channels)], "dilation": d, "act": "relu"})
        cin = channels
    layers.append({"w": [[[rng.uniform(-0.3, 0.3)] for _ in range(cin)]], "b": [-1.0], "dilation": 1, "act": "softplus"})
    return {"rate_hz": 25, "decimate": 4, "dt": 0.04, "kernel": kernel,
            "mean": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "std": [0.5, 0.3, 0.3, 100.0, 50.0, 50.0],
            "layers": layers, "meta": {"trained": None, "recordings": 0, "loro": {"pm_pct95": pm_pct95}}}
```

- [ ] **Step 2: Write the failing tests `tests/test_stepnet.py`**

```python
import json
import math
import random

import pytest

from seed_models import seed_stepnet
from stepnet import DECIMATE, DT, StepCounter, StepNet, conv_causal, decimate


def rnd_samples(n, seed=1):
    rng = random.Random(seed)
    return [tuple(rng.uniform(-2, 2) for _ in range(6)) for _ in range(n)]


def test_decimate_averages_blocks_and_keeps_the_remainder():
    s = [(float(i), 0, 0, 0, 0, 0) for i in range(10)]
    dec, rem = decimate(s)
    assert [d[0] for d in dec] == [1.5, 5.5] and [r[0] for r in rem] == [8.0, 9.0]
    dec2, rem2 = decimate([(10.0, 0, 0, 0, 0, 0), (11.0, 0, 0, 0, 0, 0), (12.0, 0, 0, 0, 0, 0)], rem)
    assert [d[0] for d in dec2] == [9.5] and [r[0] for r in rem2] == [12.0]
    assert DECIMATE == 4 and DT == 0.04


def test_conv_causal_matches_a_hand_computed_case():
    # one input channel [1, 2, 3], one output, kernel 2, dilation 1, w = [0.5 (older), 1.0 (newest)], b = 0.25
    out = conv_causal([[1.0, 2.0, 3.0]], [[[0.5, 1.0]]], [0.25], 1, "relu")
    assert out == [[pytest.approx(1.25), pytest.approx(2.75), pytest.approx(4.25)]]
    # dilation 2: the older tap looks two samples back, zero before the start
    out = conv_causal([[1.0, 2.0, 3.0]], [[[0.5, 1.0]]], [0.0], 2, "relu")
    assert out == [[pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.5)]]
    # softplus head and a negative pre-activation
    out = conv_causal([[-4.0]], [[[1.0]]], [0.0], 1, "softplus")
    assert out[0][0] == pytest.approx(math.log1p(math.exp(-4.0)))


def test_forward_is_nonnegative_and_streaming_equals_offline():
    net = StepNet(seed_stepnet())
    x = [tuple(v / 10 for v in s) for s in rnd_samples(300)]
    offline = net.forward(x)
    assert len(offline) == 300 and min(offline) >= 0.0 and net.pad == 28
    rng = random.Random(3)
    ctr = StepCounter(net)
    raw = [s for s in x for _ in range(DECIMATE)]      # decimating this gives back exactly x
    i, got = 0, []
    while i < len(raw):
        n = rng.randint(1, 37)                         # batches of random size, like the real stream
        ctr.on_samples(raw[i:i + n])
        got += ctr.process()
        i += n
    assert len(got) == len(offline)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, offline))
    assert ctr.count == pytest.approx(sum(offline) * DT)


def test_reset_keeps_or_clears_the_right_things():
    ctr = StepCounter(StepNet(seed_stepnet()))
    ctr.on_samples(rnd_samples(400))
    ctr.process()
    assert ctr.count > 0 and len(ctr._ctx) == 28
    c = ctr.count
    ctr.on_samples(rnd_samples(10, seed=7))   # pending at the restart: still the old stream, so it is processed first
    ctr.reset_context()                 # sensor stream restarted: forget the past, keep the count
    assert ctr.count > c and ctr._ctx == [] and ctr._rem == [] and ctr._pending == []
    c = ctr.count
    fresh = rnd_samples(40, seed=9)
    ctr.on_samples(fresh)
    new = ctr.process()
    want = ctr.net.forward(decimate(fresh)[0])
    assert len(new) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(new, want))
    assert ctr.count == pytest.approx(c + sum(want) * DT)
    ctr.reset()                         # the wearer tapped Reset
    assert ctr.count == 0.0 and ctr.rate == 0.0 and ctr.snapshot()["wave"] == []


def test_snapshot_has_the_live_fields_and_the_half_width():
    ctr = StepCounter(StepNet(seed_stepnet(pm_pct95=10.0)))
    s = ctr.snapshot()
    assert s == {"count": 0, "exact": 0.0, "rate": 0.0, "spm": 0, "pm": 0, "pm_pct": 10.0, "trained": False, "wave": []}
    ctr.count = 38.4
    s = ctr.snapshot()
    assert s["count"] == 38 and s["pm"] == 4          # ceil(38 * 10 / 100)
    ctr.count = 1.2
    assert ctr.snapshot()["pm"] == 1                  # never less than 1 once counting
    ctr.on_samples(rnd_samples(400))
    ctr.process()
    assert ctr.snapshot()["spm"] == round(60 * sum(list(ctr.wave)[-25:]) / 25)


def _bad(mut):
    spec = seed_stepnet()
    mut(spec)
    return spec


@pytest.mark.parametrize("mut", [
    lambda s: s["layers"][1].update(w=s["layers"][1]["w"][:-1]),   # wrong number of input channels
    lambda s: s["std"].__setitem__(0, 0.0),
    lambda s: s["std"].__setitem__(0, "nan"),
    lambda s: s["layers"][0].update(w=[[[] for _ in oc] for oc in s["layers"][0]["w"]]),   # kernel []
    lambda s: s.update(decimate=2),
    lambda s: s["layers"][-1].update(act="relu"),
    lambda s: s.pop("layers"),
], ids=["channels", "std0", "stdnan", "kernel0", "decimate2", "headrelu", "nolayers"])
def test_load_validates(tmp_path, mut):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(seed_stepnet()))
    assert StepNet.load(p).pm_pct95 == 100.0
    p.write_text(json.dumps(_bad(mut)))
    with pytest.raises(ValueError):
        StepNet.load(p)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_stepnet.py -q`
Expected: `ModuleNotFoundError: No module named 'stepnet'`

- [ ] **Step 4: Write `app/python/stepnet.py`**

```python
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
            self.layers = [([[[float(v) for v in wc] for wc in wo] for wo in l["w"]], [float(v) for v in l["b"]],
                            int(l["dilation"]), l["act"]) for l in spec["layers"]]
            self.meta = spec.get("meta", {}) or {}
            dec, dt = int(spec.get("decimate", DECIMATE)), float(spec.get("dt", DT))
            self.pm_pct95 = float((self.meta.get("loro") or {}).get("pm_pct95", 100.0))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ValueError("bad StepNet spec: %s" % e)
        if len(self.mean) != CHANNELS or len(self.std) != CHANNELS or any(not (s > 0) or not math.isfinite(s) for s in self.std):
            raise ValueError("mean/std must have %d entries and finite std > 0" % CHANNELS)
        if dec != DECIMATE or abs(dt - DT) > 1e-9:
            raise ValueError("spec decimation does not match the board (%d, %.3f)" % (DECIMATE, DT))
        cin = CHANNELS
        self.pad = 0
        for w, b, d, act in self.layers:
            if act not in ACTS or d < 1 or not w or any(len(wo) != cin for wo in w) or len(b) != len(w):
                raise ValueError("layer shapes do not chain (%d in)" % cin)
            K = len(w[0][0])
            if K < 1 or any(len(wc) != K for wo in w for wc in wo):
                raise ValueError("empty or ragged kernel")
            self.pad += (K - 1) * d
            cin = len(w)
        if cin != 1 or self.layers[-1][3] != "softplus":
            raise ValueError("the head must be one softplus channel")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def forward(self, x):
        """x: [T] decimated raw 6-tuples -> [T] step rates (steps/s); the zero padding comes after standardising, so the
        history before x[0] is taken as the mean posture, not as zero signal."""
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
        seq = self._ctx + dec
        rates = self.net.forward(seq)[len(self._ctx):]
        self._ctx = seq[-self.net.pad:] if self.net.pad else []
        self.count += sum(rates) * DT
        self.rate = rates[-1]
        self.wave.extend(rates)
        return rates

    def reset_context(self):
        """The sensor stream restarted: the next samples are not contiguous with the past. Pending samples still belong
        to the old stream, so they are processed first; then the leftover and the context are dropped. The count stays."""
        self.process()
        self._rem, self._ctx = [], []

    def reset(self):
        self.count, self.rate = 0.0, 0.0
        self.wave.clear()

    def snapshot(self):
        n = int(self.count)
        pm = max(1, int(math.ceil(n * self.pm_pct95 / 100.0))) if n >= 1 else 0
        last = list(self.wave)[-RATE_HZ:]  # the newest second, so the steps-per-minute figure is steady
        spm = int(round(60 * sum(last) / len(last))) if last else 0
        return {"count": n, "exact": round(self.count, 2), "rate": round(self.rate, 3), "spm": spm,
                "pm": pm, "pm_pct": self.pm_pct95, "trained": bool(self.net.meta.get("trained")),
                "wave": [round(r, 3) for r in self.wave]}

    @property
    def pm_pct95(self):
        return self.net.pm_pct95
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_stepnet.py -q`
Expected: `6 passed`

- [ ] **Step 6: Time the pure-Python cost on the Mac (sanity, the UNO Q is ~10× slower)**

Run:
```bash
cd ~/ME193/P6-Gait-Wearable && .venv/bin/python -c "
import sys, time, random; sys.path[:0]=['app/python','tests']
from seed_models import seed_stepnet; from stepnet import StepNet, StepCounter
c=StepCounter(StepNet(seed_stepnet())); rng=random.Random(0)
c.on_samples([tuple(rng.random() for _ in range(6)) for _ in range(50)]); t=time.perf_counter(); c.process(); print('one 0.5 s hop: %.1f ms' % ((time.perf_counter()-t)*1000))"
```
Expected: a few ms (so tens of ms on the board, far under the 500 ms hop). If it is above 40 ms on the Mac, stop and report.

- [ ] **Step 7: Commit**

```bash
git add app/python/stepnet.py tests/test_stepnet.py tests/seed_models.py
git commit -m "P6-Gait-Wearable: StepNet forward pass and streaming StepCounter in pure Python (decimate 100->25 Hz, causal dilated convs, exact streaming)"
```

---

### Task 2: Seed weights file and deploy script

**Files:**
- Create: `models/stepnet.json` (generated)
- Modify: `scripts/deploy.sh:21`

- [ ] **Step 1: Generate the seed file**

```bash
cd ~/ME193/P6-Gait-Wearable && .venv/bin/python -c "
import sys, json; sys.path.insert(0, 'tests'); from seed_models import seed_stepnet
json.dump(seed_stepnet(), open('models/stepnet.json', 'w'))" && .venv/bin/python -c "
import sys; sys.path.insert(0, 'app/python'); from stepnet import StepNet; n = StepNet.load('models/stepnet.json'); print('ok pad', n.pad, 'pm', n.pm_pct95)"
```
Expected: `ok pad 28 pm 100.0`

- [ ] **Step 2: Ship it in `scripts/deploy.sh`**

Change the line
```bash
mkdir -p "$STAGE/models" && cp models/model1.json models/model2.json "$STAGE/models/"
```
to
```bash
mkdir -p "$STAGE/models" && cp models/model1.json models/model2.json models/stepnet.json "$STAGE/models/"
```

- [ ] **Step 3: Commit**

```bash
git add models/stepnet.json scripts/deploy.sh
git commit -m "P6-Gait-Wearable: seed stepnet.json (random weights, pm 100 %) and deploy ships it"
```

---

### Task 3: Counted takes in the recorder

**Files:**
- Modify: `app/python/recorder.py`
- Modify: `tests/test_recorder.py` (append)

- [ ] **Step 1: Append the failing tests to `tests/test_recorder.py`**

```python
from recorder import STEP_ACTIVITIES, parse_counted   # add to the existing `from recorder import ...` line at the top


def test_parse_counted():
    assert parse_counted("20261101-101500-steps-walk-n42-Rafae.csv") == ("WALK", 42, "Rafae")
    assert parse_counted("20261101-101500-steps-bike-n0.csv") == ("BIKE", 0, "")
    assert parse_counted("20261101-101500-steps-walk-n42-Rafae-2.csv") == ("WALK", 42, "Rafae")
    assert parse_counted("20261101-101500-steps-walk-Rafae.csv") is None      # stopped without a count
    assert parse_counted("20261026-135026-guided-Rafae.csv") is None
    assert parse_counted("20261101-101500-steps-walk-42.csv") is None          # no n-marker: a numeric who, not a count
    assert parse_counted("20261101-101500-steps-walk-5-Bob.csv") is None
    assert parse_counted("20261101-101500-steps-hop-n5.csv") is None            # not an activity
    assert STEP_ACTIVITIES == ("WALK", "RUN", "STAND", "BIKE", "MIXED")


def test_counted_take_is_renamed_with_its_count(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    r.start_counted("walk", who="Rafae")
    assert r.active and r.path.name.endswith("-steps-walk-Rafae.csv")
    st = r.status()
    assert st["kind"] == "counted" and st["elapsed_s"] == 0.0 and st["label"] == "WALK" and st["prompt"].startswith("WALK")
    clk.t += 12.3
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)
    assert r.status()["elapsed_s"] == 12.3
    rows = read(r.path)
    assert [row[1] for row in rows[1:]] == ["WALK"] * 10          # no get-ready gap: every row carries the activity
    assert r.stop(steps=42) == r.path.name
    assert not r.active
    assert r.path.name.endswith("-steps-walk-n42-Rafae.csv") and r.path.exists()
    assert r.sessions() == [r.path.name]
    assert r.status()["kind"] is None and r.status()["elapsed_s"] is None


def test_counted_take_without_a_count_keeps_its_name(tmp_path):
    r = Recorder(tmp_path, clock=Clock(), rolling=False)
    r.start_counted("MIXED")
    name = r.path.name
    assert r.stop() == name
    assert r.path.name == name and name.endswith("-steps-mixed.csv") and parse_counted(name) is None


def test_steps_are_ignored_for_other_kinds_and_bad_activities_rejected(tmp_path):
    r = Recorder(tmp_path, clock=Clock(), rolling=False)
    r.start_single("WALK", seconds=10)
    name = r.path.name
    assert r.stop(steps=7) == name
    assert r.path.name == name
    assert r.stop() is None                                         # idle
    with pytest.raises(ValueError):
        r.start_counted("HOP")


def test_rename_never_overwrites_and_negative_count_is_rejected(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    names = []
    for _ in range(3):                      # same second, same count: -2, -3 suffixes, nothing overwritten
        r.start_counted("walk")
        names.append(r.stop(steps=5))
    assert len(set(names)) == 3 and all(parse_counted(n)[:2] == ("WALK", 5) for n in names)
    assert len(list(tmp_path.glob("*.csv"))) == 3
    r.start_counted("walk")
    with pytest.raises(ValueError):
        r.stop(steps=-1)
    assert r.active                          # a rejected stop leaves the take open
```

Add `import pytest` to the top of `tests/test_recorder.py` if it is not there, and merge the `STEP_ACTIVITIES, parse_counted` import into the file's existing `from recorder import ...` line.

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_recorder.py -q`
Expected: `ImportError: cannot import name 'STEP_ACTIVITIES'`

- [ ] **Step 3: Extend `app/python/recorder.py`**

Add after `ROLLING_KEEP = 2`:
```python
STEP_ACTIVITIES = ("WALK", "RUN", "STAND", "BIKE", "MIXED")   # counted takes (step counter training)
COUNTED_CAP_S = 3600
COUNTED_RE = re.compile(r"^\d{8}-\d{6}-steps-(walk|run|stand|bike|mixed)-n(\d+)(?:-([A-Za-z0-9_-]+?))?(?:-2)?\.csv$")
```

Add after `clean()`:
```python
def parse_counted(name):
    """'<stamp>-steps-<activity>-n<steps>[-<who>][-2].csv' -> (ACTIVITY, steps, who) or None for any other file."""
    m = COUNTED_RE.match(name)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3) or ""
```

In `__init__`, after `self.who = ""` add:
```python
        self.kind = None      # "guided" | "single" | "counted" while a take is open
        self._stamp = ""
        self._activity = None  # lower-case activity of an open counted take
```

Add a method after `start_single`:
```python
    def start_counted(self, activity, who=""):
        """An open-ended take of one activity whose step count the wearer types at stop()."""
        activity = activity.upper()
        if activity not in STEP_ACTIVITIES:
            raise ValueError("activity must be one of %s" % ", ".join(STEP_ACTIVITIES))
        self._start([(activity, COUNTED_CAP_S)], who, "steps-" + activity.lower(), kind="counted", activity=activity.lower())
```

Change `_start`'s signature to
```python
    def _start(self, steps, who, name, kind="single", activity=None):
```
and inside it replace `stamp = time.strftime(...)` / `base = ...` with
```python
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._stamp, self.kind, self._activity = stamp, kind, activity
            base = "%s-%s%s" % (stamp, name, "-" + self.who if self.who else "")
```
Update the two existing callers: `start_guided` → `self._start(steps, who, "guided", kind="guided")`; `start_single` → `self._start([("", GAP_S), (label, seconds)], who, label.lower())`.

In `_close_session` add `self.kind = None` after `self._writer = None` (outside the `if`, so it always clears).

Replace `stop` (it returns the final file name, or None when idle; a negative count raises ValueError):
```python
    def stop(self, steps=None):
        """Stop the open take and return its final file name, or None if none was open. For a counted take, `steps`
        (int >= 0) is the wearer's count and goes into the file name; it is ignored for other takes."""
        if steps is not None and int(steps) < 0:
            raise ValueError("steps must be >= 0")
        with self._lock:
            if self._file is None:
                return None
            counted = self.kind == "counted"
            self._close_session()
            if counted and steps is not None:
                self._rename_counted(int(steps))
            return self.path.name

    def _rename_counted(self, steps):
        who = "-" + self.who if self.who else ""
        target = self.dir / ("%s-steps-%s-n%d%s.csv" % (self._stamp, self._activity, steps, who))
        k = 2
        while target.exists():
            target = self.dir / ("%s-steps-%s-n%d%s-%d.csv" % (self._stamp, self._activity, steps, who, k))
            k += 1
        self.path.rename(target)
        self.path = target
        self._sessions = None
```

In `status()` change the locked block and the return to include the kind and the elapsed time:
```python
        with self._lock:
            now = self._clock()
            cur = self._step(now)
            active = self._file is not None
            name = self.path.name if self.path else None
            rows, who = self.rows, self.who
            kind = self.kind if active else None
            elapsed = round(now - self._t0, 1) if active else None
        return {"active": active, "file": name, "rows": rows, "kind": kind, "elapsed_s": elapsed,
                "label": cur[0] if cur else None, "remaining_s": round(cur[1], 1) if cur else None,
                "prompt": self._prompt(cur), "who": who}
```

Update the module docstring's file list: add `<stamp>-steps-<activity>[-n<steps>][-who].csv (counted take; the count is added at stop)`.

- [ ] **Step 4: Run the recorder tests, then everything**

Run: `.venv/bin/python -m pytest tests/test_recorder.py -q` → Expected: all pass (existing + 5 new).
Run: `.venv/bin/python -m pytest tests -q` → Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/python/recorder.py tests/test_recorder.py
git commit -m "P6-Gait-Wearable: counted takes in the recorder (open-ended single activity, typed count renamed into the file name as n<steps>, parse_counted, kind/elapsed in status)"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `app/python/pipeline.py`
- Modify: `app/python/main.py`
- Modify: `tests/test_pipeline.py` (append)

- [ ] **Step 1: Append failing tests to `tests/test_pipeline.py`**

```python
from seed_models import seed_stepnet
from stepnet import StepNet


def make_steps(tmp_path, clk):
    return Gait(Tree(SEED1), Tree(SEED2), Recorder(tmp_path, clock=clk, rolling=False), clock=clk,
                stepnet=StepNet(seed_stepnet()))


def test_steps_advance_on_walking_and_reset(tmp_path):
    clk = Clock()
    g = make_steps(tmp_path, clk)
    assert g.snapshot()["steps"]["count"] == 0
    feed(g, clk, "WALK", 4.0)
    clk.t += 0.5
    g.tick()                                       # one more hop flushes the samples that arrived since the last one
    s = g.snapshot()["steps"]
    assert s["exact"] > 0 and len(s["wave"]) == 100 and s["pm_pct"] == 100.0
    if s["count"]:
        assert s["pm"] >= 1
    g.steps.reset()
    assert g.snapshot()["steps"]["count"] == 0 and g.snapshot()["steps"]["wave"] == []


def test_fifo_reset_clears_the_step_context_but_keeps_the_count(tmp_path):
    clk = Clock()
    g = make_steps(tmp_path, clk)
    feed(g, clk, "WALK", 2.0)
    g.steps.process()                              # flush what arrived since the last hop; the reset must not change the count
    before = g.steps.count
    clk.t += 0.1
    g.on_batch(to_batch(99, synth("WALK", 0.1), resets=1))
    assert g.steps._ctx == [] and len(g.steps._pending) == 10   # context gone, the new batch waits for the next hop
    assert g.steps.count == before


def test_without_a_step_model_the_snapshot_says_none(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 1.0)
    assert g.snapshot()["steps"] is None
```

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: `TypeError: Gait.__init__() got an unexpected keyword argument 'stepnet'`

- [ ] **Step 3: Edit `app/python/pipeline.py`**

Imports: add `from stepnet import StepCounter`.

Constructor signature and body:
```python
    def __init__(self, model1, model2, recorder, rate=RATE, clock=time.monotonic, races_path=None, stepnet=None):
        ...
        self.steps = StepCounter(stepnet) if stepnet is not None else None
```
(place the new line right after `self.race = ...`).

In `on_batch`, inside the `with self._lock:` block:
```python
            if resets != self._resets:  # the sensor stream restarted: older samples are not contiguous with these
                self.ring.clear()
                if self.steps:
                    self.steps.reset_context()
                self._resets = resets
            if samples:
                self._last_sample_t = now
                self.ring.push(seq, samples)
                if self.steps:
                    self.steps.on_samples(samples)
```

In `tick`, right after `self._next_tick = now + HOP_S` and before the existing `with self._lock:` block add:
```python
        if self.steps:
            with self._lock:
                self.steps.process()   # the count advances on every hop, whether or not a 2 s window is full yet
```

In `snapshot`, inside the lock also take the steps snapshot and add it to the dict:
```python
        with self._lock:
            recent = self.ring.recent(4.0)
            steps = self.steps.snapshot() if self.steps else None
        ...
        return {..., "steps": steps,
                "recorder": self.recorder.status(), "sessions": self.recorder.sessions()}
```
Update the class docstring: "Owns the ring buffer, the two trees, the votes, the recorder and the step counter."

- [ ] **Step 4: Edit `app/python/main.py`**

```python
from stepnet import StepNet
...
def load_stepnet(path):
    try:
        return StepNet.load(path)
    except (OSError, ValueError) as e:
        log.warning("no step model: %s", e)
        return None


gait = Gait(Tree.load(MODELS / "model1.json"), Tree.load(MODELS / "model2.json"), Recorder(DATA),
            stepnet=load_stepnet(MODELS / "stepnet.json"))
```
(`log` is defined before `gait` already; move `load_stepnet` between the logger and `gait`.) Extend the periodic log line with `steps=%s` using `int(gait.steps.count) if gait.steps else None`.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/python/pipeline.py app/python/main.py tests/test_pipeline.py
git commit -m "P6-Gait-Wearable: the pipeline feeds StepCounter every hop, resets its context on a FIFO reset, puts steps in the snapshot; main loads stepnet.json or runs without"
```

---

### Task 5: Web routes and the Steps tab

**Files:**
- Modify: `app/python/web.py`
- Modify: `app/python/pipeline.py` (`Gait.reset_steps()`)
- Modify: `app/python/page.html`
- Create: `app/python/race.js` (the race drawing script, moved out of `page.html` to keep it under 500 lines)
- Modify: `tests/test_web.py` (fixture + page test + append), `tests/test_pipeline.py` (append)

- [ ] **Step 1: Give the test server a step model and append failing tests to `tests/test_web.py`**

Change the fixture:
```python
from seed_models import SEED1, SEED2, seed_stepnet
from stepnet import StepNet
...
    g = Gait(Tree(SEED1), Tree(SEED2), Recorder(tmp_path, rolling=False), stepnet=StepNet(seed_stepnet()))
```
In `test_page_and_state`, the `/race` assertion also checks `b'src="/race.js"' in body`, then:
```python
    status, ctype, body = req(s, "GET", "/race.js")
    assert status == 200 and "javascript" in ctype and b"function drawRace" in body
```
Append:
```python
def test_steps_routes(server):
    s, g = server
    status, ctype, body = req(s, "GET", "/steps")
    assert status == 200 and "text/html" in ctype and b"stepCount" in body
    status, _, body = req(s, "POST", "/record?label=MIXED&steps=1&who=ann")
    rec = json.loads(body)["recorder"]
    assert status == 200 and rec["active"] and rec["kind"] == "counted" and rec["file"].endswith("-steps-mixed-ann.csv")
    assert json.loads(body)["steps"]["take"] == 0
    g.steps.count = 7.0
    assert json.loads(req(s, "GET", "/state")[2])["steps"]["take"] == 7
    status, _, body = req(s, "POST", "/stop?steps=abc")
    assert status == 400 and "steps" in json.loads(body)["error"]
    status, _, body = req(s, "POST", "/stop?steps=-1")
    assert status == 400
    status, _, body = req(s, "POST", "/stop?steps=17")
    snap = json.loads(body)
    assert status == 200 and not snap["recorder"]["active"] and any(n.endswith("-steps-mixed-n17-ann.csv") for n in snap["sessions"])
    status, _, body = req(s, "POST", "/stop?steps=5")              # nothing open: the count would be lost, so say so
    assert status == 400 and "not saved" in json.loads(body)["error"]
    status, _, body = req(s, "POST", "/record?label=HOP&steps=1")
    assert status == 400
    status, _, body = req(s, "POST", "/record?label=MIXED")          # MIXED is only a counted activity
    assert status == 400
    g.steps.count = 5.0
    status, _, body = req(s, "POST", "/steps/reset")
    assert status == 200 and json.loads(body)["steps"]["count"] == 0 and json.loads(body)["steps"]["take"] == 0
    req(s, "POST", "/record?label=WALK")                            # a plain take: a stray count must not end it
    status, _, body = req(s, "POST", "/stop?steps=3")
    assert status == 400 and "not saved" in json.loads(body)["error"] and g.recorder.active
    g.recorder.stop()
```

Append to `tests/test_pipeline.py` (the take's start count is kept on the board, so a reload or a second phone shows the same "device so far"):
```python
def test_counted_take_reports_steps_since_its_start(tmp_path):
    clk = Clock()
    g = make_steps(tmp_path, clk)
    feed(g, clk, "WALK", 2.0)
    clk.t += 0.5
    g.tick()
    start = g.steps.count
    g.start_counted("WALK", who="ann")
    assert g.snapshot()["steps"]["take"] == 0 and g.recorder.kind == "counted"
    feed(g, clk, "WALK", 2.0)
    clk.t += 0.5
    g.tick()
    s = g.snapshot()["steps"]
    assert start > 0 and 0 <= s["take"] < s["exact"] and s["take"] == round(g.steps.count - start, 2)
    g.reset_steps()
    assert g.snapshot()["steps"]["take"] == 0
    g.recorder.stop()

```

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_web.py -q`
Expected: `test_steps_routes` fails (404 on `/steps`), `test_counted_take_reports_steps_since_its_start` fails (no `Gait.start_counted`).

- [ ] **Step 3: Edit `app/python/web.py`**

Import: `from recorder import LABELS, STEP_ACTIVITIES, clean, parse_counted`.

Read the race script next to the page: `HERE = Path(__file__).resolve().parent`, `PAGE = (HERE / "page.html").read_bytes()`, `RACE_JS = (HERE / "race.js").read_bytes()`.

`do_GET`: `if url.path in ("/", "/race", "/steps"):  # one page; /race and /steps open on their tabs`, then
`elif url.path == "/race.js": self._send(200, "text/javascript; charset=utf-8", RACE_JS)`.

`do_POST` `/record` branch:
```python
            elif url.path == "/record":
                label = q.get("label", [""])[0].upper()
                if q.get("steps", ["0"])[0] == "1":
                    if label not in STEP_ACTIVITIES:
                        return self._json({"error": "activity must be one of %s" % ", ".join(STEP_ACTIVITIES)}, 400)
                    self.gait.start_counted(label, who=who)
                else:
                    if label not in LABELS:
                        return self._json({"error": "label must be one of %s" % ", ".join(LABELS)}, 400)
                    rec.start_single(label, who=who)
            elif url.path == "/stop":
                steps = q.get("steps", [None])[0]
                if steps is not None:
                    if not steps.isdigit():
                        return self._json({"error": "steps must be a whole number >= 0"}, 400)
                    steps = int(steps)
                    if rec.kind != "counted":   # a stray count must never end a plain take
                        return self._json({"error": "no counted take is open; count not saved"}, 400)
                name = rec.stop(steps=steps)
                if steps is not None and (name is None or parse_counted(name) is None):
                    # the take had already ended at its cap, or none was open: the count has nowhere to go
                    return self._json({"error": "no counted take was open; count not saved"}, 400)
            elif url.path == "/steps/reset":
                self.gait.reset_steps()
```
Docstring: add `GET /steps`, `GET /race.js` and the new POST forms.

`app/python/pipeline.py`: `self._take_start = 0.0` in `Gait.__init__`; add the two methods below (`StepCounter.reset()` must run under the Gait lock, since the Bridge thread feeds the counter under it); in `snapshot()`, under the lock, `steps["take"] = round(self.steps.count - self._take_start, 2)` when there is a step model.
```python
    def start_counted(self, activity, who=""):
        """Open a counted take and remember the count at its start, so every phone shows the steps since then."""
        with self._lock:
            self._take_start = self.steps.count if self.steps else 0.0
        self.recorder.start_counted(activity, who=who)

    def reset_steps(self):
        if self.steps:
            with self._lock:
                self.steps.reset()
                self._take_start = 0.0
```
`/stop?steps=N` refuses before stopping when the open take is not a counted one (a stray count never ends a plain take); the check after `stop()` covers a take that hit its cap in between.

- [ ] **Step 4: Edit `app/python/page.html`**

Tab bar: `<button class="tab" data-tab="gait">Gait</button><button class="tab" data-tab="steps">Steps</button><button class="tab" data-tab="race">Race</button>`.

CSS, add:
```css
  .count { font-size: 72px; font-weight: 800; line-height: 1; color: #fff; }
  .count .pm { font-size: 28px; color: #888; margin-left: 10px; font-weight: 500; }
  .take { background: #111; border-radius: 10px; padding: 12px 14px; margin: 10px 0; }
  .take button.act.on { background: #0093e7; } .take button.act { background: #333; }
  .take input { width: 110px; }
  .take .hidden { display: none; }
```
(the page's existing `section.hidden` rule does not hide the take's `div`s, and `.row` would otherwise win.)

Insert this section right after `<section id="race" ...>...</section>` (before `<section id="gait">`):
```html
<section id="steps">
<div class="count"><span id="stepCount">0</span><span id="stepPm" class="pm"></span></div>
<div class="muted">StepNet: all six IMU channels at 25 Hz, trained in PyTorch · <span id="stepAct"></span></div>
<div class="row">
  <div class="card"><small>Steps / min</small><b id="spm">--</b></div>
  <div class="card"><small>Take</small><b id="takeClock">--</b></div>
  <div class="card"><small>Model error (95 %)</small><b id="pmPct">--</b></div>
</div>
<canvas id="rateWave" width="640" height="120" style="height:120px"></canvas>
<div class="muted">step rate, last 4 s; each spike is one step</div>
<div class="row"><button class="sec" onclick="post('/steps/reset')">Reset count</button></div>
<div class="take">
  <h1>Counted take</h1>
  <div id="takeIdle">
    <div class="row">
      <button class="act on" data-a="WALK">Walk</button><button class="act" data-a="RUN">Run</button>
      <button class="act" data-a="STAND">Stand</button><button class="act" data-a="BIKE">Bike</button><button class="act" data-a="MIXED">Mixed</button>
    </div>
    <div class="row"><input id="who2" placeholder="who (optional)"><button onclick="takeStart()">Start</button></div>
    <div class="muted">Count your steps out loud. Stand = fidget on the spot (0 steps). Bike = lie back and pedal (0). Mixed = anything, e.g. stand, walk, stop, walk.</div>
  </div>
  <div id="takeLive" class="hidden row" style="align-items:center">
    <span id="takeInfo" class="muted"></span><span style="flex:1"></span><button class="stop" onclick="takeStop()">Stop</button>
  </div>
  <div id="takeAsk" class="hidden">
    <div class="row" style="align-items:center">
      <span>How many steps did you count?</span>
      <input id="takeSteps" type="number" min="0" step="1" inputmode="numeric">
      <button onclick="takeSave()">Save</button><button class="sec" onclick="takeDiscard()">Discard</button>
    </div>
    <div id="takeErr" class="muted"></div>
    <div class="muted">pre-filled with the device's own count for this take; correct it to what you counted</div>
  </div>
</div>
</section>
```

`post(path)` returns its request promise; a non-OK response rejects with the server's `error` text (fallback `HTTP <status>`), which also shows in `#rec` as before:
```js
function post(path) {   // -> the request's promise; it rejects with the server's error text, which also shows in #rec
  const who = encodeURIComponent($("who").value.trim());
  const req = fetch(path + (path.includes("?") ? "&" : "?") + "who=" + who, {method: "POST"})
    .then(r => r.ok ? r.json() : r.json().catch(() => ({})).then(b => { throw new Error(b.error || "HTTP " + r.status); }))
    .then(render);
  req.catch(e => { $("rec").textContent = "error: " + e.message; });
  return req;
}
```
Script additions (put after `drawWave`). The take's start count lives on the board (`steps.take`); a failed Save stays on the ask row and shows the reason in `#takeErr`; the take controls sync even with no step model; `who` and `who2` mirror each other:
```js
// ---- steps tab ----
const rateCanvas = $("rateWave"), rctx2 = rateCanvas.getContext("2d");
function sizeRate() { rateCanvas.width = rateCanvas.clientWidth * devicePixelRatio; rateCanvas.height = 120 * devicePixelRatio; }
sizeRate(); window.addEventListener("resize", sizeRate);
function drawRate(w) {
  const W = rateCanvas.width, H = rateCanvas.height, hi = 4;
  rctx2.fillStyle = "#0a0a0a"; rctx2.fillRect(0, 0, W, H);
  const y = r => H - 2 - Math.min(r, hi) / hi * (H - 4);
  rctx2.strokeStyle = "#333"; rctx2.beginPath(); rctx2.moveTo(0, y(1)); rctx2.lineTo(W, y(1)); rctx2.stroke();
  if (w.length < 2) return;
  rctx2.strokeStyle = "#ffde00"; rctx2.lineWidth = 2 * devicePixelRatio; rctx2.beginPath();
  w.forEach((r, i) => { const x = i / (w.length - 1) * W; i ? rctx2.lineTo(x, y(r)) : rctx2.moveTo(x, y(r)); });
  rctx2.stroke();
}
let takeActivity = "WALK", takeMode = "idle", lastSteps = null;
["who", "who2"].forEach((id, i, ids) => $(id).addEventListener("input", () => { $(ids[1 - i]).value = $(id).value; }));   // one name for both tabs
document.querySelectorAll(".take .act").forEach(b => b.onclick = () => { takeActivity = b.dataset.a; document.querySelectorAll(".take .act").forEach(x => x.classList.toggle("on", x === b)); });
function setTakeMode(m) { takeMode = m; ["Idle", "Live", "Ask"].forEach(k => $("take" + k).classList.toggle("hidden", m !== k.toLowerCase())); }
function takeStart() {
  $("takeErr").textContent = "";
  post("/record?label=" + takeActivity + "&steps=1"); setTakeMode("live");
}
function takeStop() {                               // the file keeps recording (standing still) until Save or Discard
  $("takeSteps").value = lastSteps ? Math.max(0, Math.round(lastSteps.take)) : 0;
  setTakeMode("ask"); $("takeSteps").focus();
}
function takeSave() {
  const n = parseInt($("takeSteps").value, 10);
  if (!(n >= 0)) { $("takeSteps").focus(); return; }
  post("/stop?steps=" + n).then(() => setTakeMode("idle"))   // a failed save stays on the ask row with the reason
    .catch(e => { $("takeErr").textContent = e.message; });
}
function takeDiscard() { post("/stop"); setTakeMode("idle"); }
function renderSteps(st, rec, act) {
  lastSteps = st;
  const counted = rec.active && rec.kind === "counted";   // the take controls work with or without a step model
  $("takeClock").textContent = counted ? rec.elapsed_s.toFixed(0) + " s" : "--";
  $("takeInfo").textContent = counted ? `${rec.label} · ${rec.elapsed_s.toFixed(0)} s · ${rec.rows} rows` + (st ? ` · device so far ${Math.max(0, Math.round(st.take))}` : "") : "";
  if (counted && takeMode === "idle") setTakeMode("live");
  if (!rec.active && takeMode === "live") setTakeMode("idle");
  if (!st) { $("stepCount").textContent = "--"; $("stepPm").textContent = ""; $("stepAct").textContent = "no step model"; return; }
  $("stepAct").textContent = act + (st.trained ? "" : " · untrained seed weights");   // do not trust the count before training
  $("stepCount").textContent = st.count; $("stepPm").textContent = st.count ? "± " + st.pm : "";
  $("spm").textContent = st.spm; $("pmPct").textContent = "± " + st.pm_pct + " %";
  drawRate(st.wave || []);
}
```
In `render(s)` add `renderSteps(s.steps, s.recorder, s.act);`. Replace `showTab` (the Gait canvas is resized when its tab opens):
```js
function showTab(name) {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  ["gait", "race", "steps"].forEach(t => $(t).classList.toggle("hidden", t !== name));
  if (name === "gait") sizeCanvas();
  if (name === "race") fitRace();
  if (name === "steps") sizeRate();
  history.replaceState(null, "", name === "gait" ? "/" : "/" + name);
}
document.querySelectorAll(".tab").forEach(b => b.onclick = () => showTab(b.dataset.tab));
const p = location.pathname.slice(1) || location.hash.slice(1);
showTab(["race", "steps"].includes(p) ? p : "gait");
```
Keep `page.html` under 500 lines: the Steps tab pushes it over, so move the race drawing script (everything from `const ghostColor = ...` through `drawRace();`: `renderRace`, `ordinal`, the stadium, runners, HUD, `fitRace`, `drawRace`) into `app/python/race.js`, served at `GET /race.js`, and load it with `<script src="/race.js"></script>` just before the main `<script>`. Both are classic scripts sharing one global scope, so every function the page calls stays global. race.js runs first, so its one load-time element lookup uses `document.getElementById("raceCanvas")`; its other `$(...)` calls are inside functions that run after the main script defines `$`. (Result: `page.html` ~241 lines, `race.js` ~340.) Check syntax and duplicate top-level names with `node --check` on `race.js`, on the inline script, and on the two concatenated.

- [ ] **Step 5: Run the tests and look at the page**

Run: `.venv/bin/python -m pytest tests -q` → Expected: all pass.
Run the demo harness and open the page:
```bash
cd ~/ME193/P6-Gait-Wearable && .venv/bin/python - <<'EOF' &
import sys, threading, time; sys.path[:0] = ["app/python", "tests"]
from models import Tree; from seed_models import SEED1, SEED2, seed_stepnet; from stepnet import StepNet
from pipeline import Gait; from recorder import Recorder; from web import serve; from synth import synth; from test_pipeline import to_batch
g = Gait(Tree(SEED1), Tree(SEED2), Recorder("/tmp/steps-demo", rolling=False), stepnet=StepNet(seed_stepnet()))
serve(g, port=8090); print("http://127.0.0.1:8090/steps")
i = 0
while True:
    g.on_batch(to_batch(i, synth("WALK", 0.1, phase=i * 0.1))); i += 1; g.tick(); time.sleep(0.1)
EOF
```
Open `http://127.0.0.1:8090/steps` in the browser: the count ticks, the yellow rate trace moves, Start → Stop → the number box is pre-filled → Save closes the take and lists the file on the Gait tab. Kill the harness afterwards.

- [ ] **Step 6: Commit**

```bash
git add app/python/web.py app/python/pipeline.py app/python/page.html app/python/race.js tests/test_web.py tests/test_pipeline.py docs/superpowers/plans/2026-11-01-step-counter.md
git commit -m "P6-Gait-Wearable: Steps tab (count ± half-width, rate trace, reset, counted-take controls) and routes /steps, /record?steps=1, /stop?steps=, /steps/reset; race drawing script moved to race.js"
```

---

### Task 6: Deploy and record the counted takes (user at the bench)

**Files:** none (board + `data/`)

- [ ] **Step 1: Deploy** (board on the Mac cable): `scripts/deploy.sh`. Watch `adb shell tail -f ~/ArduinoApps/gait-wearable/data/app.log` for `models loaded` and no `no step model` warning.
- [ ] **Step 2: Phone on the hotspot**, open `http://<board>:8080/steps`. The seed net shows nonsense counts; that is expected before training.
- [ ] **Step 3: Record ~25 takes** following the protocol: walk slow ×3, walk normal ×4, walk fast ×3, run ×4, stand with fidgets ×3 (type 0), air-bike ×3 (type 0), mixed ×5. Each 15 to 40 s, count out loud, type the number on Stop. Name `Rafae` in the who box.
- [ ] **Step 4: Pull**: board back on the cable, `scripts/pull_data.sh`; check `ls data/*-steps-*` shows the files with counts.

---

### Task 7: PyTorch model and JSON export (`train/stepnet_torch.py`)

**Files:**
- Create: `train/stepnet_torch.py`
- Modify: `train/requirements.txt` (add `torch`)
- Create: `tests/test_steps_train.py` (first test)

- [ ] **Step 1: Install torch**

Run: `echo torch >> train/requirements.txt && .venv/bin/pip install -r train/requirements.txt -q && .venv/bin/python -c "import torch; print(torch.__version__)"`
Expected: a version prints.

- [ ] **Step 2: Write the failing test `tests/test_steps_train.py`**

```python
import json
import random

import pytest

torch = pytest.importorskip("torch")

from stepnet import DT, StepNet          # noqa: E402
from stepnet_torch import StepNetTorch  # noqa: E402


def test_the_torch_net_is_causal():
    torch.manual_seed(0)
    model = StepNetTorch()
    x = torch.randn(1, 6, 80)
    y = x.clone()
    y[:, :, 41:] = torch.randn(1, 6, 39)          # change only the future of t = 40
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.equal(a[..., :41], b[..., :41]) and not torch.equal(a[..., 41:], b[..., 41:])


def test_export_matches_the_pure_python_forward(tmp_path):
    torch.manual_seed(0)
    model = StepNetTorch()
    mean, std = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.3, 0.3, 100.0, 50.0, 50.0]
    model.export(tmp_path / "s.json", mean, std, {"trained": "test"})
    spec = json.loads((tmp_path / "s.json").read_text())
    assert spec["dt"] == DT and len(spec["layers"]) == 4 and spec["layers"][-1]["act"] == "softplus"
    net = StepNet(spec)
    rng = random.Random(1)
    x = [tuple(rng.uniform(-2, 2) for _ in range(6)) for _ in range(120)]
    xt = torch.tensor([[(s[c] - mean[c]) / std[c] for s in x] for c in range(6)], dtype=torch.float32)[None]
    with torch.no_grad():
        ref = model(xt)[0, 0].tolist()
        cnt = model.count(xt).item()
    got = net.forward(x)
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-5
    assert cnt == pytest.approx(sum(got) * DT, abs=1e-3)
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python -m pytest tests/test_steps_train.py -q`
Expected: `ModuleNotFoundError: No module named 'stepnet_torch'`

- [ ] **Step 4: Write `train/stepnet_torch.py`**

```python
"""StepNet in PyTorch: the same causal dilated conv stack as app/python/stepnet.py, plus the JSON export it reads."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "python"))
from stepnet import CHANNELS, DECIMATE, DT, RATE_HZ  # noqa: E402


class StepNetTorch(nn.Module):
    def __init__(self, channels=12, kernel=5, dilations=(1, 2, 4)):
        super().__init__()
        self.kernel, self.dilations = kernel, tuple(dilations)
        self.convs = nn.ModuleList()
        cin = CHANNELS
        for d in self.dilations:
            self.convs.append(nn.Conv1d(cin, channels, kernel, dilation=d))
            cin = channels
        self.head = nn.Conv1d(cin, 1, 1)

    def forward(self, x):
        """x: (B, 6, T) standardised -> rate (B, 1, T) in steps/s, causal."""
        h = x
        for conv in self.convs:
            h = F.relu(conv(F.pad(h, ((self.kernel - 1) * conv.dilation[0], 0))))
        return F.softplus(self.head(h))

    def count(self, x):
        """(B, 6, T) -> (B,) predicted steps."""
        return self.forward(x).sum(dim=-1).squeeze(1) * DT

    def spec(self, mean, std, meta):
        layers = [{"w": c.weight.detach().tolist(), "b": c.bias.detach().tolist(), "dilation": c.dilation[0], "act": "relu"}
                  for c in self.convs]
        layers.append({"w": self.head.weight.detach().tolist(), "b": self.head.bias.detach().tolist(), "dilation": 1, "act": "softplus"})
        return {"rate_hz": RATE_HZ, "decimate": DECIMATE, "dt": DT, "kernel": self.kernel,
                "mean": [float(v) for v in mean], "std": [float(v) for v in std], "layers": layers, "meta": meta}

    def export(self, path, mean, std, meta):
        Path(path).write_text(json.dumps(self.spec(mean, std, meta)))
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/test_steps_train.py -q` → Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add train/stepnet_torch.py train/requirements.txt tests/test_steps_train.py
git commit -m "P6-Gait-Wearable: StepNetTorch (causal dilated convs + softplus head) with JSON export that the board's stepnet.py reproduces to 1e-4"
```

---

### Task 8: Training, LORO, metrics, export (`train/steps.py`)

**Files:**
- Create: `train/steps.py`
- Modify: `tests/synth.py` (append `steps_take`)
- Modify: `tests/test_steps_train.py` (append)
- Modify: `app/python/stepnet.py`, `tests/test_stepnet.py` (a `null` `pm_pct95` falls back to 100 %)

- [ ] **Step 1: Add synthetic counted takes to `tests/synth.py`**

```python
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
```

- [ ] **Step 2: Append the failing tests to `tests/test_steps_train.py`**

```python
import numpy as np                                                  # noqa: E402

from synth import steps_take                                        # noqa: E402
from steps import (EPOCHS, Recording, augment_pair, fit, gain, load_recordings, loro, metrics, normalisation,  # noqa: E402
                   predict, rates, rotate, stretch, to_array)


def synthetic_recordings():
    specs = [("WALK", 20, 1.6), ("WALK", 30, 1.9), ("WALK", 44, 2.1), ("RUN", 36, 2.8), ("RUN", 25, 2.6),
             ("STAND", 0, 1.8), ("BIKE", 0, 1.2), ("WALK", 12, 1.7)]
    recs = []
    for i, (kind, n, cad) in enumerate(specs):
        x = to_array(steps_take(n, cad, kind=kind) if n else steps_take(30, cad, kind=kind))
        recs.append(Recording("%d-steps-%s-n%d-t.csv" % (i, kind.lower(), n), kind, n, "t", x, False))
    return recs


def test_augmentations_preserve_what_they_should():
    rng = np.random.default_rng(0)
    x = to_array(steps_take(10))
    assert x.shape[0] == 6
    r = rotate(x, rng)
    assert r.shape == x.shape
    assert np.allclose(np.linalg.norm(r[:3], axis=0), np.linalg.norm(x[:3], axis=0), atol=1e-6)   # a rotation keeps |a|
    assert np.allclose(np.linalg.norm(r[3:], axis=0), np.linalg.norm(x[3:], axis=0), atol=1e-6)
    g = gain(x, rng)
    assert g.shape == x.shape and 0.85 < (g[0].std() / x[0].std()) < 1.15
    s = stretch(x, rng)
    assert 0.69 * x.shape[1] < s.shape[1] < 1.31 * x.shape[1]
    a, b = synthetic_recordings()[:2]
    px, py = augment_pair(a, b)
    assert px.shape[1] == a.x.shape[1] + b.x.shape[1] and py == a.steps + b.steps


@pytest.fixture(scope="module")
def fitted():
    """One full-length fit on the synthetic takes, shared by the tests below (~3 s). The cadence only takes over from
    "time spent moving" late in training (a 60-epoch fit is still a duration counter), so this is the real EPOCHS."""
    recs = synthetic_recordings()
    return recs, fit(recs, epochs=EPOCHS, seed=0)


def test_fit_learns_synthetic_counts_and_loro_runs(fitted):
    recs, (model, losses, fmean, fstd) = fitted
    mean, std = normalisation(recs)
    assert mean.shape == (6,) and (std > 0).all()
    assert np.array_equal(fmean, mean) and np.array_equal(fstd, std)
    assert losses[-1] < losses[0]
    worst = max(abs(predict(model, mean, std, r.x) - r.steps) for r in recs)
    assert worst < 6, worst                                           # a tiny net, synthetic data
    res = loro(recs[:4], epochs=15, seed=0)
    assert len(res) == 4 and {r["name"] for r in res} == {r.name for r in recs[:4]}
    m = metrics(res)
    assert set(m) >= {"n", "mae_steps", "mape_pct", "bias_steps", "ci95_pct", "pm_pct95", "by_activity"}


def test_rate_tracks_cadence(fitted):
    """The same 44 steps at four cadences (none of these takes is in the training set): the count stays within 10 %
    and the rate while moving follows the cadence, so the net counts steps, not time spent moving."""
    _, (model, _, mean, std) = fitted
    for cad in (1.6, 1.8, 2.1, 2.3):
        x = to_array(steps_take(44, cad))
        pred, moving = predict(model, mean, std, x), rates(model, mean, std, x)[25:-25].mean()   # 1 s still each end
        assert abs(pred - 44) <= 4.4, (cad, pred)
        assert abs(moving - cad) <= 0.1 * cad, (cad, moving)


def test_load_recordings_skips_gaps_and_sorts_out_demo_takes(tmp_path):
    def write(name, gap_at=None):
        rows, t = ["t,label,ax,ay,az,gx,gy,gz"], 0.0
        for i, smp in enumerate(steps_take(6)):
            t += 1.0 if i == gap_at else 0.01
            rows.append("%.3f,WALK," % t + ",".join("%.4f" % v for v in smp))
        (tmp_path / name).write_text("\n".join(rows) + "\n")
    write("20261101-120001-steps-walk-n6-Rafae.csv")
    write("20261101-120002-steps-walk-n6-Rafae-demo.csv")
    write("20261101-120003-steps-walk-n6-demo.csv")
    write("20261101-120004-steps-walk-n6-Ademo.csv")        # a person called Ademo, not a demo take
    write("20261101-120005-steps-walk-n6-Rafae-2.csv", gap_at=200)
    write("20261101-120006-guided-Rafae.csv")
    logged = []
    recs = load_recordings(tmp_path, log=logged.append)
    assert [(r.name[14], r.demo) for r in recs] == [("1", False), ("2", True), ("3", True), ("4", False)]
    assert len(logged) == 1 and "5-steps" in logged[0]
    assert metrics([{"activity": "STAND", "true": 0, "pred": 0.4}])["pm_pct95"] is None
```

- [ ] **Step 3: Run them**

Run: `.venv/bin/python -m pytest tests/test_steps_train.py -q`
Expected: `ModuleNotFoundError: No module named 'steps'`

- [ ] **Step 4: Write `train/steps.py`**

```python
"""Train StepNet from the counted takes and measure how good it is.

  .venv/bin/python train/steps.py                 # fit on data/*-steps-*-n<N>*.csv -> models/stepnet.json
  .venv/bin/python train/steps.py --report        # + leave-one-recording-out, 5 seeds, demo takes, figures, docs/steps_results.md

A take's true count is in its file name (recorder.parse_counted). Takes whose `who` is "demo" or ends in "-demo" are
never trained on: they are the fresh takes the video shows and are reported separately. A take with a break in t over
0.8 s (samples lost) is skipped. Decimation is the board's own (stepnet.decimate), normalisation is fit on the training
fold only. Loss: smooth-L1 between the integrated rate and the typed count; augmentation keeps the count true (pairing
two takes, frame rotation, gain, and on every pass a time-stretch by U(0.7, 1.3): the same count over a new duration,
which is what makes the rate follow the cadence instead of the time spent moving).
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "python"))
sys.path.insert(0, str(ROOT / "train"))
from recorder import parse_counted  # noqa: E402
from stepnet import decimate  # noqa: E402
from stepnet_torch import StepNetTorch  # noqa: E402

DATA, MODELS, FIGS = ROOT / "data", ROOT / "models", ROOT / "docs" / "figures"
EPOCHS, LR, LR_DROP_AT, SEEDS, BOOT = 400, 3e-3, 0.75, 5, 2000
P_AUG, STRETCH = 0.5, (0.7, 1.3)   # pairing/rotation/gain each half the time; a time-stretch in STRETCH on every pass
MAX_GAP_S = 0.8    # a break in t this long (FIFO overflow, re-probe; train.py's threshold) loses steps: skip the take


@dataclass
class Recording:
    name: str
    activity: str
    steps: int
    who: str
    x: np.ndarray      # (6, T) decimated raw channels
    demo: bool


def load_rows(path):
    """-> (t list, [(ax, ay, az, gx, gy, gz), ...]) in file order."""
    ts, out = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                row = float(r["t"]), tuple(float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz"))
            except (TypeError, ValueError):
                continue   # a truncated last row (power pulled)
            ts.append(row[0])
            out.append(row[1])
    return ts, out


def to_array(samples):
    """100 Hz samples -> (6, T) decimated array, exactly as the board does it."""
    dec, _ = decimate(samples)
    return np.array(dec, dtype=np.float32).T


def is_demo(who):
    w = who.lower()
    return w == "demo" or w.endswith("-demo")


def load_recordings(data_dir=DATA, log=print):
    recs = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        parsed = parse_counted(p.name)
        if not parsed:
            continue
        act, n, who = parsed
        ts, rows = load_rows(p)
        gap = float(np.diff(ts).max()) if len(ts) > 1 else 0.0
        if gap > MAX_GAP_S:
            log("skipping %s: a %.1f s gap in t (samples lost, the count would not match)" % (p.name, gap))
            continue
        x = to_array(rows)
        if x.size == 0:
            continue
        recs.append(Recording(p.name, act, n, who, x, is_demo(who)))
    return recs


# --- augmentation (count-preserving) -----------------------------------------------------------------------------
def rotate(x, rng, max_deg=15.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return np.vstack([R @ x[:3], R @ x[3:]]).astype(np.float32)


def gain(x, rng):
    return (x * rng.uniform(0.9, 1.1)).astype(np.float32)


def stretch(x, rng, lo=STRETCH[0], hi=STRETCH[1]):
    f = rng.uniform(lo, hi)
    T = x.shape[1]
    T2 = max(2, int(round(T * f)))
    src, dst = np.linspace(0, 1, T), np.linspace(0, 1, T2)
    return np.vstack([np.interp(dst, src, ch) for ch in x]).astype(np.float32)


def augment_pair(a, b):
    return np.concatenate([a.x, b.x], axis=1), a.steps + b.steps


def normalisation(recs):
    allx = np.concatenate([r.x for r in recs], axis=1)
    return allx.mean(axis=1), np.maximum(allx.std(axis=1), 1e-3)


def tensor(x, mean, std):
    return torch.tensor((x - mean[:, None]) / std[:, None], dtype=torch.float32)[None]


# --- fitting and predicting ---------------------------------------------------------------------------------------
def fit(recs, epochs=EPOCHS, seed=0, augment=True, log=None):
    """-> (model, per-epoch mean loss, mean, std). mean/std = normalisation(recs), the inputs the model expects."""
    mean, std = normalisation(recs)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StepNetTorch()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for ep in range(epochs):
        if ep == int(epochs * LR_DROP_AT):
            for g in opt.param_groups:
                g["lr"] = LR / 10
        total = 0.0
        for i in rng.permutation(len(recs)):
            x, y = recs[i].x, recs[i].steps
            if augment:
                if rng.random() < P_AUG:
                    x, y = augment_pair(recs[i], recs[int(rng.integers(len(recs)))])
                if rng.random() < P_AUG:
                    x = rotate(x, rng)
                if rng.random() < P_AUG:
                    x = gain(x, rng)
                x = stretch(x, rng)   # always: the same count over a new duration forces rate ~ cadence
            loss = F.smooth_l1_loss(model.count(tensor(x, mean, std)), torch.tensor([float(y)]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(recs))
        if log and (ep % 25 == 0 or ep == epochs - 1):
            log("epoch %3d  loss %.3f" % (ep, losses[-1]))
    return model, losses, mean, std


def predict(model, mean, std, x):
    with torch.no_grad():
        return model.count(tensor(x, mean, std)).item()


def rates(model, mean, std, x):
    with torch.no_grad():
        return model(tensor(x, mean, std))[0, 0].numpy()


def loro(recs, epochs=EPOCHS, seed=0, log=None):
    """Leave-one-recording-out: fit without each take and predict it. -> [{name, activity, who, true, pred}]."""
    out = []
    for i, held in enumerate(recs):
        rest = recs[:i] + recs[i + 1:]
        model, _, mean, std = fit(rest, epochs=epochs, seed=seed)
        pred = predict(model, mean, std, held.x)
        out.append({"name": held.name, "activity": held.activity, "who": held.who, "true": held.steps, "pred": round(pred, 2)})
        if log:
            log("LORO %2d/%d %-45s true %3d  pred %6.1f" % (i + 1, len(recs), held.name, held.steps, pred))
    return out


def seed_spread(recs, n=SEEDS, epochs=EPOCHS):
    """Fit n times with different seeds on all takes; -> per-take std of the predicted count (mean over takes)."""
    preds = []
    for s in range(n):
        model, _, mean, std = fit(recs, epochs=epochs, seed=s)
        preds.append([predict(model, mean, std, r.x) for r in recs])
    return float(np.mean(np.std(np.array(preds), axis=0)))


def metrics(results, rng=None):
    rng = rng or np.random.default_rng(0)
    err = np.array([r["pred"] - r["true"] for r in results])
    nz = np.array([r["true"] > 0 for r in results])
    pct = 100 * err[nz] / np.array([r["true"] for r in results])[nz] if nz.any() else np.array([0.0])
    boots = [rng.choice(pct, len(pct)).mean() for _ in range(BOOT)]
    by = {}
    for act in sorted({r["activity"] for r in results}):
        e = np.array([r["pred"] - r["true"] for r in results if r["activity"] == act])
        by[act] = {"n": int(len(e)), "mae_steps": round(float(np.abs(e).mean()), 2), "bias_steps": round(float(e.mean()), 2)}
    return {"n": len(results), "mae_steps": round(float(np.abs(err).mean()), 2), "mape_pct": round(float(np.abs(pct).mean()), 2),
            "bias_steps": round(float(err.mean()), 2), "ci95_pct": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
            "pm_pct95": round(float(np.percentile(np.abs(pct), 95)), 2) if nz.any() else None, "by_activity": by}


# --- the report ---------------------------------------------------------------------------------------------------
def write_results(path, m, results, demo, seed_std, losses, n_train):
    lines = ["# Step counter results", "", "Generated by `train/steps.py --report` on %s from %d counted takes." % (time.strftime("%Y-%m-%d %H:%M"), n_train), "",
             "## Leave-one-recording-out", "", "| metric | value |", "|---|---|",
             "| takes | %d |" % m["n"], "| mean absolute error | %.2f steps |" % m["mae_steps"], "| mean absolute percentage error | %.1f %% |" % m["mape_pct"],
             "| bias (predicted − true) | %+.2f steps |" % m["bias_steps"], "| 95 %% bootstrap interval of the mean %% error | %+.1f %% to %+.1f %% |" % tuple(m["ci95_pct"]),
             "| 95th percentile of the absolute %% error (shown live as ±) | %s |" % ("n/a (no take with steps)" if m["pm_pct95"] is None else "%.1f %%" % m["pm_pct95"]), "| seed-to-seed spread of a prediction (5 fits) | %.2f steps |" % seed_std, "",
             "| activity | takes | MAE (steps) | bias (steps) |", "|---|---|---|---|"]
    lines += ["| %s | %d | %.2f | %+.2f |" % (a, v["n"], v["mae_steps"], v["bias_steps"]) for a, v in m["by_activity"].items()]
    lines += ["", "| take | activity | true | predicted | error |", "|---|---|---|---|---|"]
    lines += ["| %s | %s | %d | %.1f | %+.1f |" % (r["name"], r["activity"], r["true"], r["pred"], r["pred"] - r["true"]) for r in results]
    if demo:
        lines += ["", "## Fresh takes the net never saw (`-demo`)", "", "| take | true | predicted | error |", "|---|---|---|---|"]
        lines += ["| %s | %d | %.1f | %+.1f |" % (r["name"], r["true"], r["pred"], r["pred"] - r["true"]) for r in demo]
    lines += ["", "Final fit: %d epochs, last-epoch loss %.3f." % (len(losses), losses[-1]), ""]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="LORO + seeds + demo + figures + docs/steps_results.md")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args(argv)
    recs = load_recordings(a.data)
    train, demo = [r for r in recs if not r.demo], [r for r in recs if r.demo]
    print("%d counted takes (%d training, %d demo)" % (len(recs), len(train), len(demo)))
    if len(train) < 3:
        sys.exit("need at least 3 counted takes in %s (names like ...-steps-walk-n42-Rafae.csv)" % a.data)
    meta = {"trained": time.strftime("%Y-%m-%dT%H:%M:%S"), "recordings": len(train), "epochs": a.epochs, "loro": None, "seed_std_steps": None}
    results = None
    if a.report:
        if len(train) < 4:
            sys.exit("--report needs at least 4 training takes")
        results = loro(train, epochs=a.epochs, seed=a.seed, log=print)
        m = metrics(results)
        meta["loro"] = m
        meta["seed_std_steps"] = round(seed_spread(train, epochs=a.epochs), 2)
        print(json.dumps(m, indent=1))
    else:  # keep the last measured half-width so the live ± does not fall back to 100 %, and say which run measured it
        try:
            old = json.loads((MODELS / "stepnet.json").read_text())["meta"]
            meta["loro"] = old.get("loro")
            if meta["loro"]:
                meta["loro_from"] = old.get("loro_from") or old.get("trained")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    model, losses, mean, std = fit(train, epochs=a.epochs, seed=a.seed, log=print)
    demo_res = [{"name": r.name, "true": r.steps, "pred": round(predict(model, mean, std, r.x), 2)} for r in demo]
    model.export(MODELS / "stepnet.json", mean, std, meta)
    print("wrote", MODELS / "stepnet.json")
    if a.report:   # the results first, so a figure error cannot lose them
        write_results(ROOT / "docs" / "steps_results.md", m, results, demo_res, meta["seed_std_steps"], losses, len(train))
        print("wrote docs/steps_results.md")
        from steps_plots import figures
        figures(FIGS, results, m, train, model, mean, std, losses)
        print("wrote docs/figures/steps_*.png")
    for r in demo_res:
        print("demo %-45s true %3d  pred %6.1f" % (r["name"], r["true"], r["pred"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_steps_train.py -q -x`
Expected: `6 passed` in about 5 s (one shared 400-epoch fit). The cadence probe is the test that matters: without
the always-on wide time-stretch the net learns "time spent moving × ~1.85 steps/s" (44-step walks at 1.6/1.8/2.1/2.3 Hz
read 52.5/46.6/39.8/36.2), and it needs the full EPOCHS (a 60- or 150-epoch fit is still a duration counter).

- [ ] **Step 5b: Let the board fall back when no take had steps**

In `StepNet.__init__` (`app/python/stepnet.py`) replace the `pm_pct95` line with:

```python
            pm = (self.meta.get("loro") or {}).get("pm_pct95")
            self.pm_pct95 = 100.0 if pm is None else float(pm)   # None: training had no take with steps
```

and append to `tests/test_stepnet.py`:

```python
def test_a_missing_half_width_falls_back_to_100_percent():
    assert StepNet(seed_stepnet(pm_pct95=None)).pm_pct95 == 100.0
```

- [ ] **Step 6: Commit**

```bash
git add train/steps.py tests/synth.py tests/test_steps_train.py app/python/stepnet.py tests/test_stepnet.py
git commit -m "P6-Gait-Wearable: train/steps.py (counted-take dataset from file names, count-preserving augmentation, smooth-L1 fit, leave-one-recording-out, seed spread, metrics, export)"
```

---

### Task 9: Figures (`train/steps_plots.py`)

**Files:**
- Create: `train/steps_plots.py`
- Modify: `tests/test_steps_train.py` (append)

- [ ] **Step 1: Append the failing test**

```python
def test_figures_are_written(tmp_path):
    from steps_plots import figures
    recs = synthetic_recordings()
    model, losses, mean, std = fit(recs, epochs=3, seed=0)
    res = [{"name": r.name, "activity": r.activity, "who": r.who, "true": r.steps, "pred": r.steps + 1.0} for r in recs]
    figures(tmp_path, res, metrics(res), recs, model, mean, std, losses)
    for n in ("steps_scatter.png", "steps_errors.png", "steps_rate.png", "steps_training.png"):
        assert (tmp_path / n).stat().st_size > 1000
```

- [ ] **Step 2: Run** → Expected: `ModuleNotFoundError: No module named 'steps_plots'`

- [ ] **Step 3: Write `train/steps_plots.py`** (the rate figure has four stacked panels, no twin axis: one axis per quantity)

```python
"""Figures for the step-counter write-up: LORO scatter, error histogram, one take's traces + rate + count, training loss."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stepnet import DT  # noqa: E402
from steps import rates  # noqa: E402

COLORS = {"WALK": "#16a034", "RUN": "#ff7800", "STAND": "#555555", "BIKE": "#0093e7", "MIXED": "#b03ac0"}   # the phone page's
INK, MUTED, MAIN = "#222222", "#888888", "#0093e7"
CHANNELS = ("#4c78a8", "#e0872b", "#6b6b6b")   # x, y, z: three distinct, non-activity hues


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(path, results, m):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    hi = max([r["true"] for r in results] + [r["pred"] for r in results] + [10]) * 1.1
    ax.plot([0, hi], [0, hi], color="#999", lw=1, label="perfect")
    xs = np.linspace(0, hi, 50)
    if m["pm_pct95"] is not None:   # None when no take had steps
        ax.fill_between(xs, xs * (1 - m["pm_pct95"] / 100), xs * (1 + m["pm_pct95"] / 100), color="#ddd", alpha=0.6, label="±%.0f %% (95th pct)" % m["pm_pct95"])
    for act in sorted({r["activity"] for r in results}):
        pts = [r for r in results if r["activity"] == act]
        ax.scatter([r["true"] for r in pts], [r["pred"] for r in pts], color=COLORS.get(act, "k"), label=act, zorder=3, s=36, edgecolor="white", linewidth=0.8, clip_on=False)
    ax.set_xlabel("counted steps (true)")
    ax.set_ylabel("StepNet steps (leave-one-recording-out)")
    ax.set_title("MAE %.1f steps, MAPE %.1f %%" % (m["mae_steps"], m["mape_pct"]))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    style(ax)
    save(fig, path)


def errors(path, results):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    e = [r["pred"] - r["true"] for r in results]
    ax.hist(e, bins=max(5, len(e) // 2), color=MAIN, edgecolor="white")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("predicted − true (steps)")
    ax.set_ylabel("takes")
    ax.set_title("Leave-one-recording-out errors")
    style(ax)
    save(fig, path)


def rate_curve(path, rec, model, mean, std):
    r = rates(model, mean, std, rec.x)
    t = np.arange(rec.x.shape[1]) * DT
    fig, axs = plt.subplots(4, 1, figsize=(7, 7.5), sharex=True)
    for ax, (chs, unit, off) in zip(axs[:2], ((("ax", "ay", "az"), "accel (g)", 0), (("gx", "gy", "gz"), "gyro (deg/s)", 3))):
        for i, (name, c) in enumerate(zip(chs, CHANNELS)):
            ax.plot(t, rec.x[off + i], color=c, lw=0.8, label=name)
        ax.set_ylabel(unit)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))   # headroom so the legend never sits on the trace
        ax.legend(fontsize=7, ncol=3, loc="upper right", frameon=False)
    axs[0].set_title("%s: %d counted steps" % (rec.name, rec.steps))
    axs[2].fill_between(t, 0, r, color="#ffde00", alpha=0.85)
    axs[2].plot(t, r, color="#c9a800", lw=0.8)
    axs[2].set_ylabel("step rate (steps/s)")
    axs[3].plot(t, np.cumsum(r) * DT, color=MAIN, lw=2, label="count = area under the rate")
    axs[3].axhline(rec.steps, color=MUTED, ls="--", lw=1, label="counted: %d" % rec.steps)
    axs[3].set_ylabel("steps")
    axs[3].set_xlabel("time (s)")
    axs[3].legend(fontsize=7, loc="upper left", frameon=False)
    for ax in axs:
        style(ax)
    save(fig, path)


def training(path, losses):
    fig, ax = plt.subplots(figsize=(5.2, 3))
    ax.plot(losses, color=MAIN)
    ax.set_xlabel("epoch")
    ax.set_ylabel("smooth-L1 loss (steps)")
    ax.set_yscale("log")
    ax.set_title("Training loss")
    style(ax)
    save(fig, path)


def figures(out_dir, results, m, recs, model, mean, std, losses):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scatter(out / "steps_scatter.png", results, m)
    errors(out / "steps_errors.png", results)
    walks = [r for r in recs if r.activity == "WALK" and r.steps > 0] or recs
    rate_curve(out / "steps_rate.png", max(walks, key=lambda r: r.steps), model, mean, std)
    training(out / "steps_training.png", losses)
```

- [ ] **Step 4: Run the tests** → `.venv/bin/python -m pytest tests/test_steps_train.py -q` → Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add train/steps_plots.py tests/test_steps_train.py
git commit -m "P6-Gait-Wearable: step-counter figures (LORO scatter with ± band, error histogram, traces + rate + count, training loss)"
```

---

### Task 10: Train on the real takes, report, deploy

**Files:** generated `models/stepnet.json`, `docs/steps_results.md`, `docs/figures/steps_*.png`

- [ ] **Step 1: Full report run** (takes a few minutes: ~25 LORO fits + 5 seeds + the final fit)

Run: `cd ~/ME193/P6-Gait-Wearable && .venv/bin/python train/steps.py --report 2>&1 | tee /tmp/steps-report.log | tail -40`
Expected: LORO lines, a metrics JSON, `wrote models/stepnet.json`, `wrote docs/steps_results.md ...`.

- [ ] **Step 2: Judge it.** Open `docs/steps_results.md`. Good: MAPE under ~8 %, 0-step takes under 2 steps, no activity with a large bias. If WALK is fine but 0-step takes count steps, or MAPE is above 15 %: try `--epochs 600`; if still poor, report the table to the user and stop (the spec names the fallback: an auxiliary gyro-peak target; that is a design change, not a plan step).

- [ ] **Step 3: Confirm the board code reproduces the export**: `.venv/bin/python -m pytest tests/test_steps_train.py tests/test_stepnet.py -q` and
```bash
.venv/bin/python -c "
import sys; sys.path[:0]=['app/python','train']; from stepnet import StepNet; from steps import load_recordings, predict
from steps import to_array; import json
net = StepNet.load('models/stepnet.json'); recs = load_recordings()[:3]
for r in recs:
    dec = [tuple(c) for c in r.x.T]; print(r.name, r.steps, round(sum(net.forward(dec)) * 0.04, 1))"
```
Expected: the pure-Python counts match the LORO/fit predictions to about 0.1.

- [ ] **Step 4: Deploy** (board on the cable): `scripts/deploy.sh`; open the Steps tab on the phone, walk 20 counted steps: the count should land within the shown ±.

- [ ] **Step 5: Commit**

```bash
git add models/stepnet.json docs/steps_results.md docs/figures/steps_scatter.png docs/figures/steps_errors.png docs/figures/steps_rate.png docs/figures/steps_training.png
git commit -m "P6-Gait-Wearable: StepNet trained on N counted takes (LORO MAE x.x steps, MAPE y.y %), results and figures"
```
(fill N, x.x, y.y from the report.)

---

### Task 11: Demo takes and the video (user at the bench)

- [ ] **Step 1:** On the hotspot, record three fresh walks with who = `Rafae-demo` (e.g. 25, 40, 60 steps), typing the counted number; also a short run if wanted.
- [ ] **Step 2:** Film the phone screen and the feet in one shot for one of them (the live count versus your spoken count), or screen-record the phone while someone films.
- [ ] **Step 3:** Board on the cable: `scripts/pull_data.sh && .venv/bin/python train/steps.py --report` again (the demo takes now appear in the "fresh takes" table of `docs/steps_results.md`; the trained weights do not change since demo takes are excluded). Commit the updated results and figures as in Task 10 Step 5.

---

### Task 12: Manim explainer animation (`docs/animation/`)

**Files:**
- Create: `docs/animation/requirements.txt`, `docs/animation/stepnet_explainer.py`, `docs/animation/render.sh`
- Modify: `.gitignore` (add `docs/animation/media/`)

- [ ] **Step 1: Install Manim into the venv**

```bash
cd ~/ME193/P6-Gait-Wearable && printf 'manim>=0.18\n' > docs/animation/requirements.txt && .venv/bin/pip install -r docs/animation/requirements.txt -q && .venv/bin/manim --version
```
Expected: `Manim Community v0.18.x` or newer. (Manim ≥ 0.18 ships its own video encoder via PyAV; no ffmpeg or LaTeX is needed because the scenes use `Text` only.) If `pycairo` or `manimpango` fails to build, run `brew install cairo pango pkg-config` and retry (pycairo needed cairo here). `DecimalNumber` defaults to MathTex (LaTeX), so the scene passes `mob_class=Text`; counters change with `ChangeDecimalToValue`, not `.animate.set_value` (which morphs glyphs).

- [ ] **Step 2: Write `docs/animation/stepnet_explainer.py`**

```python
"""How the step counter works, as one Manim scene (~75 s, captions only, no LaTeX).

  docs/animation/render.sh            # -> docs/animation/stepnet_explainer.mp4 (1080p) and _small.mp4 (480p, for Notion)
"""
import math
import textwrap

import numpy as np
from manim import (BLUE, ChangeDecimalToValue, DOWN, GREEN, GREY_B, LEFT, ORANGE, RED, RIGHT, UP, WHITE, YELLOW, Arrow, Axes, Create,
                   DecimalNumber, Dot, FadeIn, FadeOut, Line, Rectangle, RoundedRectangle, Scene, Text, Transform,
                   ValueTracker, VGroup, VMobject, Write, always_redraw, config, rate_functions)

config.background_color = "#0b0b10"
Text.set_default(font="Helvetica Neue")
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
CH_COLORS = [RED, GREEN, BLUE, ORANGE, YELLOW, "#c084fc"]
STEP_F, STEP_T0, N_STEPS = 1.5, 0.6, 7  # the demo walk in part 4: 7 steps at 1.5 steps/s


def caption(txt, size=26, chars=72):
    """Bottom caption, wrapped to lines and shrunk if it would still overflow the frame."""
    t = Text("\n".join(textwrap.wrap(txt, chars)), font_size=size, color=GREY_B, line_spacing=0.9)
    if t.width > config.frame_width - 1.0:
        t.scale_to_fit_width(config.frame_width - 1.0)
    return t.to_edge(DOWN, buff=0.35)


def num(value, **kw):
    """A DecimalNumber drawn with Text glyphs (the default MathTex needs LaTeX)."""
    return DecimalNumber(value, mob_class=Text, **kw)


def title(txt, size=40):
    t = Text(txt, font_size=size)
    if t.width > config.frame_width - 1.0:
        t.scale_to_fit_width(config.frame_width - 1.0)
    return t.to_edge(UP, buff=0.4)


def trace(fn, x0, x1, n=200, **kw):
    return VMobject(**kw).set_points_as_corners([np.array([x, fn(x), 0]) for x in np.linspace(x0, x1, n)])


def step_rate(t):
    """One raised-cosine bump per step, each with area exactly 1, so the integral is the step count."""
    u = t - STEP_T0
    if u < 0 or u > N_STEPS / STEP_F:
        return 0.0
    return STEP_F * (1 - math.cos(2 * math.pi * STEP_F * u)) ** 2 / 1.5


def steps_so_far(t):
    u = min(max(t - STEP_T0, 0.0), N_STEPS / STEP_F)
    w = 2 * math.pi * STEP_F
    # integral of f(1 - cos wu)^2 / 1.5 = f/1.5 * (1.5u - 2 sin(wu)/w + sin(2wu)/(4w))
    return STEP_F / 1.5 * (1.5 * u - 2 * math.sin(w * u) / w + math.sin(2 * w * u) / (4 * w))


class StepNetExplainer(Scene):
    def construct(self):
        self.part_wearable()
        self.part_prepare()
        self.part_net()
        self.part_rate_to_count()
        self.part_loss()
        self.part_backprop()
        self.part_ship()

    def clear_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

    def swap_caption(self, old, txt):
        new = caption(txt)
        self.play(FadeOut(old), FadeIn(new), run_time=0.6)
        return new

    # 1. the wearable: a thigh with the board, six traces scrolling out of it
    def part_wearable(self):
        ttl = title("A thigh-worn step counter", 44)
        thigh = RoundedRectangle(corner_radius=0.6, width=1.6, height=4, color=GREY_B).move_to(LEFT * 5.2 + UP * 0.3)
        board = Rectangle(width=1.0, height=1.4, color=BLUE, fill_opacity=0.3).move_to(thigh.get_center() + UP * 0.4)
        chip = Rectangle(width=0.4, height=0.4, color=WHITE, fill_opacity=0.6).move_to(board.get_center() + DOWN * 0.4)
        lbl = Text("UNO Q + MPU6050", font_size=22).next_to(thigh, DOWN, buff=0.2)
        self.play(Write(ttl), Create(thigh), FadeIn(board), FadeIn(chip), Write(lbl))
        rows = VGroup()
        for i, (name, col) in enumerate(zip(CH, CH_COLORS)):
            y = 2.15 - i * 0.78
            f, amp = (1.8, 0.26) if i < 3 else (0.9, 0.3)
            tr = trace(lambda x, y=y, a=amp, ff=f, i=i: y + a * math.sin(2 * math.pi * ff * x + i), 0, 5.5,
                       color=col, stroke_width=3).shift(RIGHT * -2.0)
            tag = Text(name, font_size=22, color=col).next_to(tr, LEFT, buff=0.2)
            rows.add(VGroup(tag, tr))
        self.play(*[Create(r[1]) for r in rows], *[FadeIn(r[0]) for r in rows], run_time=3)
        cap = caption("Six channels at 100 Hz: 3 accelerations (g) and 3 rotation rates (deg/s). "
                      "Every one of them goes into the net.")
        self.play(FadeIn(cap))
        self.wait(2)
        self.clear_all()

    # 2. decimate 4 -> 1 and standardise
    def part_prepare(self):
        ttl = title("Prepare: 100 Hz to 25 Hz, then standardise")
        self.play(Write(ttl))
        dots = VGroup(*[Dot(point=np.array([-5 + i * 0.5, 1.3 + 0.6 * math.sin(i), 0]), color=RED) for i in range(20)])
        self.play(FadeIn(dots))
        cap = caption("Average every 4 samples into 1. Walking lives below a few Hz, so 25 Hz keeps all of it, "
                      "and the board does a quarter of the work.")
        self.play(FadeIn(cap))
        for b in range(5):
            grp = dots[4 * b:4 * b + 4]
            m = Dot(point=np.mean([d.get_center() for d in grp], axis=0), color=RED, radius=0.14)
            self.play(*[Transform(d, m.copy()) for d in grp], run_time=0.35)
        self.wait(0.5)
        cap = self.swap_caption(cap, "Standardise each channel: subtract its mean, divide by its spread. "
                                     "Now a g and a deg/s weigh the same.")
        axis = Line(LEFT * 5.5 + DOWN * 1.2, RIGHT * 5.5 + DOWN * 1.2, color=GREY_B)
        zero = Text("0", font_size=22, color=GREY_B).next_to(axis, LEFT, buff=0.15)
        formula = Text("z = (x - mean) / std", font_size=34, color=YELLOW).move_to(RIGHT * 2.5 + UP * 0.2)
        self.play(Create(axis), FadeIn(zero), Write(formula))
        self.play(dots.animate.scale(0.8).move_to(DOWN * 1.2 + LEFT * 0.2), run_time=1.5)
        self.wait(2)
        self.clear_all()

    # 3. the net: three dilated causal conv layers and the head
    def layers_drawing(self):
        cols = []
        spec = [(6, "6 inputs"), (12, "conv 12, k5\ndilation 1"), (12, "conv 12, k5\ndilation 2"),
                (12, "conv 12, k5\ndilation 4"), (1, "rate")]
        for li, (n, label) in enumerate(spec):
            x = -5.2 + li * 2.6
            units = VGroup(*[Dot(point=np.array([x, 0.45 + (n - 1) / 2 * 0.26 - k * 0.26, 0]), radius=0.08,
                                 color=CH_COLORS[k] if li == 0 else (YELLOW if li == 4 else WHITE)) for k in range(n)])
            tag = Text(label, font_size=20, color=GREY_B, line_spacing=0.8)
            tag.move_to(np.array([x, -1.45, 0]))
            cols.append(VGroup(units, tag))
        links = VGroup()
        for a, b in zip(cols[:-1], cols[1:]):
            for ua in a[0]:
                for ub in b[0]:
                    links.add(Line(ua.get_center(), ub.get_center(), stroke_width=0.6, color=GREY_B, stroke_opacity=0.35))
        return VGroup(*cols), links

    def part_net(self):
        ttl = title("The net: causal dilated 1-D convolutions")
        self.play(Write(ttl))
        cols, links = self.layers_drawing()
        self.play(Create(links, lag_ratio=0.0), *[FadeIn(c) for c in cols], run_time=2)
        cap = caption("Each layer looks only at the present and the past (causal). Kernel 5 with dilations "
                      "1, 2, 4 lets the output see the last 1.2 s of movement.")
        self.play(FadeIn(cap))
        # one time step: the input window lights up, the activity runs left to right, one rate value comes out
        out = cols[-1][0][0]
        for _ in range(2):
            for c in cols[1:-1]:
                self.play(c[0].animate.set_color(ORANGE), run_time=0.15)
            self.play(out.animate.set_color(ORANGE).scale(1.6), run_time=0.2)
            self.play(*[c[0].animate.set_color(WHITE) for c in cols[1:-1]], out.animate.set_color(YELLOW).scale(1 / 1.6),
                      run_time=0.25)
        cap = self.swap_caption(cap, "ReLU after each conv; a Softplus head keeps the output at or above zero: "
                                     "a step rate in steps per second. About 1,800 weights in all.")
        self.wait(3)
        self.clear_all()

    # 4. rate curve -> area -> count
    def part_rate_to_count(self):
        ttl = title("Rate to count: the area under the curve")
        self.play(Write(ttl))
        ax = Axes(x_range=[0, 6, 1], y_range=[0, 4.5, 1], x_length=9, y_length=3.2,
                  axis_config={"color": GREY_B, "include_numbers": False}).shift(LEFT * 0.8 + UP * 0.45)
        xl = Text("time (s)", font_size=22).next_to(ax.x_axis, DOWN, buff=0.15)
        yl = Text("steps / s", font_size=22).rotate(math.pi / 2).next_to(ax.y_axis, LEFT, buff=0.2)
        self.play(Create(ax), Write(xl), Write(yl))
        full = ax.plot(step_rate, x_range=[0, 6, 0.01], color=YELLOW)
        t = ValueTracker(0.01)
        curve = always_redraw(lambda: ax.plot(step_rate, x_range=[0, t.get_value(), 0.01], color=YELLOW))
        area = always_redraw(lambda: ax.get_area(full, x_range=[0, t.get_value()], color=YELLOW, opacity=0.35))
        counter = num(0, num_decimal_places=1, font_size=56, color=WHITE)
        counter.add_updater(lambda m: m.set_value(steps_so_far(t.get_value())))
        counter.move_to(RIGHT * 5.4 + UP * 0.7)
        clbl = Text("steps", font_size=24, color=GREY_B).next_to(counter, DOWN)
        cap = caption("count = sum of rate x 0.04 s (one sample at 25 Hz). Each step is one bump of area 1, "
                      "so the running area ticks up by one per step.")
        self.play(FadeIn(cap), FadeIn(counter), FadeIn(clbl))
        self.add(area, curve)
        self.play(t.animate.set_value(6), run_time=6, rate_func=rate_functions.linear)
        counter.clear_updaters()
        self.wait(1.5)
        self.clear_all()

    # 5. loss: predicted count vs the typed count
    def part_loss(self):
        ttl = title("Training signal: the number you counted")
        self.play(Write(ttl))
        pred = VGroup(Text("StepNet", font_size=28, color=GREY_B),
                      num(31.6, num_decimal_places=1, font_size=64, color=YELLOW)).arrange(DOWN)
        true = VGroup(Text("you counted", font_size=28, color=GREY_B),
                      num(40, num_decimal_places=0, font_size=64, color=GREEN)).arrange(DOWN)
        pred.move_to(LEFT * 3 + UP * 1.0)
        true.move_to(RIGHT * 3 + UP * 1.0)
        self.play(FadeIn(pred), FadeIn(true))
        cap = caption("No per-step labels needed. Each take has one number: the steps you counted out loud. "
                      "The loss is the gap.")
        self.play(FadeIn(cap))
        gap = Line(pred.get_right() + RIGHT * 0.3, true.get_left() + LEFT * 0.3, color=RED, stroke_width=4)
        loss = Text("loss = smooth-L1(31.6, 40) = 7.9", font_size=34, color=RED).move_to(DOWN * 0.9)
        self.play(Create(gap), Write(loss))
        self.wait(2)
        cap = self.swap_caption(cap, "Zero-step takes (standing, fidgeting, air-biking) teach it to hold a flat zero "
                                     "when you are not walking.")
        self.wait(2)
        self.clear_all()

    # 6. back-propagation: the error flows backwards, the weights nudge
    def part_backprop(self):
        ttl = title("Back-propagation: the error flows backwards")
        self.play(Write(ttl))
        cols, links = self.layers_drawing()
        self.play(FadeIn(links), FadeIn(cols))
        loss = num(7.9, num_decimal_places=2, font_size=40, color=RED)
        lgrp = VGroup(Text("loss", font_size=22, color=GREY_B), loss).arrange(DOWN, buff=0.15)
        lgrp.next_to(cols[-1][0], RIGHT, buff=0.35)
        loss.add_updater(lambda m: m.next_to(lgrp[0], DOWN, buff=0.15))  # stay centred as digits change
        self.play(FadeIn(lgrp))
        cap = caption("PyTorch's autograd works out how the loss changes with every weight. "
                      "Colour = gradient sign: red pushes a weight down, blue pushes it up.")
        self.play(FadeIn(cap))
        rng = np.random.default_rng(0)
        y_arrow = cols[1][0].get_top()[1] + 0.35
        xs = [c[0].get_center()[0] for c in cols]
        for val in [4.2, 2.1, 1.0, 0.6]:
            arrows = VGroup(*[Arrow(np.array([xs[i] - 0.15, y_arrow, 0]), np.array([xs[i - 1] + 0.15, y_arrow, 0]),
                                    color=RED, buff=0, stroke_width=4) for i in range(4, 0, -1)])
            self.play(Create(arrows, lag_ratio=0.3), run_time=0.8)
            picked = [links[int(k)] for k in rng.choice(len(links), 60, replace=False)]
            recol = [l.animate.set_stroke(color=RED if rng.random() < 0.5 else BLUE, opacity=0.9, width=2) for l in picked]
            self.play(*recol, ChangeDecimalToValue(loss, val), run_time=0.6)
            self.play(FadeOut(arrows), *[l.animate.set_stroke(color=GREY_B, opacity=0.35, width=0.6) for l in picked], run_time=0.4)
        cap = self.swap_caption(cap, "Adam nudges each weight against its gradient. 300 passes over ~25 takes, with "
                                     "rotation, gain, time-stretch and paired takes so it generalises.")
        self.wait(3)
        self.clear_all()

    # 7. ship the weights to the board, the phone shows the count
    def part_ship(self):
        ttl = title("Ship it: the weights become a JSON file on the board", 38)
        self.play(Write(ttl))
        js = RoundedRectangle(corner_radius=0.2, width=3.4, height=2.0, color=YELLOW).move_to(LEFT * 4.6 + UP * 0.4)
        jtxt = Text('stepnet.json\n{"layers": [...],\n "mean": [...],\n "std": [...]}', font_size=20,
                    line_spacing=0.8).move_to(js)
        board = Rectangle(width=1.6, height=1.2, color=BLUE, fill_opacity=0.3).move_to(UP * 0.4)
        blbl = Text("UNO Q", font_size=22).next_to(board, DOWN, buff=0.15)
        phone = RoundedRectangle(corner_radius=0.3, width=2.2, height=3.6, color=WHITE).move_to(RIGHT * 4.6 + UP * 0.4)
        count = num(0, num_decimal_places=0, font_size=64, color=WHITE).move_to(phone.get_center() + UP * 0.5)
        count.add_updater(lambda m: m.move_to(phone.get_center() + UP * 0.5))
        pm = Text("± 2 steps", font_size=26, color=GREY_B).next_to(count, DOWN)
        a1 = Arrow(js.get_right(), board.get_left(), color=GREY_B, buff=0.15)
        a2 = Arrow(board.get_right(), phone.get_left(), color=GREY_B, buff=0.15)
        self.play(Create(js), Write(jtxt))
        self.play(Create(a1), FadeIn(board), FadeIn(blbl))
        self.play(Create(a2), Create(phone), FadeIn(count), FadeIn(pm))
        cap = caption("The UNO Q runs the same convolutions in plain Python, 25 times a second, and the phone shows "
                      "the count with its measured uncertainty.")
        self.play(FadeIn(cap))
        self.play(ChangeDecimalToValue(count, 38), run_time=4, rate_func=rate_functions.linear)
        self.wait(2)
        end = caption("ME 193 · P6 step counter · trained in PyTorch, counted on a thigh", size=26)
        self.play(FadeOut(cap), run_time=0.5)
        self.play(FadeIn(end))
        self.wait(2)
```

- [ ] **Step 3: Write `docs/animation/render.sh`**

```bash
#!/usr/bin/env bash
# Render the explainer: 1080p for the repo/YouTube and a small 480p cut for Notion (free plan: 5 MB per upload).
set -euo pipefail
cd "$(dirname "$0")"
MANIM="${MANIM:-../../.venv/bin/manim}"
"$MANIM" -qh --media_dir media -o stepnet_explainer.mp4 stepnet_explainer.py StepNetExplainer
"$MANIM" -ql --media_dir media -o stepnet_explainer_small.mp4 stepnet_explainer.py StepNetExplainer
cp media/videos/stepnet_explainer/1080p60/stepnet_explainer.mp4 . 2>/dev/null || cp media/videos/stepnet_explainer/1080p30/stepnet_explainer.mp4 .
cp media/videos/stepnet_explainer/480p15/stepnet_explainer_small.mp4 .
ls -la stepnet_explainer*.mp4
```
`chmod +x docs/animation/render.sh`; add `docs/animation/media/` and `docs/animation/stepnet_explainer.mp4` to `.gitignore` (the small cut is committed).

- [ ] **Step 4: Dry run, then render**

Run: `cd docs/animation && ../../.venv/bin/manim -ql --dry_run stepnet_explainer.py StepNetExplainer` → Expected: finishes with no exception (fix any API mismatch here, e.g. a renamed import, before the real render).
Run: `./render.sh` → Expected: two MP4s listed; the small one under 5 MB. Open `stepnet_explainer.mp4` and watch it once: every caption readable, nothing overlapping the title. Fix and re-render as needed.

- [ ] **Step 5: Commit**

```bash
git add docs/animation/requirements.txt docs/animation/stepnet_explainer.py docs/animation/render.sh docs/animation/stepnet_explainer_small.mp4 .gitignore
git commit -m "P6-Gait-Wearable: Manim explainer of the step counter (sensors, decimation, causal convs, rate to count, loss, back-propagation, ship to board) + 480p cut for Notion"
```

---

### Task 13: README section, Notion drafts, spec/plan sync

**Files:**
- Modify: `README.md`
- Create: `docs/notion-steps.md`
- Modify: `docs/superpowers/specs/2026-11-01-step-counter-design.md` (Revision note if anything changed)

- [ ] **Step 1: README** — add a section after "Race mode", about 20 lines, using the real numbers from `docs/steps_results.md`:

```markdown
## Step counter (StepNet)

A small causal 1-D conv net, trained in PyTorch on all six IMU channels, outputs a step rate; the count is the area
under it. The board runs the exported weights (`models/stepnet.json`) in plain Python and the **Steps** tab shows the
count with its measured uncertainty (`38 ± 2`).

| | |
|---|---|
| Input | ax ay az gx gy gz, 100 → 25 Hz, standardised |
| Net | conv 6→12 (dil 1) → 12→12 (dil 2) → 12→12 (dil 4), ReLU, kernel 5; softplus head → steps/s; ~1,800 weights |
| Training | N counted takes (typed count in the file name), smooth-L1 on the integrated count, rotation/gain/stretch/pairing augmentation |
| Leave-one-recording-out | MAE x.x steps, MAPE y.y %, 95th pct z % (the live ±) |

Record counted takes on the Steps tab (count out loud, type the number on Stop), then:

    scripts/pull_data.sh
    .venv/bin/python train/steps.py --report      # LORO, seeds, figures, docs/steps_results.md, models/stepnet.json
    scripts/deploy.sh

How it works, animated: `docs/animation/stepnet_explainer_small.mp4` (`docs/animation/render.sh` rebuilds it).
```

- [ ] **Step 2: `docs/notion-steps.md`** — drafts for the new Notion page, in the voice of the P6 page: an intro paragraph; **three paragraphs on how the net works** (1: what goes in and the decimate/standardise step and why every sensor; 2: the causal dilated convs, Softplus rate, area = count, why a rate not a classifier; 3: training from typed totals, the loss, augmentation, back-propagation in PyTorch, export to plain Python on the board); "How good is it?" with the LORO table and the four figures, in plain words (what MAE/MAPE/95th percentile mean for a user); captions for the demo video and the animation; GitHub link `https://github.com/rafaeshafi/Me193/tree/main/P6-Gait-Wearable`; a heading "Reflection questions" left empty for the user. Keep every paragraph under 120 words.

- [ ] **Step 3: Spec sync** — if the implementation deviated (e.g. no numpy path on the board; the take keeps recording while the count is typed), add a short "Revision 1" list at the end of the spec.

- [ ] **Step 4: Full test run and commit**

Run: `.venv/bin/python -m pytest tests -q` → Expected: all pass.
```bash
git add README.md docs/notion-steps.md docs/superpowers/specs/2026-11-01-step-counter-design.md
git commit -m "P6-Gait-Wearable: README step-counter section, Notion drafts for the step-counter page, spec revision notes"
```

Push only when the user asks.
