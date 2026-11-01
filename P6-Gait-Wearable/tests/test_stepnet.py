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
    i = 0
    while i < len(raw):
        n = rng.randint(1, 37)                         # batches of random size, like the real stream
        ctr.on_samples(raw[i:i + n])
        ctr.process()
        i += n
    got = list(ctr.wave_all)
    assert len(got) == len(offline)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, offline))
    assert ctr.count == pytest.approx(sum(offline) * DT)


def test_reset_keeps_or_clears_the_right_things():
    ctr = StepCounter(StepNet(seed_stepnet()))
    ctr.on_samples(rnd_samples(400))
    ctr.process()
    assert ctr.count > 0 and len(ctr._ctx) == 28
    c = ctr.count
    ctr.reset_context()                 # sensor stream restarted: forget the past, keep the count
    assert ctr.count == c and ctr._ctx == [] and ctr._rem == []
    ctr.reset()                         # the wearer tapped Reset
    assert ctr.count == 0.0 and ctr.rate == 0.0 and ctr.snapshot()["wave"] == []


def test_snapshot_has_the_live_fields_and_the_half_width():
    ctr = StepCounter(StepNet(seed_stepnet(pm_pct95=10.0)))
    s = ctr.snapshot()
    assert s == {"count": 0, "exact": 0.0, "rate": 0.0, "spm": 0, "pm": 0, "pm_pct": 10.0, "wave": []}
    ctr.count = 38.4
    s = ctr.snapshot()
    assert s["count"] == 38 and s["pm"] == 4          # ceil(38 * 10 / 100)
    ctr.count = 1.2
    assert ctr.snapshot()["pm"] == 1                  # never less than 1 once counting


def test_load_validates(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(seed_stepnet()))
    assert StepNet.load(p).pm_pct95 == 100.0
    bad = seed_stepnet()
    bad["layers"][1]["w"] = bad["layers"][1]["w"][:-1]   # wrong number of input channels
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        StepNet.load(p)
