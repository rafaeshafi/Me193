import json
import random

import pytest

torch = pytest.importorskip("torch")

from stepnet import DT, StepNet          # noqa: E402
from stepnet_torch import StepNetTorch  # noqa: E402


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
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-4
    assert cnt == pytest.approx(sum(got) * DT, abs=1e-3)


import numpy as np                                                  # noqa: E402

from synth import steps_take                                        # noqa: E402
from steps import Recording, augment_pair, fit, gain, loro, metrics, normalisation, predict, rotate, stretch, to_array  # noqa: E402


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
    assert 0.8 * x.shape[1] < s.shape[1] < 1.2 * x.shape[1]
    a, b = synthetic_recordings()[:2]
    px, py = augment_pair(a, b)
    assert px.shape[1] == a.x.shape[1] + b.x.shape[1] and py == a.steps + b.steps


def test_fit_learns_synthetic_counts_and_loro_runs():
    recs = synthetic_recordings()
    mean, std = normalisation(recs)
    assert mean.shape == (6,) and (std > 0).all()
    model, losses = fit(recs, epochs=60, seed=0)
    assert losses[-1] < losses[0]
    worst = max(abs(predict(model, mean, std, r.x) - r.steps) for r in recs)
    assert worst < 6, worst                                           # a tiny net, synthetic data, 60 epochs
    res = loro(recs[:4], epochs=15, seed=0)
    assert len(res) == 4 and {r["name"] for r in res} == {r.name for r in recs[:4]}
    m = metrics(res)
    assert set(m) >= {"n", "mae_steps", "mape_pct", "bias_steps", "ci95_pct", "pm_pct95", "by_activity"}
