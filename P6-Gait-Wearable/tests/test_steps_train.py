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


def test_figures_are_written(tmp_path):
    from steps_plots import figures
    recs = synthetic_recordings()
    model, losses, mean, std = fit(recs, epochs=3, seed=0)
    res = [{"name": r.name, "activity": r.activity, "who": r.who, "true": r.steps, "pred": r.steps + 1.0} for r in recs]
    figures(tmp_path, res, metrics(res), recs, model, mean, std, losses)
    for n in ("steps_scatter.png", "steps_errors.png", "steps_rate.png", "steps_training.png"):
        assert (tmp_path / n).stat().st_size > 1000
