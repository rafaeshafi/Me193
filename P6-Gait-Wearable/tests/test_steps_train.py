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
