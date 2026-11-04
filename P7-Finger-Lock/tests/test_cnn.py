import json

import numpy as np
import torch

from export import export, save_checkpoint, to_spec
from lock.cnn_np import NumpyNet, conv2d, maxpool2, softmax
from model import FingerNet


def test_conv_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 10, 10)).astype(np.float32)
    w = rng.normal(size=(5, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=5).astype(np.float32)
    ref = torch.nn.functional.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b), padding=1)[0].numpy()
    assert np.allclose(conv2d(x, w, b), ref, atol=1e-4)


def test_maxpool_matches_torch():
    x = np.random.default_rng(1).normal(size=(2, 8, 8)).astype(np.float32)
    ref = torch.nn.functional.max_pool2d(torch.from_numpy(x)[None], 2)[0].numpy()
    assert np.array_equal(maxpool2(x), ref)


def test_whole_net_parity_and_export_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = FingerNet().eval()
    spec = to_spec(model, 0.4, 0.2)
    net = NumpyNet(spec)
    assert net.n_params() == model.n_params() == 285894
    rng = np.random.default_rng(2)
    for _ in range(3):
        x = rng.random((64, 64), dtype=np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy((x - 0.4) / 0.2)[None, None])[0].numpy()
        assert np.allclose(net.logits(x), ref, atol=1e-4)
        cls, p = net.predict(x)
        assert cls == int(ref.argmax()) and abs(sum(p) - 1) < 1e-6
    pt = tmp_path / "f.pt"
    save_checkpoint(model, 0.4, 0.2, pt, {"trained": "today"})
    out = tmp_path / "f.json"
    export(pt, out)
    net2 = NumpyNet.load(out)
    x = rng.random((64, 64), dtype=np.float32)
    assert np.allclose(net2.logits(x), net.logits(x), atol=1e-5)
    assert json.load(open(out))["trained"] == "today"


def test_softmax():
    p = softmax([1.0, 2.0, 3.0])
    assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 2
