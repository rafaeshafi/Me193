"""The two inference engines behind one call: predict(x) -> (class, probabilities). numpy for the board (and the
Mac, to prove it); PyTorch on the Mac, which can also give a saliency map."""
import numpy as np

from lock.cnn_np import NumpyNet


class NumpyEngine:
    name = "numpy"

    def __init__(self, json_path):
        self.net = NumpyNet.load(json_path)
        self.size, self.mean, self.std = self.net.size, self.net.mean, self.net.std

    def predict(self, x):
        return self.net.predict(x)

    def saliency(self, x):
        return None


class TorchEngine:
    name = "torch"

    def __init__(self, pt_path):
        import torch  # only the Mac has it
        from export import load_checkpoint  # train/ is on the path when tools run
        self.torch = torch
        self.model, ck = load_checkpoint(pt_path)
        self.size, self.mean, self.std = ck["size"], ck["mean"], ck["std"]

    def _tensor(self, x):
        return self.torch.from_numpy(((np.asarray(x, dtype=np.float32) - self.mean) / self.std))[None, None]

    def predict(self, x):
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(self._tensor(x))[0], 0).numpy()
        return int(p.argmax()), [float(v) for v in p]

    def saliency(self, x):
        """|d score_of_winner / d pixel|, 0..1: the pixels the net is counting."""
        t = self._tensor(x).requires_grad_(True)
        out = self.model(t)[0]
        out[int(out.argmax())].backward()
        g = t.grad[0, 0].abs().numpy()
        return g / (g.max() + 1e-9)


def make_engine(kind, models_dir):
    from pathlib import Path
    models_dir = Path(models_dir)
    if kind == "torch":
        return TorchEngine(models_dir / "fingers.pt")
    return NumpyEngine(models_dir / "fingers.json")
