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
