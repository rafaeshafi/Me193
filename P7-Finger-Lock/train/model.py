"""FingerNet: three 3x3 convolutions (16, 32, 64 filters), each followed by ReLU and 2x2 max-pooling, then one hidden
linear layer and a 6-way output (0..5 fingers). Cross-entropy trains it. About 286k parameters."""
import torch
from torch import nn

CLASSES = [0, 1, 2, 3, 4, 5]
CHANNELS = (16, 32, 64)
HIDDEN = 64


class FingerNet(nn.Module):
    def __init__(self, size=64, channels=CHANNELS, hidden=HIDDEN, n_classes=len(CLASSES), dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        side = size // (2 ** len(channels))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in * side * side, hidden), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        self.size, self.channels = size, tuple(channels)

    def forward(self, x):
        return self.classifier(self.features(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def layer_specs(model):
    """The model as the plain list of layers lock/cnn_np.py runs (conv/relu/maxpool/flatten/linear/dropout)."""
    out = []
    for m in list(model.features) + list(model.classifier):
        if isinstance(m, nn.Conv2d):
            out.append({"type": "conv", "w": m.weight.detach().cpu().numpy().tolist(), "b": m.bias.detach().cpu().numpy().tolist()})
        elif isinstance(m, nn.Linear):
            out.append({"type": "linear", "w": m.weight.detach().cpu().numpy().tolist(), "b": m.bias.detach().cpu().numpy().tolist()})
        elif isinstance(m, nn.ReLU):
            out.append({"type": "relu"})
        elif isinstance(m, nn.MaxPool2d):
            out.append({"type": "maxpool"})
        elif isinstance(m, nn.Flatten):
            out.append({"type": "flatten"})
        elif isinstance(m, nn.Dropout):
            out.append({"type": "dropout"})
        else:
            raise TypeError(f"unsupported layer {m}")
    return out
