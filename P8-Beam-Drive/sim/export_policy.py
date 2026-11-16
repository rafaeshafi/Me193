"""Save a trained policy as plain numbers (policy.npz) so the real robot can run it with numpy alone.

    python sim/export_policy.py runs/onshape/logs/rsl_rl/beam_straight/<date>/model_299.pt

Writes real/policy.npz: the three layers of the actor network and the observation scaling it was trained with.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "real" / "policy.npz"


def export(checkpoint, out=OUT):
    actor = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    arrays = {name: actor[key].numpy().astype(np.float64) for name, key in {
        "mean": "obs_normalizer._mean", "std": "obs_normalizer._std",
        "w0": "mlp.0.weight", "b0": "mlp.0.bias", "w1": "mlp.2.weight", "b1": "mlp.2.bias",
        "w2": "mlp.4.weight", "b2": "mlp.4.bias"}.items()}
    np.savez(out, **arrays)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_file():
        sys.exit("usage: export_policy.py model_299.pt   (the checkpoint file must exist)")
    print(f"wrote {export(sys.argv[1])}")
