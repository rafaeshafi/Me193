"""The numpy policy gives the same commands as the torch network it was exported from."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "sim"), str(ROOT / "real")]

import export_policy  # noqa: E402
import policy_np  # noqa: E402


def test_numpy_matches_torch(tmp_path):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.ELU(), torch.nn.Linear(64, 64), torch.nn.ELU(), torch.nn.Linear(64, 2))
    mean, std = torch.randn(1, 11), torch.rand(1, 11) + 0.1
    actor = {f"mlp.{k}": v for k, v in net.state_dict().items()}
    actor.update({"obs_normalizer._mean": mean, "obs_normalizer._std": std})
    torch.save({"actor_state_dict": actor}, tmp_path / "model.pt")
    export_policy.export(tmp_path / "model.pt", tmp_path / "policy.npz")
    policy = policy_np.Policy(tmp_path / "policy.npz")
    for _ in range(20):
        obs = torch.randn(11) * 3
        want = net((obs - mean[0]) / (std[0] + 1e-2)).detach().numpy()
        assert np.allclose(policy.act(obs.numpy()), want, atol=1e-5)


def test_observation_order():
    obs = policy_np.make_obs(0.5, 0.1, (1.0, 2.0), (0.3, 0.4), (0.7, 0.8))
    assert obs.shape == (11,)
    assert np.allclose(obs[:5], [np.sin(0.5), np.cos(0.5), 0.1, 1.0, 2.0])
    assert np.allclose(obs[5:9], [np.sin(0.3), np.sin(0.4), np.cos(0.3), np.cos(0.4)])
    assert np.allclose(obs[9:], [0.7, 0.8])
