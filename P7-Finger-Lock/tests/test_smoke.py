from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_layout_exists():
    for rel in ("config.json", "lock/policy.py", "lock/cnn_np.py", "train/train.py", "app/app.yaml",
                "app/sketch/sketch.yaml", "scripts/deploy.sh", "docs/movie/frames.py"):
        assert (ROOT / rel).exists(), rel
