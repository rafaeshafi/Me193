from pathlib import Path


def test_layout_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app" / "app.yaml").exists()
    assert (root / "app" / "sketch" / "sketch.yaml").exists()
