import json

import train


def test_synthetic_training_end_to_end(tmp_path):
    meta = train.main(["--synth", "--synth-per-class", "12", "--epochs", "2", "--out-dir", str(tmp_path)])
    assert meta["n_train"] == 48 and meta["n_val"] == 24 and len(meta["history"]["augmented"]) == 2
    for rel in ("models/fingers.pt", "models/fingers.json", "models/meta.json", "docs/results.md",
                "docs/figures/train_curves.png", "docs/figures/confusion.png", "docs/figures/samples.png",
                "docs/figures/augment.png"):
        assert (tmp_path / rel).exists(), rel
    spec = json.load(open(tmp_path / "models" / "fingers.json"))
    assert spec["input"]["size"] == 64 and len(spec["layers"]) == 14


def test_ablation_writes_both_runs(tmp_path):
    meta = train.main(["--synth", "--synth-per-class", "6", "--epochs", "1", "--ablate", "--out-dir", str(tmp_path)])
    assert set(meta["history"]) == {"augmented", "no augmentation"} and meta["run"] == "augmented"
    assert (tmp_path / "docs" / "figures" / "train_curves_noaug.png").exists()
    assert "no augmentation" in (tmp_path / "docs" / "results.md").read_text()
