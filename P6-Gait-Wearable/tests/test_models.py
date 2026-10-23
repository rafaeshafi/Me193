import json
from pathlib import Path

import pytest

from features import compute
from models import Tree, Vote
from synth import synth

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "features": ["peak_x"],
    "classes": ["RUN", "STAND", "WALK"],
    "root": {"f": "peak_x", "t": 1.15, "l": {"label": "STAND"},
             "r": {"f": "peak_x", "t": 2.2, "l": {"label": "WALK"}, "r": {"label": "RUN"}}},
}


def test_tree_predict_follows_thresholds():
    t = Tree(SPEC)
    assert t.predict({"peak_x": 1.0}) == "STAND"
    assert t.predict({"peak_x": 1.15}) == "STAND"   # <= goes left, like scikit-learn
    assert t.predict({"peak_x": 1.5}) == "WALK"
    assert t.predict({"peak_x": 3.0}) == "RUN"
    assert t.depth() == 2


def test_tree_load_from_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(SPEC))
    assert Tree.load(p).predict({"peak_x": 0.9}) == "STAND"


def test_seed_models_classify_the_synthetic_signals():
    m1 = Tree.load(ROOT / "models" / "model1.json")
    m2 = Tree.load(ROOT / "models" / "model2.json")
    for kind in ("STAND", "WALK", "RUN", "BIKE"):
        f = compute(synth(kind))
        assert m2.predict(f) == kind
        if kind != "BIKE":
            assert m1.predict(f) == kind
    assert m1.predict(compute(synth("BIKE"))) in ("STAND", "WALK")  # the honest limitation of one feature


def test_tree_rejects_a_split_on_an_unlisted_feature():
    bad = {"features": ["peak_x"], "classes": ["STAND", "WALK"],
           "root": {"f": "gyro_rms", "t": 1.0, "l": {"label": "STAND"}, "r": {"label": "WALK"}}}
    with pytest.raises(ValueError):
        Tree(bad)


def test_vote_majority_and_recent_tiebreak():
    v = Vote(3)
    assert v.push("WALK") == "WALK"
    assert v.push("RUN") == "RUN"            # 1-1 tie: the newest wins
    assert v.push("WALK") == "WALK"          # 2-1
    assert v.push("RUN") == "RUN"            # window is RUN WALK RUN
    assert v.push("STAND") == "STAND"        # WALK RUN STAND: three-way tie, newest wins
