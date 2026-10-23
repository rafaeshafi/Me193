import csv
import json
import random

import numpy as np

from features import MODEL1_FEATURES, MODEL2_FEATURES, compute
from models import Tree
from synth import synth
from train import LABELS, dataset, export_tree, fit, person_of, windows


def write_session(path, kinds, seconds=6.0, who=""):
    rows = []
    t = 0.0
    for kind in kinds:
        for s in synth(kind, seconds=seconds, phase=random.random()):
            rows.append(["%.3f" % t, kind] + ["%.4f" % v for v in s])
            t += 0.01
        for s in synth("STAND", seconds=1.0):          # unlabelled gap between activities
            rows.append(["%.3f" % t, ""] + ["%.4f" % v for v in s])
            t += 0.01
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "label", "ax", "ay", "az", "gx", "gy", "gz"])
        w.writerows(rows)


def test_windows_skip_mixed_and_unlabelled_chunks(tmp_path):
    p = tmp_path / "20261023-120000-guided.csv"
    write_session(p, ["WALK", "BIKE"], seconds=4.0)
    labels = [lab for lab, _ in windows(p)]
    assert set(labels) == {"WALK", "BIKE"}
    assert labels.count("WALK") == 5 and labels.count("BIKE") == 5   # 4 s at 2 s windows / 0.5 s hop = 5 whole windows


def test_person_of_filename():
    assert person_of("20261023-120000-guided-ann.csv") == "ann"
    assert person_of("20261023-120000-walk.csv") == "me"
    assert person_of("20261023-120000-guided.csv") == "me"


def test_export_matches_sklearn_and_round_trips(tmp_path):
    for kind in LABELS:
        write_session(tmp_path / ("20261023-1200%02d-%s.csv" % (LABELS.index(kind), kind.lower())), [kind], seconds=8.0)
    feats, y, groups, persons = dataset(sorted(tmp_path.glob("*.csv")))
    assert set(y) == set(LABELS) and len(feats) == len(y) == len(groups) == len(persons)
    for names, depth in ((MODEL1_FEATURES, 3), (MODEL2_FEATURES, 5)):
        clf, X = fit(feats, y, names, depth=depth, leaf=1)
        tree = Tree(export_tree(clf, names))
        ours = [tree.predict(f) for f in feats]
        assert ours == list(clf.predict(X))
        spec = json.loads(json.dumps(export_tree(clf, names)))
        assert spec["features"] == names and set(spec["classes"]) <= set(LABELS)
    # Model 2 must separate all four synthetic activities perfectly; Model 1 is allowed to miss BIKE
    clf2, X2 = fit(feats, y, MODEL2_FEATURES, depth=5, leaf=1)
    assert (clf2.predict(X2) == np.array(y)).all()
