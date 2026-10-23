import csv
import json
import math
import random

import numpy as np

from features import MODEL1_FEATURES, MODEL2_FEATURES, compute
from models import Tree
from synth import synth
from train import LABELS, dataset, export_tree, fit, matrix, person_of, windows


def write_session(path, kinds, seconds=6.0, who=""):
    random.seed(0)
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
    # 400 labelled rows at a 50-row hop give (400 - 200) / 50 + 1 = 5 windows; the BIKE block starts at row 500, a hop multiple
    assert labels.count("WALK") == 5 and labels.count("BIKE") == 5   # 4 s at 2 s windows / 0.5 s hop = 5 whole windows


def test_person_of_filename():
    assert person_of("20261023-120000-guided-ann.csv") == "ann"
    assert person_of("20261023-120000-walk.csv") == "me"
    assert person_of("20261023-120000-guided.csv") == "me"
    assert person_of("20261023-120000-walk-ann-2.csv") == "ann"
    assert person_of("20261023-120000-walk-2.csv") == "me"


def test_export_matches_sklearn_and_round_trips(tmp_path):
    for kind in LABELS:
        write_session(tmp_path / ("20261023-1200%02d-%s.csv" % (LABELS.index(kind), kind.lower())), [kind], seconds=8.0)
    feats, y, groups, persons = dataset(sorted(tmp_path.glob("*.csv")))
    assert set(y) == set(LABELS) and len(feats) == len(y) == len(groups) == len(persons)
    jittered = [{k: v * (1 + 1e-3 * ((i % 7) - 3)) for k, v in f.items()} for i, f in enumerate(feats)]
    for names, depth, leaf in ((MODEL1_FEATURES, 3, 1), (MODEL2_FEATURES, 5, 1), (MODEL1_FEATURES, 3, 5), (MODEL2_FEATURES, 5, 5)):
        clf, X = fit(feats, y, names, depth=depth, leaf=leaf)
        tree = Tree(export_tree(clf, names))
        for rows in (feats, jittered):
            ours = [tree.predict(f) for f in rows]
            assert ours == list(clf.predict(matrix(rows, names)))
        spec = json.loads(json.dumps(export_tree(clf, names)))
        assert spec["features"] == names and spec["classes"] == list(clf.classes_)
    # Model 2 must separate all four synthetic activities perfectly; Model 1 is allowed to miss BIKE
    clf2, X2 = fit(feats, y, MODEL2_FEATURES, depth=5, leaf=1)
    assert (clf2.predict(X2) == np.array(y)).all()


def test_windows_do_not_straddle_a_time_gap(tmp_path):
    p = tmp_path / "20261023-120000-walk.csv"
    write_session(p, ["WALK"], seconds=6.0)
    with open(p) as f:
        lines = list(csv.reader(f))
    for row in lines[1 + 300:]:
        row[0] = "%.3f" % (float(row[0]) + 1.0)          # FIFO reset: t jumps 1.0 s after 3 s
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(lines)
    labels = [lab for lab, _ in windows(p)]
    assert len(labels) == 6                                # 2 x ((300 - 200) / 50 + 1)


def test_windows_survive_arrival_jitter(tmp_path):
    p = tmp_path / "20261023-120000-walk.csv"
    write_session(p, ["WALK"], seconds=30.0)
    clean = len(list(windows(p)))
    with open(p) as f:
        lines = list(csv.reader(f))
    for i, row in enumerate(lines[1:]):
        row[0] = "%.3f" % (float(row[0]) + 0.03 * math.sin(i // 10))   # one shift per 10-row batch
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(lines)
    assert clean == 57                                     # (3000 - 200) / 50 + 1
    assert len(list(windows(p))) == clean
