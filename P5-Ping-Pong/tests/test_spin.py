"""spin: StandardScaler + LogisticRegression on the 12 swing features -> flat / top / back probabilities.

Per player, from ~12 labelled swings per class, and shipped only if cross-validated accuracy reaches 0.75;
otherwise the game stays flat and says so.  The IMU-driven tests run simulated swings through the real
swing detector, so "these features separate the classes" is tested, not assumed.
"""

import numpy as np
import pytest

from pingpong import fakerig, shot, spin


def centre(label, sep=2.0, dims=12, signal=4):
    """The class mean: the classes differ in `signal` of the 12 features, like real swing features do."""
    sign = {"flat": 0.0, "top": 1.0, "back": -1.0}[label]
    return np.r_[np.full(signal, sign * sep), np.zeros(dims - signal)]


def blobs(n=12, sep=2.0, seed=0, dims=12):
    rng = np.random.default_rng(seed)
    X = [rng.normal(centre(c, sep, dims), 0.5, dims) for c in spin.CLASSES for _ in range(n)]
    y = [c for c in spin.CLASSES for _ in range(n)]
    return X, y


def test_well_separated_swings_train_a_model_that_ships_and_predicts_the_class():
    X, y = blobs()
    model, report = spin.train(X, y)
    assert report["ship"] is True and report["cv_accuracy"] >= 0.9
    assert report["n_per_class"] == {"flat": 12, "top": 12, "back": 12}
    probs = model.probs(centre("top"))
    assert set(probs) == {"flat", "top", "back"} and sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=probs.get) == "top" and probs["top"] > 0.6


def test_labels_unrelated_to_the_features_do_not_ship():
    rng = np.random.default_rng(1)
    X = [rng.normal(0, 1, 12) for _ in range(36)]
    y = [spin.CLASSES[i % 3] for i in range(36)]
    _, report = spin.train(X, y)
    assert report["ship"] is False and report["cv_accuracy"] < 0.65


def test_too_few_swings_or_a_missing_class_is_an_error_not_a_weak_model():
    X, y = blobs(n=2)
    with pytest.raises(ValueError, match="at least"):
        spin.train(X, y)
    X, y = blobs(n=12)
    keep = [i for i, label in enumerate(y) if label != "back"]
    with pytest.raises(ValueError, match="back"):
        spin.train([X[i] for i in keep], [y[i] for i in keep])


def test_the_confusion_matrix_says_which_classes_get_mixed_up():
    X, y = blobs(sep=0.6)                                         # top and back overlap with flat
    _, report = spin.train(X, y)
    confusion = report["confusion"]
    assert set(confusion) == set(spin.CLASSES) and sum(sum(row.values()) for row in confusion.values()) == 36


def test_a_model_survives_saving_and_loading_by_player_name(tmp_path):
    X, y = blobs()
    model, report = spin.train(X, y)
    spin.save("Rafae", model, report, root=tmp_path)
    again = spin.load_for("rafae", root=tmp_path)
    feat = centre("back")
    assert again.probs(feat) == pytest.approx(model.probs(feat))
    assert spin.load_for("nobody", root=tmp_path) is None


def test_a_model_that_did_not_reach_the_bar_is_not_loaded_unless_forced(tmp_path):
    rng = np.random.default_rng(2)
    X, y = [rng.normal(0, 1, 12) for _ in range(36)], [spin.CLASSES[i % 3] for i in range(36)]
    model, report = spin.train(X, y)
    spin.save("rafae", model, report, root=tmp_path)
    assert spin.load_for("rafae", root=tmp_path) is None
    assert spin.load_for("rafae", root=tmp_path, force=True) is not None


def test_a_damaged_model_file_is_reported_not_silently_ignored(tmp_path):
    (tmp_path / "rafae").mkdir()
    (tmp_path / "rafae" / "spin.joblib").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="spin"):
        spin.load_for("rafae", root=tmp_path)


def test_the_probabilities_plug_straight_into_the_shot_rules():
    X, y = blobs()
    model, _ = spin.train(X, y)
    top_probs = model.probs(centre("top"))
    t, s, a = shot.spin_from_probs(top_probs, strength=1.0)
    assert t > 0.3 and a == pytest.approx(1.0)
    back_t, _, _ = shot.spin_from_probs(model.probs(centre("back")), strength=1.0)
    assert back_t < -0.3
    assert shot.spin_from_probs(model.probs(centre("flat")), strength=1.0)[0] == pytest.approx(0.0, abs=0.25)


def test_simulated_swings_through_the_real_detector_separate_into_the_three_spins():
    train_X, train_y = fakerig.spin_dataset(n_per_class=12, seed=1)
    model, report = spin.train(train_X, train_y)
    assert report["ship"] is True and report["cv_accuracy"] >= 0.85
    test_X, test_y = fakerig.spin_dataset(n_per_class=10, seed=99)              # swings the model never saw
    right = sum(max((p := model.probs(x)), key=p.get) == label for x, label in zip(test_X, test_y))
    assert right / len(test_y) >= 0.8


def test_the_feature_vector_has_the_twelve_documented_entries():
    X, _ = fakerig.spin_dataset(n_per_class=3, seed=2)
    assert all(len(x) == 12 for x in X)
