import numpy as np

import synth
from dataset import Fingers, augment, scan, split_by_session, stats, load_gray


def test_synthetic_hands_have_more_skin_with_more_fingers():
    fist, five = synth.hand(0, np.random.default_rng(5)), synth.hand(5, np.random.default_rng(5))  # same background
    assert fist.shape == (128, 128, 3) and fist.dtype == np.uint8
    skin = lambda im: int((np.abs(im.astype(int) - np.median(im, axis=(0, 1))).sum(axis=2) > 60).sum())
    assert skin(five) > skin(fist) > 0


def test_write_scan_split(tmp_path):
    n = synth.write(tmp_path, per_class=12, seed=1)
    items = scan(tmp_path)
    assert len(items) == n == 72
    assert {l for _, l, _ in items} == set(range(6))
    tr, va, val_sessions = split_by_session(items)
    assert val_sessions == ["synth2"] and len(va) == 24 and len(tr) == 48
    tr2, va2, _ = split_by_session(items, ["synth0", "synth1"])
    assert len(va2) == 48
    xs = load_gray(va, 64)
    assert xs.shape == (24, 64, 64) and xs.dtype == np.float32


def test_augment_keeps_shape_range_and_changes_picture():
    x = np.random.default_rng(0).random((64, 64), dtype=np.float32)
    y = augment(x, np.random.default_rng(1))
    assert y.shape == x.shape and y.dtype == np.float32 and 0 <= y.min() and y.max() <= 1
    assert not np.allclose(x, y)


def test_dataset_items():
    xs = np.random.default_rng(0).random((5, 64, 64), dtype=np.float32)
    mean, std = stats(xs)
    ds = Fingers(xs, [0, 1, 2, 3, 4], mean, std, augmented=True)
    x, y = ds[3]
    assert tuple(x.shape) == (1, 64, 64) and y == 3
