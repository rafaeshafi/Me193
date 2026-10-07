"""SpinCollector: guided collection of labelled swings (flat / top / back, interleaved)."""

import pytest

from pingpong import fakerig, spin
from pingpong.spinflow import SpinCollector
from pingpong.swing import SwingParams

GPD = fakerig.GPD


def collector(n=12, **kw):
    return SpinCollector(SwingParams(u_fwd=(1.0, 0.0, 0.0), gyro_per_dps=GPD, t_pk=250.0), n_per_class=n, **kw)


def feed_all(c, samples):
    for s in samples:
        c.feed_imu(s)


def test_the_classes_are_asked_for_in_turn_not_in_blocks():
    c = collector(n=2)
    assert c.order == ["flat", "top", "back", "flat", "top", "back"]
    assert "1 of 6" in c.prompt() and "FLAT" in c.prompt().upper()


def test_a_scripted_session_collects_every_swing_with_the_label_that_was_asked_for():
    c = collector(n=12)
    stream, _ = fakerig.spin_stream(c.order, seed=3)
    feed_all(c, stream)
    X, y = c.result()
    assert c.finished() and len(X) == 36 and y == c.order
    assert all(len(x) == 12 for x in X)
    model, report = spin.train(X, y)
    assert report["ship"] is True and report["cv_accuracy"] >= 0.85


def test_progress_and_prompts_follow_the_collection():
    c = collector(n=3)
    assert c.progress() == (0, 9)
    stream, _ = fakerig.spin_stream(c.order, seed=1)
    seen = []
    for s in stream:
        before = c.progress()[0]
        c.feed_imu(s)
        if c.progress()[0] != before:
            seen.append(c.prompt())
    assert c.progress() == (9, 9) and "TOP" in seen[0].upper() and c.prompt().startswith("Done")


def test_each_captured_swing_is_announced_with_its_class():
    c = collector(n=2)
    stream, _ = fakerig.spin_stream(c.order, seed=2)
    feed_all(c, stream)
    notes = c.take_notes()
    assert len(notes) >= 6 and "flat" in notes[0].lower() and "1/6" in notes[0] and c.take_notes() == []


def test_a_swing_too_weak_to_fire_the_detector_is_not_counted_so_the_prompt_waits_for_a_real_one():
    c = collector(n=2)
    weak, _ = fakerig.spin_stream(c.order[:1], seed=4, peak=60.0)               # far below T_PK
    feed_all(c, weak)
    assert c.progress() == (0, 6) and "1 of 6" in c.prompt()


def test_swings_after_the_last_one_are_ignored():
    c = collector(n=2)
    stream, _ = fakerig.spin_stream(c.order + ["flat"], seed=5)
    feed_all(c, stream)
    assert len(c.result()[0]) == 6


def test_the_result_is_not_available_until_the_collection_is_complete():
    with pytest.raises(ValueError, match="swings"):
        collector(n=2).result()
