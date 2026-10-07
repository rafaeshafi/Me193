import random

import pytest

from pingpong.scoring import ScoreTracker


def test_streak_counts_consecutive_valid_hits_and_resets_on_rally_end():
    t = ScoreTracker()
    for i in range(1, 4):
        assert t.on_valid_hit(i) is True
    assert (t.streak, t.record) == (3, 3)
    assert t.end_rally() == 3
    assert (t.streak, t.record) == (0, 3)
    t.on_valid_hit(4)
    assert (t.streak, t.record) == (1, 3)


def test_record_is_the_best_streak_and_never_decreases():
    t = ScoreTracker()
    for i in range(5):
        t.on_valid_hit(i)
    t.end_rally()
    for i in range(5, 7):
        t.on_valid_hit(i)
    t.end_rally()
    assert t.record == 5


def test_hit_ids_are_idempotent_and_must_increase():
    t = ScoreTracker()
    assert t.on_valid_hit(10) is True
    assert t.on_valid_hit(10) is False       # a duplicate delivery of the same hit
    assert t.on_valid_hit(9) is False        # an older id
    assert t.streak == 1


def test_record_session_value_is_max_of_record_and_current_streak():
    t = ScoreTracker(scope="record_session")
    assert t.value() == 0
    for i in range(1, 4):
        t.on_valid_hit(i)
    assert t.value() == 3
    t.end_rally()
    assert t.value() == 3                    # holds after a miss
    t.on_valid_hit(10)
    assert t.value() == 3


def test_record_alltime_is_seeded_from_the_profile_best():
    t = ScoreTracker(scope="record_alltime", seed_record=12)
    assert t.value() == 12
    for i in range(1, 4):
        t.on_valid_hit(i)
    assert t.value() == 12
    assert t.record == 12


def test_live_streak_value_is_the_running_count():
    t = ScoreTracker(scope="live_streak")
    for i in range(1, 4):
        t.on_valid_hit(i)
    assert t.value() == 3
    t.end_rally()
    assert t.value() == 0


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        ScoreTracker(scope="best_ever")


def test_record_is_monotone_under_random_sequences():
    rng = random.Random(7)
    for _ in range(200):
        t = ScoreTracker()
        hit_id, last_record = 0, 0
        for _ in range(rng.randint(5, 60)):
            if rng.random() < 0.2:
                t.end_rally()
            else:
                hit_id += rng.choice([1, 1, 1, 2])
                t.on_valid_hit(hit_id)
                if rng.random() < 0.1:
                    t.on_valid_hit(hit_id)          # duplicate delivery
            assert t.record >= last_record
            assert t.streak <= t.record
            last_record = t.record
