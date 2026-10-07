import pytest

from pingpong import fixtures
from pingpong.events import ImuSample


def sample(t, gx=0, ax=1000):
    return ImuSample(t_ns=t, g=(gx, 0, 0), a=(ax, 0, 0))


def test_a_take_round_trips_with_ints_and_tuples(tmp_path):
    takes = [sample(1_000, gx=5), sample(2_000, gx=-7)]
    path = fixtures.save_take(tmp_path, "soft", 3, takes, meta={"mass": "none"})
    loaded = fixtures.load_takes(path)
    assert len(loaded) == 1
    take = loaded[0]
    assert (take["label"], take["index"], take["meta"]) == ("soft", 3, {"mass": "none"})
    assert take["samples"] == takes
    assert isinstance(take["samples"][0].g, tuple)


def test_saving_appends_so_earlier_takes_survive(tmp_path):
    fixtures.save_take(tmp_path, "hard", 0, [sample(1)])
    path = fixtures.save_take(tmp_path, "hard", 1, [sample(2)])
    assert [t["index"] for t in fixtures.load_takes(path)] == [0, 1]


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert fixtures.load_takes(tmp_path / "nope.jsonl") == []


def test_labels_are_restricted_to_safe_file_names(tmp_path):
    with pytest.raises(ValueError):
        fixtures.save_take(tmp_path, "../evil", 0, [sample(1)])


def test_load_all_groups_takes_by_label(tmp_path):
    fixtures.save_take(tmp_path, "soft", 0, [sample(1)])
    fixtures.save_take(tmp_path, "hard", 0, [sample(2)])
    grouped = fixtures.load_all(tmp_path)
    assert sorted(grouped) == ["hard", "soft"]
    assert len(grouped["soft"]) == 1
