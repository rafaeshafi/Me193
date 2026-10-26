import json

import pytest

from race import COUNTDOWN_S, DISTANCES, GHOSTS, WALK_CAP, Race, ResultStore, SpeedModel, raw_speed, replay_distance, stride_m


class Clock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


def test_stride_and_raw_speed_from_the_recordings():
    assert stride_m(1.0) == pytest.approx(0.6)
    assert stride_m(2.2) == pytest.approx(1.2)
    assert stride_m(10.0) == 2.2 and stride_m(0.0) == 0.5          # clamped
    assert raw_speed("WALK", 108, 1.3) == pytest.approx(108 / 60 * 0.75)   # 1.35 m/s = 4.9 km/h
    assert raw_speed("RUN", 160, 2.2) == pytest.approx(3.2)                 # 11.5 km/h
    assert raw_speed("RUN", 200, 4.0) == pytest.approx(200 / 60 * 2.1)      # 25 km/h
    assert raw_speed("WALK", 200, 4.0) == pytest.approx(WALK_CAP)           # walking is capped
    for act in ("STAND", "BIKE", "NO IMU", "---"):
        assert raw_speed(act, 160, 2.2) == 0.0
    assert raw_speed("RUN", None, 2.2) == 0.0 and raw_speed("RUN", 160, None) == 0.0


def test_speed_model_smooths_towards_the_target():
    m = SpeedModel()
    assert m.update("RUN", 160, 2.2, now=0.0) == pytest.approx(3.2)   # first sample lands at once
    v = m.update("STAND", None, None, now=0.5)
    assert 0 < v < 3.2                                                 # decays, does not drop to zero at once
    for i in range(1, 12):
        v = m.update("STAND", None, None, now=0.5 + i * 0.5)
    assert v < 0.02                                                    # gone after a few seconds


def test_replay_distance_integrates_a_profile():
    prof = [(0.5, 2.0), (1.0, 2.0), (1.5, 4.0)]         # (elapsed s, speed m/s) samples
    assert replay_distance(prof, 0.0) == 0.0
    assert replay_distance(prof, 1.0) == pytest.approx(2.0)
    assert replay_distance(prof, 1.5) == pytest.approx(4.0)
    assert replay_distance(prof, 9.0) == pytest.approx(4.0)   # holds the last sample


def test_race_runs_countdown_distance_finish_and_saves(tmp_path):
    clk = Clock()
    store = ResultStore(tmp_path / "races.json")
    r = Race(store, clock=clk)
    assert r.snapshot()["phase"] == "idle" and r.snapshot()["leaderboard"] == []
    with pytest.raises(ValueError):
        r.start(123, "ann")
    r.start(100, "ann")
    s = r.snapshot()
    assert s["phase"] == "countdown" and s["countdown"] == pytest.approx(COUNTDOWN_S) and s["distance"] == 100
    clk.t += COUNTDOWN_S
    r.tick(5.0, clk.t)
    assert r.snapshot()["phase"] == "running"
    for _ in range(19):                        # 19 x 1 s at 5 m/s = 95 m
        clk.t += 1.0
        r.tick(5.0, clk.t)
    s = r.snapshot()
    assert s["phase"] == "running" and s["dist"] == pytest.approx(95.0) and s["elapsed"] == pytest.approx(19.0)
    ghosts = {g["name"]: g["dist"] for g in s["ghosts"]}
    assert ghosts["Jogger"] == pytest.approx(9 / 3.6 * 19) and ghosts["Sprinter"] == 100   # sprinter already home
    clk.t += 2.0                               # 10 m in 2 s: crosses the line after 1 s of it
    r.tick(5.0, clk.t)
    s = r.snapshot()
    assert s["phase"] == "finished" and s["dist"] == 100
    assert s["result"]["time"] == pytest.approx(20.0) and s["result"]["avg_kmh"] == pytest.approx(18.0)
    assert s["result"]["beat"] == ["Jogger", "Runner"] and s["result"]["rank"] == 1
    assert store.leaderboard(100)[0]["who"] == "ann"
    assert json.loads((tmp_path / "races.json").read_text())[0]["distance"] == 100


def test_second_race_gets_the_best_ghost_and_abort_resets(tmp_path):
    clk = Clock()
    store = ResultStore(tmp_path / "races.json")
    r = Race(store, clock=clk)
    r.start(100, "ann")
    clk.t += COUNTDOWN_S
    r.tick(4.0, clk.t)
    for _ in range(25):
        clk.t += 1.0
        r.tick(4.0, clk.t)
    assert r.snapshot()["phase"] == "finished"
    r.start(100, "ann")
    names = [g["name"] for g in r.snapshot()["ghosts"]]
    assert names == [n for n, _ in GHOSTS] + ["Your best"]
    clk.t += COUNTDOWN_S + 2.0
    r.tick(4.0, clk.t)
    best = next(g for g in r.snapshot()["ghosts"] if g["name"] == "Your best")
    assert 0 < best["dist"] <= 100
    r.abort()
    assert r.snapshot()["phase"] == "idle" and len(store.leaderboard(100)) == 1
    r.start(400, "bob")                       # another distance: no best yet, only the fixed ghosts
    assert [g["name"] for g in r.snapshot()["ghosts"]] == [n for n, _ in GHOSTS]


def test_result_store_persists_and_ranks(tmp_path):
    p = tmp_path / "races.json"
    s1 = ResultStore(p)
    s1.save(400, "ann", 120.0, [(0.5, 3.3)])
    s1.save(400, "bob", 90.0, [(0.5, 4.4)])
    s1.save(100, "bob", 20.0, [(0.5, 5.0)])
    s2 = ResultStore(p)                        # re-read from disk
    board = s2.leaderboard(400)
    assert [e["who"] for e in board] == ["bob", "ann"] and board[0]["avg_kmh"] == pytest.approx(16.0)
    assert s2.best(400, "ann")["time"] == 120.0 and s2.best(1000, "ann") is None
    assert DISTANCES == (100, 400, 1000)
