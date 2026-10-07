"""Store: players, finished games and the leaderboard, in one SQLite file (gitignored under data/)."""

import threading

import pytest

from pingpong import store


def game(**kw):
    base = dict(mode="survival", level="Club", target=7, streak=5, record=5, player_points=0, cpu_points=0,
                winner=None, hits=5, misses=1, faults=0, max_kmh=31.0, duration_s=42.0, started_at_ns=1, source="live")
    base.update(kw)
    return base


def test_players_are_created_once_and_names_are_case_insensitive(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    a, b, c = db.player_id("Rafae"), db.player_id("rafae"), db.player_id("Maya")
    assert a == b and a != c
    assert db.players() == ["maya", "rafae"]


def test_a_finished_game_is_stored_and_survives_reopening_the_file(tmp_path):
    path = tmp_path / "pp.db"
    db = store.Store(path)
    db.record_game("rafae", game(streak=9, record=9))
    db.close()
    again = store.Store(path)
    assert again.player_stats("rafae")["games"] == 1 and again.player_stats("rafae")["best_streak"] == 9


def test_the_survival_leaderboard_ranks_each_player_by_their_best_streak(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    for name, streak in (("rafae", 12), ("rafae", 7), ("maya", 15), ("omar", 3), ("zed", 15)):
        db.record_game(name, game(streak=streak, record=streak))
    board = db.leaderboard("survival", limit=3)
    assert board[0][1] == 15 and board[1][1] == 15 and board[2] == ("rafae", 12)          # one row per player
    assert [row[0] for row in board[:2]] == ["maya", "zed"]                                # a tie: first to set it wins
    assert len(db.leaderboard("survival", limit=10)) == 4


def test_the_leaderboard_can_be_cut_by_level(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    db.record_game("rafae", game(level="Rookie", streak=20, record=20))
    db.record_game("maya", game(level="Pro", streak=6, record=6))
    assert db.leaderboard("survival", level="Pro") == [("maya", 6)]
    assert db.leaderboard("survival", level="Rookie") == [("rafae", 20)]


def test_the_match_leaderboard_counts_wins(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    for name, winner in (("rafae", "player"), ("rafae", "player"), ("rafae", "cpu"), ("maya", "player")):
        db.record_game(name, game(mode="match", winner=winner, player_points=7 if winner == "player" else 3,
                                  cpu_points=2 if winner == "player" else 7, streak=2, record=2))
    assert db.leaderboard("match") == [("rafae", 2), ("maya", 1)]


def test_only_live_games_count_for_the_leaderboard(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    db.record_game("rafae", game(streak=99, record=99, source="fake"))
    db.record_game("rafae", game(streak=4, record=4, source="live"))
    assert db.leaderboard("survival") == [("rafae", 4)]
    assert db.player_stats("rafae")["games"] == 2                                          # it is still on file


def test_player_stats_summarise_the_history(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    db.record_game("rafae", game(streak=4, record=4, max_kmh=20.0, hits=4))
    db.record_game("rafae", game(streak=10, record=10, max_kmh=35.0, hits=10))
    db.record_game("rafae", game(mode="match", winner="player", streak=3, record=3, hits=9))
    s = db.player_stats("rafae")
    assert s["games"] == 3 and s["best_streak"] == 10 and s["max_kmh"] == 35.0 and s["matches_won"] == 1
    assert db.player_stats("nobody")["games"] == 0


def test_two_threads_can_write_at_once(tmp_path):
    db = store.Store(tmp_path / "pp.db")

    def writer(name):
        for i in range(30):
            db.record_game(name, game(streak=i, record=i))

    threads = [threading.Thread(target=writer, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert db.player_stats("a")["games"] == 30 and db.player_stats("b")["games"] == 30


def test_an_unknown_mode_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        store.Store(tmp_path / "pp.db").leaderboard("chess")


def test_the_board_text_lists_modes_and_levels(tmp_path):
    db = store.Store(tmp_path / "pp.db")
    db.record_game("rafae", game(streak=12, record=12))
    db.record_game("rafae", game(mode="match", winner="player"))
    text = store.format_board(db)
    assert "SURVIVAL" in text and "MATCH" in text and "rafae" in text and "12" in text
    assert "no games yet" in store.format_board(store.Store(tmp_path / "empty.db"))
