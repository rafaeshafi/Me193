"""Store: players, finished games and the leaderboard, in one SQLite file (gitignored under data/).

Names are case-insensitive.  Only LIVE games (real hub + camera) count for the leaderboard; fake
and practice games stay on file but never rank.  Survival ranks each player by their best streak
(ties go to whoever set it first), Match by number of wins.  One connection guarded by a lock:
games end a few times a minute at most, and WAL mode keeps each write to a millisecond or two.
"""

import sqlite3
import threading
import time
from pathlib import Path

import config
from pingpong import levels

MODES = ("survival", "match")
SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL REFERENCES players(id), played_at REAL NOT NULL,
    mode TEXT NOT NULL, level TEXT NOT NULL, target INTEGER, streak INTEGER NOT NULL, record INTEGER NOT NULL,
    player_points INTEGER, cpu_points INTEGER, winner TEXT, hits INTEGER, misses INTEGER, faults INTEGER,
    max_kmh REAL, duration_s REAL, source TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS games_rank ON games(mode, level, streak);
"""


def default_path():
    return config.HERE / "data" / "pingpong.db"


class Store:
    def __init__(self, path=None):
        path = Path(path or default_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()

    # --- players -----------------------------------------------------------------------------
    def player_id(self, name):
        key = str(name).strip().lower()
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO players(name, created_at) VALUES (?, ?)", (key, time.time()))
            self._db.commit()
            return self._db.execute("SELECT id FROM players WHERE name = ?", (key,)).fetchone()[0]

    def players(self):
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT name FROM players ORDER BY name")]

    # --- games ----------------------------------------------------------------------------------
    def record_game(self, name, g):
        pid = self.player_id(name)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO games(player_id, played_at, mode, level, target, streak, record, player_points, "
                "cpu_points, winner, hits, misses, faults, max_kmh, duration_s, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pid, time.time(), g["mode"], g["level"], g.get("target"), g["streak"], g["record"],
                 g.get("player_points"), g.get("cpu_points"), g.get("winner"), g.get("hits"), g.get("misses"),
                 g.get("faults"), g.get("max_kmh"), g.get("duration_s"), g.get("source", "live")))
            self._db.commit()
            return cur.lastrowid

    def leaderboard(self, mode, level=None, limit=5):
        """[(name, score)] best first: best streak (survival) or wins (match); live games only."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        where, args = "g.source = 'live' AND g.mode = ?", [mode]
        if level is not None:
            where, args = where + " AND g.level = ?", args + [level]
        if mode == "survival":
            sql = (f"SELECT p.name, b.best FROM (SELECT g.player_id, MAX(g.streak) AS best FROM games g WHERE {where} "
                   f"GROUP BY g.player_id) b JOIN players p ON p.id = b.player_id "
                   f"JOIN (SELECT g.player_id, g.streak, MIN(g.id) AS first_id FROM games g WHERE {where} "
                   f"GROUP BY g.player_id, g.streak) f ON f.player_id = b.player_id AND f.streak = b.best "
                   f"ORDER BY b.best DESC, f.first_id ASC LIMIT ?")
            args = args + args
        else:
            sql = (f"SELECT p.name, COUNT(*) AS wins FROM games g JOIN players p ON p.id = g.player_id "
                   f"WHERE {where} AND g.winner = 'player' GROUP BY g.player_id ORDER BY wins DESC, MIN(g.id) ASC LIMIT ?")
        with self._lock:
            return [(name, score) for name, score in self._db.execute(sql, args + [limit])]

    def player_stats(self, name):
        key = str(name).strip().lower()
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*), COALESCE(MAX(CASE WHEN g.source='live' THEN g.streak END), 0), "
                "COALESCE(MAX(CASE WHEN g.source='live' THEN g.max_kmh END), 0), "
                "COALESCE(SUM(CASE WHEN g.source='live' AND g.mode='match' AND g.winner='player' THEN 1 ELSE 0 END), 0) "
                "FROM games g JOIN players p ON p.id = g.player_id WHERE p.name = ?", (key,)).fetchone()
        return {"games": row[0], "best_streak": row[1], "max_kmh": row[2], "matches_won": row[3]}


def format_board(db, limit=5):
    """The leaderboard as text (./pp play --board)."""
    sections = []
    for mode, unit in (("survival", "best streak"), ("match", "wins")):
        rows = db.leaderboard(mode, limit=limit)
        if rows:
            sections.append(f"{levels.MODE_NAMES.get(mode, mode.upper())} ({unit})\n" + "\n".join(f"  {i + 1}. {n}  {s}" for i, (n, s) in enumerate(rows)))
    return "\n\n".join(sections) if sections else "no games yet: play one with ./pp play"
