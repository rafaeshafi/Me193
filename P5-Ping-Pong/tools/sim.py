"""Play the whole game loop with no sensors: how do the levels and your swing strength play out?

A scripted player returns every ball with a chosen swing strength (and, optionally, a chosen sloppiness)
against the computer in Match, at every level.  The table shows how often the computer misses a ball it
has to return, how often the player faults, and how long rallies last, which is what to look at when a
level feels too easy or too hard.  No hardware, no network: it is the same rules, judge, shot and policy
code the game uses, on a simulated clock.

Usage:
    ./pp sim                              # all levels, soft / medium / hard swings, a perfect player
    ./pp sim --points 200 --seed 7
    ./pp sim --quality 0.4                # a sloppier player: hard swings start to fault
    ./pp sim --levels 1,3
    ./pp sim --selftest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingpong import app, shot  # noqa: E402
from pingpong import levels as level_defs  # noqa: E402

SWINGS = (("soft", 350.0), ("medium", 700.0), ("hard", 1200.0))        # peak gyro, dps


def _offsets(level, quality):
    """Hand and timing offsets that give a hit quality near `quality` (0 = barely inside the gates)."""
    miss = 0.9 * (1.0 - quality)
    return miss * level.radius_sw, miss * level.late_s


def simulate(levels=(1, 2, 3), swings=SWINGS, points=60, seed=1, quality=1.0):
    rows = []
    for tag in levels:
        du, timing = _offsets(level_defs.LEVELS[tag], quality)
        for name, w_pk in swings:
            session = app.make_session(level=tag, mode="match", target=10_000, seed=seed * 100 + tag)
            app.play_until(session, lambda s: s.game.player_points + s.game.cpu_points >= points, w_pk=w_pk,
                           dt=0.02, max_sim_s=1e7, du=du, timing_s=timing)
            game, stats = session.game, session.game_stats()
            balls = max(1, stats["hits"] + stats["faults"] + stats["misses"])
            rows.append({"level": tag, "swing": name, "w_pk": w_pk,
                         "kmh": shot.kmh(shot.out_speed(shot.swing_strength(w_pk, 300.0, 1200.0))),
                         "p_cpu_miss": game.player_points / max(1, stats["hits"]),
                         "fault_rate": stats["faults"] / balls, "rally": stats["hits"] / max(1, game.player_points)})
    return rows


def format_table(rows, points, quality):
    lines = [f"Match simulation: {points} points per row; the player returns every ball, "
             + ("perfectly placed and timed" if quality >= 1.0 else f"with hit quality about {quality:.0%}"),
             f"{'level':<8}{'swing':<9}{'km/h':>6}  {'CPU misses':>10}  {'faults':>7}  {'balls per point':>16}"]
    for r in rows:
        lines.append(f"{level_defs.LEVELS[r['level']].name:<8}{r['swing']:<9}{r['kmh']:>6.0f}  {r['p_cpu_miss']:>10.0%}  "
                     f"{r['fault_rate']:>7.0%}  {r['rally']:>16.1f}")
    lines.append("CPU misses = chance the computer fails to return a given ball; balls per point = mean hits you "
                 "make before a point ends.")
    return "\n".join(lines)


def _selftest():
    rows = simulate(levels=(1, 3), swings=(("soft", 350.0), ("hard", 1200.0)), points=30, seed=1)
    soft1 = next(r for r in rows if r["level"] == 1 and r["swing"] == "soft")
    hard1 = next(r for r in rows if r["level"] == 1 and r["swing"] == "hard")
    hard3 = next(r for r in rows if r["level"] == 3 and r["swing"] == "hard")
    assert hard1["p_cpu_miss"] > soft1["p_cpu_miss"] and hard1["p_cpu_miss"] > hard3["p_cpu_miss"], rows
    print("sim selftest OK: harder swings and lower levels make the computer miss more")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--quality", type=float, default=1.0, help="hit quality of the scripted player, 0..1")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        tags = tuple(int(x) for x in args.levels.split(","))
        assert all(t in level_defs.LEVELS for t in tags) and 0.0 <= args.quality <= 1.0
    except (ValueError, AssertionError):
        print(f"--levels must be numbers from {sorted(level_defs.LEVELS)} and --quality between 0 and 1", file=sys.stderr)
        return 2
    print(format_table(simulate(tags, SWINGS, args.points, args.seed, args.quality), args.points, args.quality))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
