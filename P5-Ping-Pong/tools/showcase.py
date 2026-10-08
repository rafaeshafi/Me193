"""The intro as a video, and a picture of every screen: look at the game's new face without playing it.

Usage:
    ./pp showcase                         # writes data/showcase/intro.mp4 and a PNG of each screen
    ./pp showcase --out somewhere --size 1920x1080 --fps 60
    ./pp showcase --no-video              # just the pictures (the video takes a minute at full size)
    ./pp showcase --selftest

The pictures are drawn by the game's own renderer from made-up scores, so they need no camera and no hub.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from pingpong import hud, intro  # noqa: E402
from pingpong.events import GateResult  # noqa: E402
from pingpong.uistate import Results, UiState  # noqa: E402

STATS = (("HITS", "31"), ("LONGEST RALLY", "12"), ("TOP SPEED", "74 km/h"), ("TIME", "2:41"))
LEVELS = {1: "Rookie", 2: "Club", 3: "Pro"}


def _menu(screen, level=2, mode="survival", t_s=3.2, **kw):
    results = kw.pop("results", None)
    return hud.HudState(screen=screen, ui=UiState(screen=screen, t_s=t_s, level_tag=level, mode=mode, **kw), player_name="rafae",
                        level_name=LEVELS[level], mode=mode, results=results, hub_status="ok", mqtt_status="ok", anim_t=t_s, hub_battery=84,
                        leaderboard=(("maya", 21), ("rafae", 17), ("omar", 12), ("zed", 9)), player_points=7, cpu_points=4, streak=17,
                        record=21, target=7)


def _game(**kw):
    base = dict(level_name="Club", player_name="rafae", anim_t=1.3, hub_battery=84, mqtt_status="ok", ball=(0.1, 0.35, 1.6),
                paddle=(0.1, 0.1, 0.1), rest=(0.1, 0.0, 0.3), cpu_x_m=0.2)
    base.update(kw)
    return hud.HudState(**base)


def pictures():
    """name -> the HudState to draw."""
    won = Results(won=True, title="YOU WIN!", stats=STATS)
    return {
        "title": _menu("TITLE", cursor=(0.9, 0.85), hover="start", progress=0.45),
        "mode": _menu("MODE", cursor=(0.72, 0.45), hover="match", progress=0.5, focus=1, mode="match"),
        "opponent": _menu("OPPONENT", cursor=(0.5, 0.4), hover="opp2", progress=0.4, focus=1, mode="match"),
        "vs": _menu("VS", t_s=1.5, mode="match"),
        "win": _menu("RESULTS", mode="match", results=won, cursor=(0.95, 0.9), hover="again", progress=0.3),
        "lose": _menu("RESULTS", mode="match", results=Results(won=False, title="THE CPU WINS", stats=STATS)),
        "record": _menu("RESULTS", results=Results(won=None, title="NEW RECORD!", new_record=True, stats=STATS)),
        "rally": _game(phase="RALLY", streak=7, record=12, last_kmh=63.0, last_label="perfect", spin_text="TOPSPIN"),
        "match": _game(phase="RALLY", mode="match", player_points=3, cpu_points=5, last_kmh=41.0, last_label="late", cpu_mood="smug",
                       show_xray=True, gates=(GateResult("J1", True, "timing +12 ms"), GateResult("J2", False, "hand 0.9 SW from the ball"))),
        "countdown": _game(phase="COUNTDOWN", countdown=2, countdown_t=0.15, ball=None, paddle=None),
    }


def write_all(out, size, fps, video):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, state in pictures().items():
        cv2.imwrite(str(out / f"{name}.png"), hud.render(state, size=size))
    frames = intro.export_video(out / "intro.mp4", size=size, fps=fps) if video else 0
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/showcase")
    ap.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT of the pictures and the video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            frames = write_all(tmp, (192, 108), 5, True)
            assert frames >= 40 and len(list(Path(tmp).glob("*.png"))) == len(pictures())
        print(f"showcase selftest OK: {len(pictures())} pictures and a {frames}-frame intro")
        return 0
    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        print("--size must be WIDTHxHEIGHT, e.g. 1280x720", file=sys.stderr)
        return 2
    frames = write_all(args.out, (width, height), args.fps, not args.no_video)
    print(f"wrote {len(pictures())} pictures" + (f" and the intro ({frames} frames at {args.fps} fps)" if frames else "") + f" to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
