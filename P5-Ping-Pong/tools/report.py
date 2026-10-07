"""Summarise recorded sessions: hits, timing, which judge gate rejected what, hub health, pauses.

Usage:
    ./pp report                       # the newest session under recordings/
    ./pp report recordings/<session>  # a specific one (several allowed)
    ./pp report --all                 # every session, oldest first
    ./pp report --selftest
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingpong import recorder, sessionreport  # noqa: E402


def _sessions(root):
    root = Path(root)
    return sorted(p for p in root.iterdir() if (p / "session.json").exists()) if root.is_dir() else []


def _selftest():
    from pingpong import fakerig

    with tempfile.TemporaryDirectory() as tmp:
        rig = fakerig.FakeRig(record_dir=Path(tmp) / "run")
        rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
        rig.close()
        text = sessionreport.format_report(sessionreport.summarize(recorder.load(Path(tmp) / "run")))
    assert "hits 3" in text and "best streak 3" in text, text
    print("report selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session folders (default: the newest one)")
    ap.add_argument("--root", default=None, help="where sessions live (default: recordings/)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    root = Path(args.root) if args.root else recorder.default_root()
    chosen = [Path(p) for p in args.sessions] or (_sessions(root) if args.all else _sessions(root)[-1:])
    if not chosen:
        print(f"no recorded sessions under {root} (play with ./pp play first; recording is on by default)")
        return 1
    for path in chosen:
        print(f"== {path.name}")
        print(sessionreport.format_report(sessionreport.summarize(recorder.load(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
