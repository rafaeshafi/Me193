"""Re-run a recorded session through the real game code, optionally with changed settings.

Usage:
    ./pp replay recordings/<session>                              # does it reproduce what happened?
    ./pp replay recordings/<session> --set level.late_s=0.4      # what if the late window were wider?
    ./pp replay recordings/<session> --set swing.t_pk=150 --set judge.d95_s=0.15
    ./pp replay recordings/<session> --out recordings/<session>-replay   # keep the replayed recording
    ./pp replay --selftest

Settings are section.name=value with section one of swing (the swing detector), judge (the hit
judge) or level (the ball speed tier: early_s, late_s, radius_sw, ...).  Nothing is sent anywhere:
a replay has no broker and no hub.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingpong import recorder, replay, sessionreport  # noqa: E402


def _selftest():
    from pingpong import fakerig

    with tempfile.TemporaryDirectory() as tmp:
        rig = fakerig.FakeRig(record_dir=Path(tmp) / "orig")
        rig.run(until=lambda: rig.game.tracker.streak >= 4, max_s=60)
        rig.close()
        result = replay.replay(recorder.load(Path(tmp) / "orig"))
    assert result.hits == 4 and result.record == 4, (result.hits, result.record)
    print("replay selftest OK: the recorded session reproduces its 4 hits")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="a recorded session folder")
    ap.add_argument("--set", action="append", default=[], metavar="section.name=value")
    ap.add_argument("--out", default=None, help="keep the replayed session here (default: a temporary folder)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.session or not (Path(args.session) / "session.json").exists():
        print(f"no recorded session at {args.session!r}")
        return 1
    try:
        overrides = replay.check_overrides(replay.parse_overrides(args.set))
    except ValueError as exc:
        print(f"cannot replay: {exc}", file=sys.stderr)
        return 2
    original = recorder.load(args.session)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(args.out) if args.out else Path(tmp) / "replayed"
        result = replay.replay(original, overrides=overrides, record_dir=out)
        replayed = recorder.load(out)
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        print("== original")
        print(sessionreport.format_report(sessionreport.summarize(original)))
        print(f"== replayed" + (f" with {', '.join(args.set)}" if args.set else " unchanged"))
        print(sessionreport.format_report(sessionreport.summarize(replayed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
