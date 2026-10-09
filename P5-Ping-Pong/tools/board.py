"""./pp board: the scoreboard on the UNO Q's LED matrix, over the USB cable.

Usage:
    ./pp board deploy                 # copy uno_q/pingpong-board to the board, (re)start it, wait for it to answer
    ./pp board deploy --replace       # ... and stop another app that is running on the board first (it runs one app at a time)
    ./pp board test                   # show a few scenes on the matrix, one after another, to see that it works
    ./pp board --selftest

The first start compiles the sketch on the board, which takes a minute or two.  After that `./pp play` shows the score on the matrix
whenever the board is on the cable (`--no-board` leaves it alone).  Arduino App Lab can start and stop the app too.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingpong import boardlink, matrix  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "uno_q" / "pingpong-board"
APPS = "/home/arduino/ArduinoApps"
DEST = f"{APPS}/pingpong-board"
APP_ID = "user:pingpong-board"
WAIT_S, POLL_S = 90.0, 3.0                     # how long to wait for the app to answer after it is started, and how often to ask
SCENES = (("not in a game: the streak you are on, 12", ("idle", 12, 25), 1.0), ("... and then your best, 25", ("idle", 12, 25), 4.0),
          ("a rally: streak 7, best 20", ("rally", 7, 20), 0.0), ("a rally at your best", ("rally", 20, 20), 0.0),
          ("a match: 3 - 5", ("match", 3, 5), 0.0))


def _boards(devices_output):
    """The boards adb lists as ready: [(serial, state)] for every line below the header, and which of them are 'device'."""
    rows = [line.split() for line in devices_output.splitlines()[1:] if line.strip()]
    return [(row[0], row[1]) for row in rows if len(row) >= 2]


def _running_other_apps(app_list_output):
    """The user apps (other than this one) that the board says are running: ['ratengo', ...]."""
    names = []
    for line in app_list_output.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("user:") and " running " in f" {line} " and fields[0] != APP_ID:
            names.append(fields[0][len("user:"):])
    return names


def deploy(adb, *, replace, run=subprocess.run, out=print, connect=None, sleep=time.sleep, now=time.monotonic):
    """Put the app on the board and start it; -> 0 once it answers, else 1 (what went wrong has been said)."""
    def step(*args):
        done = run([adb, *args], capture_output=True, text=True, timeout=600)
        if done.returncode != 0:
            out(f"adb {' '.join(args)} failed: {(done.stderr or done.stdout).strip()}")
            return None
        return done

    listed = step("devices")
    if listed is None:
        return 1
    if not any(state == "device" for _, state in _boards(listed.stdout)):
        seen = ", ".join(f"{serial} ({state})" for serial, state in _boards(listed.stdout))
        out("no board on the cable" + (f" that is ready: {seen}" if seen else "") + ": plug the UNO Q in and give it half a minute to start")
        return 1
    apps = step("shell", "arduino-app-cli app list")
    if apps is None:
        return 1
    others = _running_other_apps(apps.stdout)
    if others and not replace:
        out(f"{', '.join(others)} is running on the board, which runs one app at a time: ./pp board deploy --replace stops it "
            "(Arduino App Lab starts it again)")
        return 1
    for name in others:
        out(f"stopping {name}")
        if step("shell", f"arduino-app-cli app stop {APPS}/{name}") is None:
            return 1
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "app"
        shutil.copytree(APP, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cache"))        # what Python made here stays here
        if step("shell", f"mkdir -p {DEST}") is None or step("push", f"{source}/.", DEST) is None:
            return 1
    out("starting it on the board (the first time it compiles the sketch: a minute or two)")
    if step("shell", f"arduino-app-cli app restart {DEST}") is None:
        return 1
    connect = connect or boardlink.forward_connector(adb)
    started = now()
    while True:
        try:
            connect().close()
            out("the scoreboard is running: ./pp play now shows the score on the matrix (./pp board test shows some scenes)")
            return 0
        except OSError as exc:
            if now() - started > WAIT_S:
                out(f"the app did not answer in {WAIT_S:.0f} s ({exc}); its log: adb shell arduino-app-cli app logs {DEST}")
                return 1
            sleep(POLL_S)


def show_test(connect, *, out, sleep=time.sleep):
    """A few scenes on the matrix, two seconds each, then the standby dots; -> 0, or 1 if the board cannot be reached."""
    try:
        conn = connect()
    except OSError as exc:
        out(f"cannot reach the board: {exc}")
        return 1
    try:
        for name, scene, t_s in SCENES:
            out(name)
            conn.send(boardlink.message(matrix.frame(scene, t_s)))
            sleep(2.0)
        conn.send(boardlink.message(matrix.frame(("standby",), 0.0)))
        out("done: the board is back to its standby dots")
    finally:
        conn.close()
    return 0


def selftest():
    class Done:
        def __init__(self, out=""):
            self.returncode, self.stdout, self.stderr = 0, out, ""

    class Conn:
        def close(self):
            pass

    listing = {"devices": "List of devices attached\nABC\tdevice\n", "app list": "user:ratengo  RateNgo  x  uninitialized false\n"}
    calls = []

    def run(cmd, **kw):
        calls.append(" ".join(cmd[1:]))
        return Done(next((text for key, text in listing.items() if " ".join(cmd[1:]).endswith(key)), ""))

    clock = {"t": 0.0}
    args = dict(run=run, out=lambda *_: None, connect=lambda: Conn(), sleep=lambda s: None, now=lambda: clock["t"])
    assert deploy("adb", replace=False, **args) == 0 and any(c.startswith("push") for c in calls) and calls[-1].endswith(f"app restart {DEST}")
    listing["app list"] = "user:ratengo  RateNgo  x  running false\n"
    calls.clear()
    assert deploy("adb", replace=False, **args) == 1 and not any(c.startswith("push") for c in calls)
    assert deploy("adb", replace=True, **args) == 0 and any("app stop" in c for c in calls)
    print("board selftest OK: copies the app, starts it, refuses to stop another app unless told to, waits for it to answer")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", choices=("deploy", "test"))
    ap.add_argument("--replace", action="store_true", help="deploy: stop another app that is running on the board")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.action is None:
        ap.print_help()
        return 2
    adb = boardlink.find_adb()
    if adb is None:
        print("adb was not found (Arduino App Lab installs one; PP_ADB names another)")
        return 1
    if args.action == "deploy":
        return deploy(adb, replace=args.replace)
    return show_test(boardlink.forward_connector(adb), out=print)


if __name__ == "__main__":
    raise SystemExit(main())
