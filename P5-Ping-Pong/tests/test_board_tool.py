"""./pp board: put the scoreboard app on the UNO Q over the USB cable, start it, and show something on the matrix to see it works."""

from pathlib import Path

from pingpong import boardlink, matrix
from tools import board

ADB = "/x/adb"
DEST = "/home/arduino/ArduinoApps/pingpong-board"
DEVICES = "List of devices attached\n2966999299\tdevice usb:1048576X transport_id:13\n\n"
NO_DEVICES = "List of devices attached\n\n"
IDLE = ("ID NAME ICON STATUS EXAMPLE\nexamples:inspirational/blink   Blink LED from Python   🔴   uninitialized true\n"
        "user:ratengo   RateNgo   ❤️   uninitialized false\nuser:yolo   Yolo   😋   uninitialized false\n")
RATENGO_RUNNING = IDLE.replace("user:ratengo   RateNgo   ❤️   uninitialized", "user:ratengo   RateNgo   ❤️   running")
OURS_RUNNING = IDLE + "user:pingpong-board   PingPong Scoreboard   🏓   running false\n"


class Done:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


class FakeAdb:
    def __init__(self, devices=DEVICES, apps=IDLE, fail_on=()):
        self.devices, self.apps, self.fail_on, self.calls, self.pushed = devices, apps, fail_on, [], None

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        assert cmd[0] == ADB
        if "push" in cmd:
            source = Path(cmd[cmd.index("push") + 1].rstrip("/."))
            self.pushed = sorted(str(p.relative_to(source)) for p in source.rglob("*") if p.is_file())
        text = " ".join(cmd[1:])
        if any(word in text for word in self.fail_on):
            return Done(1, "", f"failed: {text}")
        if cmd[1:] == ["devices"]:
            return Done(0, self.devices)
        if text.endswith("app list"):
            return Done(0, self.apps)
        return Done(0, "")


class FakeConnection:
    def __init__(self):
        self.sent, self.closed = [], False

    def send(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def deploy(adb, replace=False, connect=None):
    out = []
    clock = {"t": 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    code = board.deploy(ADB, replace=replace, run=adb, out=out.append, connect=connect or (lambda: FakeConnection()), sleep=sleep,
                        now=lambda: clock["t"])
    return code, "\n".join(out)


def commands(adb):
    return [" ".join(c[1:]) for c in adb.calls]


# --- deploy -----------------------------------------------------------------------------------------------------------------------------
def test_deploy_copies_the_app_to_the_board_and_starts_it_and_waits_for_it_to_answer():
    adb = FakeAdb()
    code, text = deploy(adb)
    assert code == 0
    seq = commands(adb)
    assert seq[0] == "devices" and seq[1].endswith("app list")
    assert seq.index(f"shell mkdir -p {DEST}") < next(i for i, c in enumerate(seq) if c.startswith("push")) < \
        next(i for i, c in enumerate(seq) if "app restart" in c)
    assert any(c.startswith("push") and c.endswith(DEST) for c in seq)
    assert any(c == f"shell arduino-app-cli app restart {DEST}" for c in seq)
    assert "running" in text


def test_what_is_copied_is_the_app_and_none_of_the_files_python_makes_on_this_mac():
    adb = FakeAdb()
    deploy(adb)
    assert adb.pushed == sorted(["app.yaml", "python/main.py", "python/scoreboard.py", "sketch/sketch.ino", "sketch/sketch.yaml",
                                 "sketch/src/frame_codec.cpp", "sketch/src/frame_codec.h"])


def test_without_a_board_on_the_cable_it_says_so_and_touches_nothing():
    adb = FakeAdb(devices=NO_DEVICES)
    code, text = deploy(adb)
    assert code == 1 and commands(adb) == ["devices"] and "plug" in text.lower()


def test_an_offline_or_unauthorised_board_is_not_a_board_on_the_cable():
    adb = FakeAdb(devices="List of devices attached\n2966999299\toffline\n")
    code, _ = deploy(adb)
    assert code == 1 and commands(adb) == ["devices"]


def test_another_app_running_on_the_board_is_not_stopped_unless_asked():
    adb = FakeAdb(apps=RATENGO_RUNNING)
    code, text = deploy(adb)
    assert code == 1 and "RateNgo".lower() in text.lower() and "--replace" in text
    assert not any(c.startswith("push") or "restart" in c or " stop " in c for c in commands(adb))


def test_replace_stops_the_other_app_first_and_then_deploys():
    adb = FakeAdb(apps=RATENGO_RUNNING)
    code, _ = deploy(adb, replace=True)
    seq = commands(adb)
    stop = f"shell arduino-app-cli app stop /home/arduino/ArduinoApps/ratengo"
    assert code == 0 and stop in seq and seq.index(stop) < next(i for i, c in enumerate(seq) if c.startswith("push"))


def test_the_scoreboard_already_running_is_just_restarted_with_the_new_files():
    adb = FakeAdb(apps=OURS_RUNNING)
    code, _ = deploy(adb)
    assert code == 0 and not any(" stop " in c for c in commands(adb))


def test_a_failing_step_stops_the_deploy_and_shows_adbs_own_words():
    adb = FakeAdb(fail_on=("app restart",))
    code, text = deploy(adb)
    assert code == 1 and "failed: shell arduino-app-cli app restart" in text


def test_it_waits_for_the_app_to_come_up_and_gives_up_with_a_pointer_to_its_log():
    attempts = []

    def late():
        attempts.append(1)
        if len(attempts) < 4:
            raise OSError("the scoreboard app is not running on the board")
        return FakeConnection()

    code, _ = deploy(FakeAdb(), connect=late)
    assert code == 0 and len(attempts) == 4

    def never():
        raise OSError("the scoreboard app is not running on the board")

    code, text = deploy(FakeAdb(), connect=never)
    assert code == 1 and "did not answer" in text and "app logs" in text


# --- test ---------------------------------------------------------------------------------------------------------------------------------
def test_the_test_shows_a_few_scenes_one_after_another_and_ends_on_standby():
    conn, out, slept = FakeConnection(), [], []
    code = board.show_test(lambda: conn, out=out.append, sleep=slept.append)
    assert code == 0 and conn.closed and len(conn.sent) >= 5
    assert conn.sent[0] != conn.sent[1] and conn.sent[-1] == boardlink.message(matrix.frame(("standby",), 0.0))
    assert all(line.startswith(b"F") and len(line) == 106 for line in conn.sent)
    assert out and sum(slept) >= 8


def test_the_test_says_what_is_missing_when_it_cannot_reach_the_board():
    out = []

    def nothing():
        raise OSError("adb: no devices/emulators found")

    assert board.show_test(nothing, out=out.append, sleep=lambda s: None) == 1 and "no devices" in " ".join(out)


def test_the_tools_own_selftest_passes(capsys):
    assert board.main(["--selftest"]) == 0
    assert "board selftest OK" in capsys.readouterr().out
