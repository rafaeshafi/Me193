"""The link from the game to the UNO Q's matrix: frames sent over the USB cable (adb forward + a socket), never waited for."""

import io
import subprocess
import threading
import time

import pytest

from pingpong import boardlink, hud, matrix


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeConnection:
    def __init__(self):
        self.sent, self.closed, self.broken = [], False, False

    def send(self, data):
        if self.broken:
            raise BrokenPipeError("the board went away")
        self.sent.append(data)

    def close(self):
        self.closed = True


def link(outcomes, clock=None, log=None):
    """A BoardLink whose connect() gives these outcomes in turn: an exception to raise or a connection."""
    clock = clock or FakeClock()
    said = [] if log is None else log
    queue = list(outcomes)
    calls = []

    def connect():
        calls.append(clock.t)
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return boardlink.BoardLink(connect, now=clock, log=said.append), clock, calls, said


def idle_state(streak=3, record=9):
    return hud.HudState(screen="TITLE", phase="MATCH_OVER", streak=streak, record=record)


# --- what goes over the wire ----------------------------------------------------------------------------------------------------------------
def test_a_frame_goes_as_one_line_of_a_hundred_and_four_digits():
    data = boardlink.message(matrix.frame(("rally", 7, 20), 0.0))
    assert data.startswith(b"F") and data.endswith(b"\n") and len(data) == 1 + 104 + 1
    assert set(data[1:-1]) <= set(b"01234567")
    assert [int(ch) for ch in data[1:-1].decode()] == list(matrix.frame(("rally", 7, 20), 0.0))


# --- the sender -------------------------------------------------------------------------------------------------------------------------------
def test_the_first_frame_is_sent_once_and_not_again_until_it_changes_or_the_board_has_not_heard_for_a_while():
    conn = FakeConnection()
    bl, clock, _, _ = link([conn])
    bl.show(idle_state())
    bl.pump()
    bl.pump()
    assert len(conn.sent) == 1
    clock.t += 1.0
    bl.pump()
    assert len(conn.sent) == 1                                                           # the same picture: nothing new to say
    clock.t += 1.5
    bl.pump()
    assert len(conn.sent) == 2 and conn.sent[0] == conn.sent[1]                          # ... but the board is told it is still there
    bl.show(hud.HudState(screen="GAME", phase="RALLY", mode="survival", streak=1, record=9))
    bl.pump()
    assert len(conn.sent) == 3 and conn.sent[2] != conn.sent[1]


def test_nothing_is_sent_before_the_game_has_shown_anything():
    conn = FakeConnection()
    bl, _, calls, _ = link([conn])
    bl.pump()
    assert calls == [] and conn.sent == []


def test_showing_a_state_never_waits_for_the_board():
    started = threading.Event()
    release = threading.Event()

    def connect():                                                                       # a link that hangs
        started.set()
        release.wait(5)
        return FakeConnection()

    bl = boardlink.BoardLink(connect, now=time.monotonic, log=lambda *_: None)
    bl.start()
    try:
        bl.show(idle_state())
        assert started.wait(2)
        t0 = time.perf_counter()
        for k in range(200):
            bl.show(idle_state(streak=k))
        assert time.perf_counter() - t0 < 0.5                                            # the game's turn is not held up by it
    finally:
        release.set()
        bl.close()


def test_a_board_that_is_not_there_is_said_once_and_asked_for_again_every_few_seconds():
    bl, clock, calls, said = link([OSError("no devices found")])
    bl.show(idle_state())
    for _ in range(3):
        bl.pump()
    assert len(calls) == 1 and len(said) == 1 and "board" in said[0] and "no devices found" in said[0]
    clock.t += boardlink.RETRY_S + 0.1
    bl.pump()
    assert len(calls) == 2 and len(said) == 1                                            # asked again; the same news is not repeated


def test_a_board_that_comes_back_is_found_and_the_picture_is_sent_to_it_at_once():
    conn = FakeConnection()
    bl, clock, calls, said = link([OSError("no devices found"), conn])
    bl.show(idle_state())
    bl.pump()
    clock.t += boardlink.RETRY_S + 0.1
    bl.pump()
    assert len(conn.sent) == 1 and any("connected" in line for line in said)


def test_a_link_that_breaks_is_dropped_and_made_again_with_the_latest_picture():
    first, second = FakeConnection(), FakeConnection()
    bl, clock, calls, said = link([first, second])
    bl.show(idle_state())
    bl.pump()
    first.broken = True
    bl.show(hud.HudState(screen="GAME", phase="RALLY", mode="survival", streak=4, record=9))
    bl.pump()                                                                            # the send fails: the link is dropped
    assert first.closed and len(calls) == 1
    clock.t += boardlink.RETRY_S + 0.1
    bl.pump()
    assert len(calls) == 2 and len(second.sent) == 1 and second.sent[0] == boardlink.message(
        matrix.frame(("rally", 4, 9), clock.t))


def test_the_sender_thread_delivers_and_closing_tells_the_board_the_game_is_over():
    conn = FakeConnection()
    bl = boardlink.BoardLink(lambda: conn, now=time.monotonic, log=lambda *_: None)
    bl.start()
    bl.show(idle_state())
    deadline = time.time() + 3
    while not conn.sent and time.time() < deadline:
        time.sleep(0.02)
    assert conn.sent
    bl.close()
    assert conn.closed and conn.sent[-1] == boardlink.message(matrix.frame(("standby",), 0.0))


# --- the cable ------------------------------------------------------------------------------------------------------------------------------
class FakeSocket:
    def __init__(self, greeting=b"PPBOARD 1\n"):
        self.stream, self.timeout, self.sent, self.closed = io.BytesIO(greeting), None, [], False

    def makefile(self, mode):
        return self.stream

    def settimeout(self, seconds):
        self.timeout = seconds

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


class Done:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def test_the_cable_forwards_a_port_with_adb_and_waits_for_the_boards_greeting():
    ran, opened = [], []
    sock = FakeSocket()

    def run(cmd, **kw):
        ran.append(cmd)
        return Done()

    def open_socket(address, timeout):
        opened.append((address, timeout))
        return sock

    conn = boardlink.forward_connector("/x/adb", run=run, open_socket=open_socket)()
    assert ran == [["/x/adb", "forward", f"tcp:{boardlink.LOCAL_PORT}", f"tcp:{boardlink.BOARD_PORT}"]]
    assert opened[0][0] == ("127.0.0.1", boardlink.LOCAL_PORT)
    conn.send(b"F000\n")
    assert sock.sent == [b"F000\n"] and sock.timeout is not None
    conn.close()
    assert sock.closed


def test_a_cable_with_no_board_on_it_is_an_error_the_link_can_report():
    run = lambda cmd, **kw: Done(1, "", "adb: no devices/emulators found")           # noqa: E731
    with pytest.raises(OSError, match="no devices"):
        boardlink.forward_connector("/x/adb", run=run, open_socket=lambda *a, **k: FakeSocket())()


@pytest.mark.parametrize("greeting", [b"", b"hello\n", b"PPBOARD 2\n"])
def test_an_app_on_the_board_that_does_not_greet_properly_is_not_the_scoreboard(greeting):
    sock = FakeSocket(greeting)
    with pytest.raises(OSError, match="board deploy"):
        boardlink.forward_connector("/x/adb", run=lambda cmd, **kw: Done(), open_socket=lambda *a, **k: sock)()
    assert sock.closed


def test_a_forward_to_a_board_with_the_app_not_running_closes_at_once_and_is_not_the_scoreboard():
    def refuse(address, timeout):
        raise ConnectionRefusedError("refused")

    with pytest.raises(OSError):
        boardlink.forward_connector("/x/adb", run=lambda cmd, **kw: Done(), open_socket=refuse)()


def test_an_adb_that_hangs_is_a_board_that_is_not_there_and_not_a_crash():
    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 6)

    with pytest.raises(OSError, match="did not answer"):
        boardlink.forward_connector("/x/adb", run=hang, open_socket=lambda *a, **k: FakeSocket())()


def test_the_sender_thread_survives_a_bug_and_says_so_once():
    said = []

    def broken():
        raise ValueError("a bug in the link")

    bl = boardlink.BoardLink(broken, now=time.monotonic, log=said.append)
    bl.start()
    bl.show(idle_state())
    deadline = time.time() + 2
    while not said and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)                                                                      # several more turns of the sender
    assert bl._thread.is_alive() and said == ["board: ValueError: a bug in the link"]
    bl.close()


# --- finding adb ------------------------------------------------------------------------------------------------------------------------------
def test_adb_is_the_one_app_lab_brought_so_the_adb_server_it_started_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.delenv("PP_ADB", raising=False)
    for version in ("31.0.0", "32.0.0"):
        adb = tmp_path / "Library/Arduino15/packages/arduino/tools/adb" / version / "adb"
        adb.parent.mkdir(parents=True)
        adb.write_text("")
    monkeypatch.setattr(boardlink.shutil, "which", lambda name: "/usr/local/bin/adb")
    assert boardlink.find_adb(home=tmp_path) == str(tmp_path / "Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb")


def test_adb_can_be_named_and_is_otherwise_the_one_on_the_path_or_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PP_ADB", "/opt/adb")
    assert boardlink.find_adb(home=tmp_path) == "/opt/adb"
    monkeypatch.delenv("PP_ADB")
    monkeypatch.setattr(boardlink.shutil, "which", lambda name: "/usr/local/bin/adb")
    assert boardlink.find_adb(home=tmp_path) == "/usr/local/bin/adb"
    monkeypatch.setattr(boardlink.shutil, "which", lambda name: None)
    assert boardlink.find_adb(home=tmp_path) is None


def test_without_adb_there_is_no_board_link_and_the_game_is_told_so_once():
    said = []
    assert boardlink.for_game(find=lambda: None, log=said.append) is None and len(said) == 1


def test_the_port_the_game_talks_to_is_the_one_the_board_app_publishes():
    from pathlib import Path

    app_yaml = (Path(__file__).resolve().parents[1] / "uno_q" / "pingpong-board" / "app.yaml").read_text()
    assert f"ports: [{boardlink.BOARD_PORT}]" in app_yaml
