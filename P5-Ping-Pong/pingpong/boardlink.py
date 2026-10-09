"""The game's link to the UNO Q's LED matrix over the USB cable (matrix.py says what to show; uno_q/ is the app that draws it).

The board has no network of its own when it is only on the cable, but `adb` (the tool Arduino App Lab uses) can forward a port of this Mac
through the cable to a port of the board.  So the game connects to localhost on that port and the scoreboard app on the board answers
with a greeting; after that a frame is a line of 104 digits.  Everything that can go wrong (no adb, no board, no app, a cable pulled
out) only makes the board go on showing its last frame: it is never waited for, and the game does not need it.
"""

import glob
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

from pingpong import matrix

BOARD_PORT = 7788               # the port the scoreboard app listens on (uno_q/pingpong-board/app.yaml publishes it)
LOCAL_PORT = 17788              # the port on this Mac that adb forwards to it through the cable
HELLO = b"PPBOARD 1"            # what the app says to a game that connects (and which version of this talk it speaks)
RETRY_S = 3.0                   # how long to wait before looking again for a board that was not there
KEEPALIVE_S = 2.0               # a picture that has not changed is sent again after this long (the board shows standby after 8 s of silence)
TURN_S = 0.05                   # how often the sender looks at what the game wants shown


def message(frame):
    """A frame as it goes over the wire: F, then a digit 0-7 for each of the 104 pixels row by row, then a newline."""
    return b"F" + "".join(str(v) for v in frame).encode() + b"\n"


class SocketConnection:
    def __init__(self, sock):
        self._sock = sock

    def send(self, data):
        self._sock.sendall(data)                       # (raises OSError when the link is broken)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


def forward_connector(adb, *, run=subprocess.run, open_socket=socket.create_connection):
    """-> a function that makes the link: asks adb to forward LOCAL_PORT to the board's BOARD_PORT, connects, and waits for the app's
    greeting.  Raises OSError, saying what is missing, when any of that fails."""
    def connect():
        try:
            done = run([adb, "forward", f"tcp:{LOCAL_PORT}", f"tcp:{BOARD_PORT}"], capture_output=True, text=True, timeout=6)
        except subprocess.TimeoutExpired:
            raise OSError("adb did not answer")
        if done.returncode != 0:
            raise OSError((done.stderr or done.stdout or "adb forward failed").strip())
        sock = open_socket(("127.0.0.1", LOCAL_PORT), timeout=2)
        try:
            greeting = sock.makefile("rb").readline().strip()
        except OSError:
            sock.close()
            raise
        if greeting != HELLO:                          # a forward to a port nothing listens on is closed at once, with no greeting
            sock.close()
            raise OSError("the scoreboard app is not running on the board (./pp board deploy)")
        sock.settimeout(2)                             # a send that takes longer than this is a broken link
        return SocketConnection(sock)
    return connect


def find_adb(home=None):
    """The adb to use: PP_ADB, else the one Arduino App Lab installed (a different adb would replace the adb server App Lab started and
    cut it off), else the one on the path; None if there is none."""
    named = os.environ.get("PP_ADB")
    if named:
        return named
    pattern = str(Path(home or Path.home()) / "Library/Arduino15/packages/arduino/tools/adb/*/adb")
    versions = sorted(glob.glob(pattern), key=lambda path: [int(p) if p.isdigit() else 0 for p in Path(path).parent.name.split(".")])
    return versions[-1] if versions else shutil.which("adb")


class BoardLink:
    def __init__(self, connect, *, now=time.monotonic, log=print):
        """connect() -> a connection with send(bytes) and close(), or raises OSError.  now and log can be replaced in a test."""
        self._connect, self._now, self._log = connect, now, log
        self._latest = None                              # the newest picture the game wants shown (a tuple: set whole, read whole)
        self._conn = self._sent = self._sent_at = self._said = self._thread = None
        self._next_try = 0.0
        self._stop = threading.Event()

    def show(self, state):
        """The game's turn: work out the picture for its HudState.  Cheap, and never waits for the board."""
        self._latest = matrix.frame(matrix.scene_of(state), self._now())

    def pump(self):
        """One turn of the sender: find the board if there is no link, and send the picture when it is new or the board has been quiet."""
        frame, now = self._latest, self._now()
        if frame is None:
            return
        if self._conn is None:
            if now < self._next_try:
                return
            try:
                self._conn = self._connect()
            except OSError as exc:
                self._next_try = now + RETRY_S
                self._say(f"board: not on the cable ({exc}); the game goes on without it")
                return
            self._say("board: connected, the score is on the matrix")
            self._sent = None
        if frame != self._sent or now - self._sent_at >= KEEPALIVE_S:
            try:
                self._conn.send(message(frame))
            except OSError:
                self._drop()
                self._say("board: the link broke; looking for it again")
                return
            self._sent, self._sent_at = frame, now

    def start(self):
        self._thread = threading.Thread(target=self._run, name="board", daemon=True)
        self._thread.start()

    def close(self):
        """Stop, and tell the board the game is over so it does not go on showing a score that is no longer being played."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._conn is not None:
            try:
                self._conn.send(message(matrix.frame(("standby",), 0.0)))
            except OSError:
                pass
            self._drop()

    def _run(self):
        while not self._stop.wait(TURN_S):
            try:
                self.pump()
            except Exception as exc:                     # whatever it was, the board is not worth the game: say it once and go on
                self._say(f"board: {type(exc).__name__}: {exc}")

    def _drop(self):
        conn, self._conn = self._conn, None
        self._next_try = self._now() + RETRY_S
        if conn is not None:
            conn.close()

    def _say(self, text):
        if text != self._said:                           # the same news is not repeated every few seconds
            self._said = text
            self._log(text)


def for_game(find=find_adb, log=print):
    """The link the game uses, or None when there is no adb to reach a board with."""
    adb = find()
    if adb is None:
        log("board: adb was not found, so the matrix on the UNO Q stays off (PP_ADB names an adb)")
        return None
    return BoardLink(forward_connector(adb), log=log)
