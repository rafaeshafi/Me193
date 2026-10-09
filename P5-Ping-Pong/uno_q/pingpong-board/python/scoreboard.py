"""The scoreboard on the UNO Q, Linux side: the game on the Mac sends it frames and it has the MCU draw them on the LED matrix.

The talk is one line per frame, `F` and 104 digits 0-7 (the pixels row by row); the app greets a game that connects with `PPBOARD 1`.
The game is reached through `adb forward` over the USB cable (pingpong/boardlink.py is its side).  It draws only when the picture
changes, shows three dim dots when it has heard nothing for a while (no game, the cable out, the game quit) and tries again a moment
after a draw the MCU did not take.  Standard library only: this runs in the app container on the board.
"""

import logging
import socketserver
import threading
import time

PORT = 7788
HELLO = b"PPBOARD 1\n"
PIXELS = 104                       # 8 rows x 13 columns
STANDBY_AFTER_S = 8.0              # silence this long and the board shows its standby dots
RETRY_S = 1.0                      # after a draw the MCU did not take, wait this long before the next try
STANDBY = "".join("2" if i in (43, 45, 47) else "0" for i in range(PIXELS))      # three dim dots in the middle row

log = logging.getLogger("scoreboard")


def parse(line):
    """A line of the game's talk -> the frame (104 characters, each 0-7), or None if it is not a frame."""
    line = line.strip()
    if len(line) == PIXELS + 1 and line[:1] == b"F" and all(48 <= b <= 55 for b in line[1:]):
        return line[1:].decode()
    return None


class Scoreboard:
    def __init__(self, draw, now=time.monotonic):
        """draw(frame) puts a frame on the matrix and raises if it cannot."""
        self._draw, self._now = draw, now
        self._frame = self._heard = self._drawn = self._failed_at = None
        self._lock = threading.Lock()

    def hear(self, line):
        frame = parse(line)
        if frame is not None:
            with self._lock:
                self._frame, self._heard = frame, self._now()

    def wanted(self):
        with self._lock:
            quiet = self._heard is None or self._now() - self._heard >= STANDBY_AFTER_S
            return STANDBY if quiet else self._frame

    def tick(self):
        frame, now = self.wanted(), self._now()
        if frame == self._drawn or (self._failed_at is not None and now - self._failed_at < RETRY_S):
            return
        try:
            self._draw(frame)
        except Exception as exc:                          # the MCU is still starting, or busy: try again in a moment
            self._failed_at = now
            log.warning("could not draw (%s); trying again", exc)
            return
        self._drawn, self._failed_at = frame, None

    def loop(self):
        self.tick()
        time.sleep(0.02)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.wfile.write(HELLO)
        self.wfile.flush()
        for line in self.rfile:                           # until the game hangs up
            self.server.board.hear(line)


def serve(board, port=PORT, host="0.0.0.0"):
    """Listen for the game on a background thread; returns the server (shutdown() stops it)."""
    server = socketserver.ThreadingTCPServer((host, port), _Handler)
    server.daemon_threads = True
    server.board = board
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
