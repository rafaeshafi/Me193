"""Manual stepper control on the UNO Q; App Lab runs this in the app container. Buttons on port 8080 -> Bridge -> sketch.

GET /            the page                GET  /state            {"pos": steps from start, "queued": steps still to do}
POST /move?n=50  n full steps (signed)   POST /run?dir=1|-1     keep stepping     POST /stop      stop, coils hold
POST /release    stop, coils off         POST /speed?ms=20      milliseconds a step (2..500)
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from arduino.app_utils import App, Bridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stepper")
PAGE = (Path(__file__).resolve().parent / "page.html").read_bytes()


def call(name, *args):
    try:
        return Bridge.call(name, *args, timeout=2)
    except Exception as exc:  # the MCU is still compiling/starting
        log.warning("bridge %s%s failed: %s", name, args, exc)
        return None


def arg(query, key, lo, hi, default):
    try:
        return max(lo, min(hi, int(query.get(key, [default])[0])))
    except (TypeError, ValueError):
        return default


def state():
    return {"pos": call("pos"), "queued": call("queued")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/state":
            self._json(state())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/move":
            call("move", arg(q, "n", -100000, 100000, 0))
        elif url.path == "/run":
            call("run", arg(q, "dir", -1, 1, 0))
        elif url.path == "/stop":
            call("stop")
        elif url.path == "/release":
            call("release")
        elif url.path == "/speed":
            call("speed", arg(q, "ms", 2, 500, 20))
        else:
            self._send(404, "text/plain", b"not found")
            return
        log.info("%s %s", url.path, dict(q))
        self._json(state())


server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
threading.Thread(target=server.serve_forever, daemon=True, name="web").start()
log.info("stepper control on port 8080")
App.run(user_loop=lambda: time.sleep(0.5))
