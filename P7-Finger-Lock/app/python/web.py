"""The page on port 8080: the state, the crop the net sees, the probabilities, the timings. Stdlib only.

GET /            the page              GET /state  one snapshot (JSON)     GET /roi.jpg  the latest crop
GET /stats       the timing summary    POST /button  the button (for testing without wiring it)
POST /reset      clear failed attempts (and a lockout)
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "page.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    safe = None  # set by serve()

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
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/state":
            self._json(self.safe.snapshot())
        elif path == "/stats":
            self._json(self.safe.stats())
        elif path == "/roi.jpg":
            jpg = self.safe.roi_jpg
            if jpg is None:
                self._send(404, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", jpg)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/button":
            self._json({"started": self.safe.button(), "state": self.safe.lock.state})
        elif path == "/reset":
            self.safe.lock.attempts = 0
            if self.safe.lock.state == "LOCKOUT":
                self.safe.lock._go("IDLE")
            self._json({"state": self.safe.lock.state})
        else:
            self._send(404, "text/plain", b"not found")


def serve(safe, port=8080, host="0.0.0.0"):
    Handler.safe = safe
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="web").start()
    return server
