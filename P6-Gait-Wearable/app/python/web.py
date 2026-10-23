"""The phone page: live verdicts and waveform over Server-Sent Events, recorder controls, CSV downloads. Stdlib only.

GET  /            the page            GET /events  SSE stream of Gait.snapshot() at 10 Hz
GET  /state       one snapshot (JSON) GET /data/<session.csv>  download a recording
POST /guided?who= start the guided script   POST /record?label=WALK&who=   POST /stop
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from recorder import LABELS, clean

PAGE = (Path(__file__).resolve().parent / "page.html").read_bytes()
EVENT_PERIOD_S = 0.1


class Handler(BaseHTTPRequestHandler):
    gait = None  # set by serve()

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif url.path == "/state":
            self._json(self.gait.snapshot())
        elif url.path == "/events":
            self._events()
        elif url.path.startswith("/data/"):
            self._file(url.path[len("/data/"):])
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            who = clean(q.get("who", [""])[0])
            rec = self.gait.recorder
            if url.path == "/guided":
                rec.start_guided(who=who)
            elif url.path == "/record":
                label = q.get("label", [""])[0].upper()
                if label not in LABELS:
                    return self._json({"error": "label must be one of %s" % ", ".join(LABELS)}, 400)
                rec.start_single(label, who=who)
            elif url.path == "/stop":
                rec.stop()
            else:
                return self._send(404, "text/plain", b"not found")
            self._json(self.gait.snapshot())
        except (ValueError, OSError) as e:
            return self._json({"error": str(e)}, 400)

    def _file(self, name):
        path = self.gait.recorder.dir / Path(name).name
        if Path(name).name != name or not path.is_file() or path.suffix != ".csv":
            return self._send(404, "text/plain", b"not found")
        self._send(200, "text/csv", path.read_bytes(),
                   [("Content-Disposition", 'attachment; filename="%s"' % name)])

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.connection.settimeout(5)
            self.wfile.write(b"retry: 2000\n\n")
            while True:
                self.wfile.write(("data: %s\n\n" % json.dumps(self.gait.snapshot())).encode())
                self.wfile.flush()
                time.sleep(EVENT_PERIOD_S)
        except OSError:  # broken pipe, reset, or a socket timeout
            pass


def serve(gait, port=8080, host="0.0.0.0"):
    """Serve on a background thread; returns the server (server_address[1] is the port, shutdown() stops it)."""
    Handler.gait = gait
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
