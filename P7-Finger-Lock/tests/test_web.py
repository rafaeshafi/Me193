"""The board's web layer, with a stand-in Safe (no camera, no Bridge)."""
import json
import urllib.request

import pytest

import web
from lock import config
from lock.policy import Lock


class FakeSafe:
    def __init__(self):
        self.lock = Lock(config.validate({"code": [1, 2, 3]}))
        self.roi_jpg = None

    def snapshot(self):
        return dict(self.lock.snapshot(), cls=0, fps=9.9, inference_ms=4.2, camera=False, frames=0, engine="numpy", code_len=3)

    def stats(self):
        return {"fps": 9.9}

    def button(self):
        return self.lock.press_button()


@pytest.fixture
def server():
    safe = FakeSafe()
    srv = web.serve(safe, port=0, host="127.0.0.1")
    yield safe, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url):
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


def test_page_state_stats_and_button(server):
    safe, base = server
    code, ctype, body = get(base + "/")
    assert code == 200 and ctype.startswith("text/html") and b"Finger Lock" in body
    s = json.loads(get(base + "/state")[2])
    assert s["state"] == "IDLE" and s["code_len"] == 3
    assert post(base + "/button") == {"started": True, "state": "LISTENING"}
    assert json.loads(get(base + "/stats")[2])["fps"] == 9.9


def test_roi_404_then_jpeg(server):
    safe, base = server
    with pytest.raises(urllib.error.HTTPError):
        get(base + "/roi.jpg")
    safe.roi_jpg = b"\xff\xd8fake"
    assert get(base + "/roi.jpg?1")[1] == "image/jpeg"


def test_reset_clears_lockout(server):
    safe, base = server
    safe.lock.state, safe.lock.attempts = "LOCKOUT", 3
    assert post(base + "/reset") == {"state": "IDLE"} and safe.lock.attempts == 0
