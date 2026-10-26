import http.client
import json
from pathlib import Path

import pytest

from models import Tree
from seed_models import SEED1, SEED2
from pipeline import Gait
from recorder import Recorder
from web import serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(tmp_path):
    g = Gait(Tree(SEED1), Tree(SEED2),
             Recorder(tmp_path, rolling=False))
    (tmp_path / "20261023-120000-walk.csv").write_text("t,label\n0.0,WALK\n")
    s = serve(g, port=0)
    yield s, g
    s.shutdown()
    s.server_close()


def req(server, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    c.request(method, path, body=body)
    r = c.getresponse()
    data = r.read()
    c.close()
    req.last_headers = dict(r.getheaders())
    return r.status, r.getheader("Content-Type", ""), data


def test_page_and_state(server):
    s, g = server
    status, ctype, body = req(s, "GET", "/")
    assert status == 200 and "text/html" in ctype and b"EventSource" in body
    status, ctype, body = req(s, "GET", "/state")
    assert status == 200 and json.loads(body)["act"] == "---"


def test_recorder_controls(server):
    s, g = server
    status, _, body = req(s, "POST", "/record?label=walk&who=ann")
    assert status == 200 and json.loads(body)["recorder"]["active"] is True
    assert g.recorder.who == "ann" and g.recorder.path.name.endswith("-walk-ann.csv")
    status, _, _ = req(s, "POST", "/record?label=fly")
    assert status == 400
    status, _, body = req(s, "POST", "/stop")
    assert status == 200 and json.loads(body)["recorder"]["active"] is False
    status, _, body = req(s, "POST", "/guided?who=bob")
    assert status == 200 and g.recorder.active and g.recorder.path.name.endswith("-guided-bob.csv")
    g.recorder.stop()


def test_session_download_and_path_safety(server):
    s, g = server
    status, ctype, body = req(s, "GET", "/data/20261023-120000-walk.csv")
    assert status == 200 and "text/csv" in ctype and body.startswith(b"t,label")
    assert "attachment" in req.last_headers["Content-Disposition"]
    status, _, _ = req(s, "GET", "/data/../app.yaml")
    assert status == 404
    status, _, _ = req(s, "GET", "/nope")
    assert status == 404


def test_events_stream_sends_json_frames(server):
    s, g = server
    c = http.client.HTTPConnection("127.0.0.1", s.server_address[1], timeout=3)
    c.request("GET", "/events")
    r = c.getresponse()
    assert r.status == 200 and "text/event-stream" in r.getheader("Content-Type")
    line = r.fp.readline()
    while not line.startswith(b"data: "):
        assert line.strip() in (b"", b"retry: 2000")
        line = r.fp.readline()
    assert json.loads(line[6:])["act"] == "---"
    c.close()
