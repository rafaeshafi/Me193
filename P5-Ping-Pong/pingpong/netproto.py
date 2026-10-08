"""What two laptops say to each other in an online game: small JSON messages over the MQTT broker, strictly checked.

The broker is public, so everything that arrives is untrusted: decode() returns a cleaned message or None, never raises,
rejects anything that is not exactly what the protocol says (a wrong version, a missing field, a NaN, a payload that is too big)
and pulls numbers that are merely too big or small into their range (so a bad peer cannot throw the ball off the table or the
table off the screen).  Topics live in the game's own corner of the broker, never under ME193/ where the score is.

    lobby/<CODE>           retained: a game somebody is hosting (who, which pace, how many points); gone when the host is
    room/<CODE>/host       what the host says to the guest              (QoS 1 for what must arrive, QoS 0 for the paddle and pings)
    room/<CODE>/guest      what the guest says to the host
Messages: join welcome busy | hit miss | pos | ping pong | bye rematch.
"""

import json
import math
import re
import time

VERSION = 1
NAMESPACE = "ME193-pp/v1"
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ"               # no I, L or O: nothing to mistake for a 1 or a 0
CODE_LEN = 5
NAME_MAX, SID_MAX, REASON_MAX, MAX_BYTES = 16, 16, 24, 2048
V_MIN, V_MAX = 0.5, 16.0                            # a ball's speed, m/s
X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX = 1.6, -0.5, 1.5, -1.0, 3.5
LOFT_MAX = 0.6
TARGET_MAX = 21
ROLES = ("host", "guest")


def _reject(constant):
    raise ValueError(f"{constant} is not a number")


def _num(x, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("not a finite number")
    return float(min(hi, max(lo, x)))


def _int(x, lo, hi):
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise ValueError("not a whole number in range")
    return x


def _bool(x):
    if not isinstance(x, bool):
        raise ValueError("not a yes or no")
    return x


def _triple(values, spec):
    if not isinstance(values, list) or len(values) != len(spec):
        raise ValueError("wrong length")
    return [_num(v, lo, hi) for v, (lo, hi) in zip(values, spec)]


def clean_name(name):
    """A name the typeface can draw: upper case, letters, digits, space and - _ . ' only; other symbols become gaps or go."""
    if not isinstance(name, str):
        return "PLAYER"
    kept = []
    for ch in name.upper():
        if ch.isascii() and (ch.isalnum() or ch in " -_.'"):
            kept.append(ch)
        elif ch.isascii():
            kept.append(" ")
    return " ".join("".join(kept).split())[:NAME_MAX].strip() or "PLAYER"


def clean_sid(sid):
    return re.sub(r"[^a-z0-9]", "", str(sid).lower())[:SID_MAX]


def clean_code(text):
    """A room code as typed or read from a topic, or None if it is not one (it ends up inside a topic: nothing else gets in)."""
    code = str(text).strip().upper()
    return code if len(code) == CODE_LEN and all(ch in ALPHABET for ch in code) else None


def new_code(rng):
    return "".join(rng.choice(ALPHABET) for _ in range(CODE_LEN))


# --- the messages ---------------------------------------------------------------------------------------------------------------------
def _join(m):
    return {"name": clean_name(m.get("name"))}


def _welcome(m):
    return {"name": clean_name(m.get("name")), "pace": _int(m["pace"], 1, 3), "target": _int(m["target"], 1, TARGET_MAX),
            "sid": clean_sid(m["sid"])}


def _hit(m):
    return {"n": _int(m["n"], 1, 10 ** 6), "v": _num(m["v"], V_MIN, V_MAX),
            "start": _triple(m["start"], ((-X_MAX, X_MAX), (Y_MIN, Y_MAX), (Z_MIN, Z_MAX))), "aim": _triple(m["aim"], ((0, 1), (0, 1))),
            "top": _num(m["top"], -1, 1), "side": _num(m["side"], -1, 1), "loft": _num(m["loft"], 0, LOFT_MAX), "serve": _bool(m["serve"]),
            "rally": _int(m.get("rally", 0), 0, 10 ** 4), **_score(m)}


def _score(m):
    """The score as (host, guest), when a message carries it."""
    if "score" not in m:
        return {}
    score = m["score"]
    if not isinstance(score, list) or len(score) != 2:
        raise ValueError("score")
    return {"score": [_int(score[0], 0, 99), _int(score[1], 0, 99)]}


def _miss(m):
    score = m["score"]
    if not isinstance(score, list) or len(score) != 2:
        raise ValueError("score")
    if m["reason"] not in ("miss", "fault"):
        raise ValueError("reason")
    return {"n": _int(m["n"], 1, 10 ** 6), "reason": m["reason"], "score": [_int(score[0], 0, 99), _int(score[1], 0, 99)]}


def _ping(m):
    return {"k": _int(m["k"], 0, 10 ** 9), "ts": _num(m["ts"], -1e12, 1e12)}


SCHEMAS = {
    "join": _join, "welcome": _welcome, "busy": lambda m: {}, "hit": _hit, "miss": _miss,
    "pos": lambda m: {"x": _num(m["x"], -X_MAX, X_MAX)}, "ping": _ping, "pong": _ping,
    "bye": lambda m: {"reason": re.sub(r"[^a-z ]", "", str(m.get("reason", "")).lower())[:REASON_MAX]}, "rematch": lambda m: {},
}


def encode(message):
    """A message dict as the bytes that go on the wire."""
    return json.dumps(dict(message, ver=VERSION), separators=(",", ":")).encode()


def decode(raw):
    """The bytes of a message as a cleaned dict ({"t": kind, ...fields, "sid" if it has one}), or None if it is not a good one."""
    try:
        if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_BYTES:
            return None
        m = json.loads(bytes(raw).decode("utf-8"), parse_constant=_reject)
        if not isinstance(m, dict) or m.get("ver") != VERSION or m.get("t") not in SCHEMAS:
            return None
        out = {"t": m["t"], **SCHEMAS[m["t"]](m)}
        if "sid" in m and m["t"] != "welcome":
            out["sid"] = clean_sid(m["sid"])
        return out
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, RecursionError):
        return None


# --- rooms and topics -----------------------------------------------------------------------------------------------------------------
def lobby_topic(code):
    return f"{NAMESPACE}/lobby/{code}"


def lobby_filter():
    return f"{NAMESPACE}/lobby/+"


def room_topic(code, role):
    return f"{NAMESPACE}/room/{code}/{role}"


def other_role(role):
    return "guest" if role == "host" else "host"


def code_from_topic(topic):
    prefix = f"{NAMESPACE}/lobby/"
    return clean_code(topic[len(prefix):]) if topic.startswith(prefix) else None


def lobby_entry(code, host, pace, target, now=None):
    """The retained message that says a game is open: who is hosting, at which pace, to how many points."""
    return json.dumps({"ver": VERSION, "code": code, "host": clean_name(host), "pace": pace, "target": target,
                       "t": int(time.time() if now is None else now)}, separators=(",", ":")).encode()


def read_lobby(raw):
    """What a lobby message says ({code, host, pace, target, t}), or None for an empty (closed) or bad one."""
    try:
        m = json.loads(bytes(raw).decode("utf-8"), parse_constant=_reject)
        if not isinstance(m, dict) or m.get("ver") != VERSION or clean_code(m.get("code", "")) is None:
            return None
        return {"code": m["code"], "host": clean_name(m.get("host")), "pace": _int(m["pace"], 1, 3), "target": _int(m["target"], 1, TARGET_MAX),
                "t": _int(m["t"], 0, 10 ** 11)}
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, AttributeError):
        return None
