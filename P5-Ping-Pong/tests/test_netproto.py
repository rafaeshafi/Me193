"""What two laptops say to each other in an online game: small JSON messages, strictly checked (they arrive over a public broker)."""

import json
import math
import random

import pytest

from pingpong import netproto as proto


def roundtrip(message):
    return proto.decode(proto.encode(message))


HIT = {"t": "hit", "n": 3, "v": 4.2, "start": [0.1, 0.22, 0.3], "aim": [0.4, 0.5], "top": 0.3, "side": -0.2, "loft": 0.05, "serve": False,
       "rally": 4}


def test_every_kind_of_message_survives_the_wire():
    for message in (
        {"t": "join", "name": "MAYA", "gid": "k3x9"},
        {"t": "welcome", "name": "RAFAE", "pace": 2, "target": 7, "sid": "ab12cd", "to": "k3x9"},
        {"t": "busy", "to": "k3x9"},
        HIT,
        {"t": "miss", "n": 4, "reason": "miss", "score": [3, 2]},
        {"t": "pos", "x": 0.31},
        {"t": "ping", "k": 7, "ts": 12.5},
        {"t": "pong", "k": 7, "ts": 12.5},
        {"t": "bye", "reason": "left"},
        {"t": "bye", "reason": "lost", "gid": "k3x9"},
        {"t": "rematch", "n": 9},
    ):
        got = roundtrip(message)
        assert got is not None and got["t"] == message["t"], message
        for key, value in message.items():
            assert got[key] == pytest.approx(value) if isinstance(value, float) else got[key] == value, (message["t"], key)


def test_messages_are_small_compact_json_with_a_version():
    wire = proto.encode(HIT)
    assert len(wire) < 300 and json.loads(wire)["ver"] == proto.VERSION and b" " not in wire.replace(b'"name"', b"")


@pytest.mark.parametrize("junk", [b"", b"not json", b"[]", b"42", b'{"t": "hit"}', b'{"ver": 1}', b'{"ver": 1, "t": "nonsense"}',
                                   b'{"ver": 99, "t": "pos", "x": 0.1}', b'{"ver": 1, "t": "pos", "x": "far"}', b'{"ver": 1, "t": "pos"}',
                                   b'{"ver": 1, "t": "pos", "x": NaN}', b'{"ver": 1, "t": "pos", "x": Infinity}', b"\xff\xfe\x00",
                                   b'{"ver": 1, "t": "hit", "n": 1, "v": 3.0}', b"x" * 5000, b'{"v": 1, "t": "pos", "x": 0.1}'])
def test_anything_that_is_not_a_good_message_is_dropped_never_raised(junk):
    assert proto.decode(junk) is None


def test_a_hit_is_checked_field_by_field():
    for broken in (dict(HIT, n=0), dict(HIT, n="1"), dict(HIT, v=float("nan")), dict(HIT, start=[0.0, 0.2]), dict(HIT, start="here"),
                   dict(HIT, aim=[0.5]), dict(HIT, top=None), dict(HIT, serve="yes"), {k: v for k, v in HIT.items() if k != "aim"}):
        assert roundtrip(broken) is None, broken


def test_numbers_that_are_too_big_or_small_are_pulled_into_range_not_trusted():
    got = roundtrip(dict(HIT, v=500.0, start=[9.0, -4.0, 99.0], aim=[-3.0, 8.0], top=5.0, side=-5.0, loft=3.0))
    assert got["v"] == proto.V_MAX and got["start"][0] == proto.X_MAX and got["start"][1] == proto.Y_MIN and got["start"][2] == proto.Z_MAX
    assert got["aim"] == [0.0, 1.0] and got["top"] == 1.0 and got["side"] == -1.0 and got["loft"] == proto.LOFT_MAX
    assert roundtrip(dict(HIT, v=0.01))["v"] == proto.V_MIN
    assert -proto.X_MAX <= roundtrip({"t": "pos", "x": -50.0})["x"] <= proto.X_MAX


def test_names_are_cleaned_and_cut_so_nobody_can_flood_the_screen_or_inject_anything():
    join = lambda name: roundtrip({"t": "join", "name": name, "gid": "g1"})                    # noqa: E731
    assert join("  Maya\n\t<b>x</b>  ")["name"] == "MAYA B X B"
    assert join("A" * 100)["name"] == "A" * proto.NAME_MAX
    assert join("")["name"] == "PLAYER" and join(5)["name"] == "PLAYER"
    assert join("rémi ✓")["name"] == "RMI"                                                     # what the typeface can draw


def test_pace_and_target_have_to_be_a_real_level_and_a_sensible_match():
    welcome = dict(t="welcome", name="A", sid="x", to="g1")
    assert roundtrip(dict(welcome, pace=9, target=7)) is None
    assert roundtrip(dict(welcome, pace=2, target=0)) is None
    assert roundtrip(dict(welcome, pace=3, target=11))["target"] == 11


def test_a_miss_says_why_and_the_score_as_host_then_guest():
    assert roundtrip({"t": "miss", "n": 1, "reason": "cheated", "score": [1, 0]}) is None
    assert roundtrip({"t": "miss", "n": 1, "reason": "fault", "score": [1]}) is None
    assert roundtrip({"t": "miss", "n": 1, "reason": "fault", "score": [1, -2]}) is None


def test_the_session_id_travels_when_there_is_one_and_is_cleaned():
    assert roundtrip(dict(HIT, sid="a1b2c3"))["sid"] == "a1b2c3"
    assert "sid" not in roundtrip(HIT)
    assert roundtrip(dict(HIT, sid="x" * 80))["sid"] == "x" * proto.SID_MAX


def test_a_guest_names_itself_so_that_the_answer_can_be_told_from_the_one_for_another_guest():
    assert roundtrip({"t": "join", "name": "MAYA", "gid": "K3-X9!"})["gid"] == "k3x9"            # cleaned like a session id
    for broken in ({"t": "join", "name": "MAYA"}, {"t": "join", "name": "MAYA", "gid": "!!"}, {"t": "join", "name": "MAYA", "gid": 7},
                   {"t": "busy"}, {"t": "busy", "to": ""}, {"t": "welcome", "name": "A", "pace": 1, "target": 7, "sid": "x"}):
        assert roundtrip(broken) is None, broken
    assert roundtrip({"t": "bye", "reason": "lost", "gid": "k3x9"})["gid"] == "k3x9"
    assert "gid" not in roundtrip({"t": "bye", "reason": "left"})


def test_a_rematch_is_numbered_like_a_hit_so_that_a_copy_of_it_is_not_taken_for_another():
    assert roundtrip({"t": "rematch", "n": 5})["n"] == 5
    assert roundtrip({"t": "rematch"}) is None and roundtrip({"t": "rematch", "n": 0}) is None


# --- rooms and topics -----------------------------------------------------------------------------------------------------------------------
def test_a_room_code_is_five_letters_that_cannot_be_mistaken_for_each_other_and_is_random_enough():
    codes = {proto.new_code(random.Random(k)) for k in range(500)}
    assert len(codes) > 480
    assert all(len(c) == proto.CODE_LEN and set(c) <= set(proto.ALPHABET) for c in codes)
    assert not set("ILO01") & set(proto.ALPHABET)


def test_every_topic_is_in_the_games_own_corner_of_the_broker_never_the_official_score_topic():
    code = "ABCDE"
    topics = [proto.lobby_topic(code), proto.lobby_filter(), proto.room_topic(code, "host"), proto.room_topic(code, "guest")]
    assert all(t.startswith("ME193-pp/") and not t.startswith("ME193/") for t in topics)
    assert proto.lobby_topic(code) in {t for t in topics} and proto.lobby_topic(code).endswith(code)
    assert proto.code_from_topic(proto.lobby_topic(code)) == code and proto.code_from_topic("elsewhere/ABCDE") is None
    assert proto.other_role("host") == "guest" and proto.other_role("guest") == "host"


def test_a_code_is_cleaned_before_it_becomes_part_of_a_topic():
    assert proto.clean_code("  abcde \n") == "ABCDE"
    assert proto.clean_code("ab/cde") is None and proto.clean_code("ABC") is None and proto.clean_code("ABCDEF") is None
    assert proto.clean_code("ABC#E") is None and proto.clean_code("AB+DE") is None


def test_a_lobby_entry_names_the_host_the_pace_and_the_target():
    entry = proto.lobby_entry("ABCDE", "RAFAE", 2, 7, now=1_700_000_000)
    got = proto.read_lobby(entry)
    assert got == {"code": "ABCDE", "host": "RAFAE", "pace": 2, "target": 7, "t": 1_700_000_000}
    assert proto.read_lobby(b"") is None and proto.read_lobby(b"junk") is None
    assert proto.read_lobby(json.dumps({"ver": 1, "code": "AB/DE", "host": "x", "pace": 2, "target": 7, "t": 1}).encode()) is None
    assert proto.read_lobby(proto.lobby_entry("ABCDE", "R", 9, 7, now=1)) is None


def test_nothing_in_here_is_ever_not_a_number():
    got = roundtrip(dict(HIT, v=4.0))
    assert all(math.isfinite(x) for x in [got["v"], *got["start"], *got["aim"], got["top"], got["side"], got["loft"]])
