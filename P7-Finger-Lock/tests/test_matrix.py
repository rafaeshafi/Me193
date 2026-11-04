from lock import matrix as m


def test_frame_bytes_length_and_levels():
    b = m.to_bytes(m.slots([2, 5], 3, 0.5))
    assert len(b) == 104 and max(b) <= 7


def test_slots_show_entered_digits_and_progress():
    f = m.slots([2], candidate=5, progress=1.0)
    assert f[1][1:4] == [7, 7, 7]          # top row of the '2' glyph in slot 0
    assert f[1][5:8] == [3, 3, 3]          # candidate 5 glows mid in slot 1
    assert f[5][9:12] == [1, 1, 1]         # empty slot 2 is a dim underline
    assert all(v == 7 for v in f[7])       # full progress bar


def test_pictures_per_state():
    for state, rows in ((m.UNLOCKED, m.TICK), (m.DENIED, m.CROSS), (m.LOCKOUT, m.SKULL)):
        f = m.picture({"state": state, "entered": []}, 0.0)
        assert f == [[7 if ch == "1" else 0 for ch in row] for row in rows]
    idle = m.picture({"state": m.IDLE, "entered": []}, 0.0)
    assert sum(sum(r) for r in idle) == 1


def test_listening_picture_has_countdown():
    f = m.picture({"state": m.LISTENING, "entered": [], "candidate": 0, "progress": 0.0, "remaining": 5, "timeout_total": 10}, 0.0)
    assert f[7][:6] == [1] * 6 and f[7][7:] == [0] * 6


def test_text_rendering():
    t = m.to_text(m.slots([1, 2, 3]))
    assert t.count("\n") == 7 and len(t.split("\n")[0]) == 13
