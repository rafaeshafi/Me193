"""Pictures for the UNO Q's 13x8 LED matrix: three digit slots on top, a hold-progress bar at the bottom, a tick, a
cross, a skull. A frame is 104 brightness levels 0..7, row by row, the bytes the sketch's `draw` provider takes."""
from lock.policy import DENIED, IDLE, LISTENING, LOCKOUT, UNLOCKED

W, H, LEVELS = 13, 8, 8
BRIGHT, MID, DIM = 7, 3, 1
GLYPHS = {  # 3 wide x 5 tall
    0: ["111", "101", "101", "101", "111"], 1: ["010", "110", "010", "010", "111"], 2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "111", "001", "111"], 4: ["101", "101", "111", "001", "001"], 5: ["111", "100", "111", "001", "111"],
    "_": ["000", "000", "000", "000", "111"], "?": ["111", "001", "011", "000", "010"],
}
TICK = ["0000000000000", "0000000000010", "0000000000100", "0000000001000", "0100000010000", "0010000100000",
        "0001001000000", "0000110000000"]
CROSS = ["0000000000000", "0010000000100", "0001000001000", "0000100010000", "0000010100000", "0000100010000",
         "0001000001000", "0010000000100"]
SKULL = ["0000111110000", "0001111111000", "0011011101100", "0011011101100", "0001111111000", "0000101010000",
         "0000111110000", "0000010100000"]


def blank():
    return [[0] * W for _ in range(H)]


def put_glyph(frame, glyph, x, y, level=BRIGHT):
    for r, row in enumerate(GLYPHS[glyph]):
        for c, ch in enumerate(row):
            if ch == "1" and 0 <= y + r < H and 0 <= x + c < W:
                frame[y + r][x + c] = level


def put_rows(frame, rows, level=BRIGHT):
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                frame[r][c] = level


def slots(entered, candidate=0, progress=0.0, blink_on=True):
    """Three slots: digits already in, '_' for the empty ones; the current candidate glows dim in the next slot."""
    frame = blank()
    for i in range(3):
        x = 1 + i * 4
        if i < len(entered):
            put_glyph(frame, entered[i], x, 1)
        elif i == len(entered) and candidate and blink_on:
            put_glyph(frame, candidate, x, 1, MID)
        else:
            put_glyph(frame, "_", x, 1, DIM)
    lit = int(round(progress * W))
    for c in range(lit):
        frame[7][c] = BRIGHT
    return frame


def idle(t):
    """A dim dot that walks the bottom row, so a glance says the safe is alive and waiting."""
    frame = blank()
    frame[7][int(t * 2) % W] = DIM
    return frame


def countdown(frame, remaining, total):
    """Dim the unused end of the bottom row to show how much entry time is left (over the progress bar)."""
    left = 0 if not total else int(round(W * max(remaining, 0) / total))
    for c in range(left, W):
        if frame[7][c] == 0:
            frame[7][c] = 0
    for c in range(left):
        if frame[7][c] == 0:
            frame[7][c] = DIM
    return frame


def picture(snapshot, t):
    """The frame for a Lock.snapshot() at time t (seconds, for the blinks)."""
    s = snapshot["state"]
    blink = int(t * 4) % 2 == 0
    if s == LISTENING:
        frame = slots(snapshot["entered"], snapshot.get("candidate", 0), snapshot.get("progress", 0.0), blink)
        return countdown(frame, snapshot.get("remaining") or 0, snapshot.get("timeout_total", 0))
    if s == UNLOCKED:
        frame = blank()
        put_rows(frame, TICK)
        return frame
    if s == DENIED:
        frame = blank()
        put_rows(frame, CROSS)
        return frame
    if s == LOCKOUT:
        frame = blank()
        put_rows(frame, SKULL, BRIGHT if blink else MID)
        return frame
    return idle(t)


def to_bytes(frame):
    return bytes(max(0, min(LEVELS - 1, int(v))) for row in frame for v in row)


def to_text(frame):
    return "\n".join("".join(".:-=+*#@"[max(0, min(7, int(v)))] for v in row) for row in frame)
