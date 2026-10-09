"""What the UNO Q's LED matrix (8 rows x 13 columns, 8 brightness levels) shows of the game.

Without a game going it alternates, every three seconds, between the streak you are on (the last game's, until the next starts) and
your best, each as big 3 x 5 digits with a letter (C, B) beside it.  In a Rally it is the streak in big digits with a bar along the
bottom that fills towards your best (and flashes once you are at it); in a Match it is the points, yours, a dash, the computer's.  A
scene is a small tuple (`scene_of`), a frame is the 104 levels of the matrix row by row (`frame`): the game works out both, the board
only draws what it is sent.
"""

ROWS, COLS = 8, 13
FULL, BAR, LABEL, FLASH_DIM, TRACK = 7, 5, 3, 2, 1             # brightness levels
CYCLE_S, FLASH_S = 6.0, 0.5                                   # idle: current for half of this, then the best; a bar at your best flashes
IN_GAME = ("COUNTDOWN", "RALLY", "POINT_OVER")

DIGITS = (("###", "#.#", "#.#", "#.#", "###"), (".#.", "##.", ".#.", ".#.", "###"), ("###", "..#", "###", "#..", "###"),
          ("###", "..#", "###", "..#", "###"), ("#.#", "#.#", "###", "..#", "..#"), ("###", "#..", "###", "..#", "###"),
          ("###", "#..", "###", "#.#", "###"), ("###", "..#", "..#", "..#", "..#"), ("###", "#.#", "###", "#.#", "###"),
          ("###", "#.#", "###", "..#", "###"))
NARROW_ONE = (".#", "##", ".#", ".#", ".#")                    # a 1 two columns wide, for the scores that are tight
LETTERS = {"C": ("###", "#..", "#..", "#..", "###"), "B": ("##.", "#.#", "##.", "#.#", "##.")}


def scene_of(state):
    """What the matrix should show for the game's HudState: ("idle", streak, best) | ("rally", streak, best) | ("match", you, cpu)."""
    if state.screen == "GAME" and state.phase in IN_GAME:
        if state.mode == "match":
            return ("match", state.player_points, state.cpu_points)
        return ("rally", state.streak, state.record)
    return ("idle", state.streak, state.record)


def frame(scene, t_s):
    """The 104 brightness levels (0..7, row by row) for a scene at time t_s (what moves on it: the alternation and the flash)."""
    canvas = [[0] * COLS for _ in range(ROWS)]
    kind = scene[0]
    if kind == "idle":
        _idle(canvas, scene[1], scene[2], t_s)
    elif kind == "rally":
        _rally(canvas, scene[1], scene[2], t_s)
    elif kind == "match":
        for left, rows in score_items(scene[1], scene[2]):
            _blit(canvas, rows, left, 1, FULL)
    else:
        for col in (4, 6, 8):
            canvas[3][col] = FLASH_DIM
    return tuple(v for row in canvas for v in row)


def _text(value, narrow=False):
    """The digits of a number as five rows of # and . with a blank column between them (a 1 is narrow when asked)."""
    digits = str(min(max(int(value), 0), 999))
    glyphs = [NARROW_ONE if narrow and d == "1" else DIGITS[int(d)] for d in digits]
    return tuple(".".join(g[r] for g in glyphs) for r in range(5))


def _blit(canvas, rows, left, top, level):
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#" and 0 <= top + r < ROWS and 0 <= left + c < COLS:
                canvas[top + r][left + c] = level


def _idle(canvas, current, best, t_s):
    showing_best = (t_s % CYCLE_S) >= CYCLE_S / 2
    value = best if showing_best else current
    number = _text(value)
    if len(number[0]) > 7:                                               # three digits leave no room for the letter
        _blit(canvas, number, (COLS - len(number[0])) // 2, 1, FULL)
        return
    _blit(canvas, LETTERS["B" if showing_best else "C"], 0, 1, LABEL)
    _blit(canvas, number, COLS - len(number[0]), 1, FULL)


def _rally(canvas, streak, best, t_s):
    number = _text(streak)
    _blit(canvas, number, (COLS - len(number[0])) // 2, 0, FULL)
    at_best = best > 0 and streak >= best
    filled = min(COLS, int(COLS * streak / best + 0.5)) if best > 0 else 0
    level = (FULL if (t_s % FLASH_S) < FLASH_S / 2 else FLASH_DIM) if at_best else BAR
    for row in (6, 7):
        for col in range(COLS):
            canvas[row][col] = level if col < filled else TRACK


def score_items(you, cpu):
    """A match's scoreboard as [(left column, rows of # and .)]: your points, what is between, the computer's.  The digits are three
    columns wide; a 1 is two wide when the two numbers would not fit otherwise."""
    for narrow in (False, True):
        mine, theirs = _text(you, narrow), _text(cpu, narrow)
        wide, wide_them = len(mine[0]), len(theirs[0])
        if wide + wide_them + 5 <= COLS:
            break
    gap = min(5, COLS - wide - wide_them)                                # what is between them
    if gap >= 3:
        between = tuple("." + "#" * (gap - 2) + "." if r == 2 else "." * gap for r in range(5))            # a dash with a blank column either side
    else:
        between = tuple("#" + "." * (gap - 1) if r in (1, 3) else "." * gap for r in range(5))
    left = max(0, (COLS - wide - gap - wide_them) // 2)
    return [(left, mine), (left + wide, between), (left + wide + gap, theirs)]
