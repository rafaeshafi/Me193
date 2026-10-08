"""Where the buttons of every menu screen are, as fractions of the screen (x0, y0, x1, y1; y down), and how long each takes to
hold.  flow.py presses them with the hand and screens.py draws them, so both read this one table."""

from pingpong import holdstart

START_HOLD_S, CHOICE_HOLD_S, BACK_HOLD_S = holdstart.HOLD_S, 1.2, 1.0

# the top right, from the START button's corner out to the edges of the screen (reaching past it counts), and its mirror
START_HIT = (holdstart.A_MIN, 0.0, 1.0, 1.0 - holdstart.B_MIN)
CHANGE_HIT = (0.0, 0.0, 1.0 - holdstart.A_MIN, 1.0 - holdstart.B_MIN)
BACK_HIT = (0.0, 0.86, 0.19, 1.0)

MODE_CARDS = {"rally": (0.085, 0.25, 0.355, 0.86), "match": (0.365, 0.25, 0.635, 0.86), "online": (0.645, 0.25, 0.915, 0.86)}
OPPONENT_CARDS = {1: (0.085, 0.24, 0.355, 0.86), 2: (0.365, 0.24, 0.635, 0.86), 3: (0.645, 0.24, 0.915, 0.86)}
BACK_BUTTON = (0.03, 0.89, 0.17, 0.97)                      # where the BACK pill is drawn (the hit area above is larger)

# playing a friend: a card to host a game (with a chip for each pace along its bottom) and the open games to join, one row each
HOST_CARD = (0.06, 0.24, 0.40, 0.86)
HOST_CHIPS = {tag: (0.075 + 0.1 * (tag - 1), 0.64, 0.165 + 0.1 * (tag - 1), 0.82) for tag in (1, 2, 3)}
ROOM_ROWS = tuple((0.46, y, 0.94, y + 0.19) for y in (0.24, 0.45, 0.66))
MAX_ROOMS = len(ROOM_ROWS)

TARGETS = {
    "INTRO": (("skip", START_HIT, START_HOLD_S),),
    "TITLE": (("start", START_HIT, START_HOLD_S),),
    "MODE": tuple((name, rect, CHOICE_HOLD_S) for name, rect in MODE_CARDS.items()) + (("back", BACK_HIT, BACK_HOLD_S),),
    "OPPONENT": tuple((f"opp{tag}", rect, CHOICE_HOLD_S) for tag, rect in OPPONENT_CARDS.items()) + (("back", BACK_HIT, BACK_HOLD_S),),
    "WAIT": (("cancel", BACK_HIT, BACK_HOLD_S),),
}


def targets(screen, rooms=0, can_again=True):
    """The buttons of a screen as (id, rect, hold seconds); a screen with none (the face-off, a game) has an empty tuple.  `rooms`: how
    many open games the online list has (each of the first three can be pointed at); `can_again`: the results offer a play again."""
    if screen == "ONLINE":
        chips = tuple((f"host{tag}", rect, CHOICE_HOLD_S) for tag, rect in HOST_CHIPS.items())
        rows = tuple((f"join{k}", ROOM_ROWS[k], CHOICE_HOLD_S) for k in range(min(rooms, MAX_ROOMS)))
        return chips + rows + (("back", BACK_HIT, BACK_HOLD_S),)
    if screen == "RESULTS":
        return ((("again", START_HIT, START_HOLD_S),) if can_again else ()) + (("change", CHANGE_HIT, CHOICE_HOLD_S),)
    return TARGETS.get(screen, ())
