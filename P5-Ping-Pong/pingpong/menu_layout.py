"""Where the buttons of every menu screen are, as fractions of the screen (x0, y0, x1, y1; y down), and how long each takes to
hold.  flow.py presses them with the hand and screens.py draws them, so both read this one table."""

from pingpong import holdstart

START_HOLD_S, CHOICE_HOLD_S, BACK_HOLD_S = holdstart.HOLD_S, 1.2, 1.0

# the top right, from the START button's corner out to the edges of the screen (reaching past it counts), and its mirror
START_HIT = (holdstart.A_MIN, 0.0, 1.0, 1.0 - holdstart.B_MIN)
CHANGE_HIT = (0.0, 0.0, 1.0 - holdstart.A_MIN, 1.0 - holdstart.B_MIN)
BACK_HIT = (0.0, 0.86, 0.19, 1.0)

MODE_CARDS = {"rally": (0.10, 0.25, 0.46, 0.86), "match": (0.54, 0.25, 0.90, 0.86)}
OPPONENT_CARDS = {1: (0.085, 0.24, 0.355, 0.86), 2: (0.365, 0.24, 0.635, 0.86), 3: (0.645, 0.24, 0.915, 0.86)}
BACK_BUTTON = (0.03, 0.89, 0.17, 0.97)                      # where the BACK pill is drawn (the hit area above is larger)

TARGETS = {
    "INTRO": (("skip", START_HIT, START_HOLD_S),),
    "TITLE": (("start", START_HIT, START_HOLD_S),),
    "MODE": tuple((name, rect, CHOICE_HOLD_S) for name, rect in MODE_CARDS.items()) + (("back", BACK_HIT, BACK_HOLD_S),),
    "OPPONENT": tuple((f"opp{tag}", rect, CHOICE_HOLD_S) for tag, rect in OPPONENT_CARDS.items()) + (("back", BACK_HIT, BACK_HOLD_S),),
    "RESULTS": (("again", START_HIT, START_HOLD_S), ("change", CHANGE_HIT, CHOICE_HOLD_S)),
}


def targets(screen):
    """The buttons of a screen as (id, rect, hold seconds); a screen with none (the face-off, a game) has an empty tuple."""
    return TARGETS.get(screen, ())
