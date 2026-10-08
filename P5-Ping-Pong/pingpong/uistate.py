"""What a menu screen is told in order to be drawn (flow.py makes it, screens.py draws it) and what the results screen shows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UiState:
    screen: str = "TITLE"            # INTRO | TITLE | MODE | OPPONENT | VS | RESULTS
    t_s: float = 0.0                 # seconds since this screen came up (what the animations run on)
    cursor: tuple | None = None      # (a, b): where the hand points over the screen (across, up; 0..1), None when it is not seen
    hover: str | None = None         # the button the hand is on
    progress: float = 0.0            # how far the hold on it has come, 0..1
    focus: int = 0                   # the choice the keys and the cards have in focus
    mode: str = "survival"           # the game chosen so far: survival (a rally) | match
    level_tag: int = 1               # the opponent chosen so far: 1 Rookie, 2 Club, 3 Pro


@dataclass(frozen=True)
class Results:
    won: bool | None = None          # a match: did the player win; a rally: None
    title: str = "GAME OVER"         # the big words
    new_record: bool = False
    stats: tuple = ()                # ((label, text), ...): hits, longest rally, top speed, time


def results_from_summary(summary, new_record):
    """The results screen for a finished game (the summary Session makes of it); `new_record`: a record fell during it."""
    match = summary["mode"] == "match"
    won = (summary["winner"] == "player") if match else None
    title = ("YOU WIN!" if won else "THE CPU WINS") if match else "NEW RECORD!" if new_record else "GAME OVER"
    seconds = int(summary["duration_s"])
    speed = f"{summary['max_kmh']:.0f} km/h" if summary["max_kmh"] > 0 else "-"
    return Results(won=won, title=title, new_record=new_record,
                   stats=(("HITS", str(summary["hits"])), ("LONGEST RALLY", str(summary["streak"])), ("TOP SPEED", speed),
                          ("TIME", f"{seconds // 60}:{seconds % 60:02d}")))
