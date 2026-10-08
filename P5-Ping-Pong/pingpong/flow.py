"""The way into a game, like a console sports game: an intro flying down to the court, a title, a choice of game and of
opponent, a face-off, and after the match its results.

    INTRO -> TITLE -> MODE -> OPPONENT -> VS -> (the game) -> RESULTS -> (again | OPPONENT)

Every screen can be pointed at with the hub and held on (dwell.py), driven with a few keys, or short-cut with the AprilTag cards:
the START card, or SPACE, starts a game at once with the choices so far (the assignment's own way in), and a LEVEL card picks
the opponent.  The flow decides nothing about the game itself: it hands back Actions (start a game with this level and mode,
play this sound, buzz the hub, change the music) that the Session carries out, and it stays out of the way while a game is on.
Pure logic with the clock passed in.
"""

from collections import namedtuple

from pingpong import dwell, menu_layout
from pingpong.uistate import UiState

S = 1_000_000_000
INTRO_S = 10.0
VS_S = 2.4
TICK_GAP_S = 0.35                 # the hover tick (sound and buzz) is not made more often than this
GAME_GRACE_S = 0.25               # a game that has only just been started has not reached its phase yet

Action = namedtuple("Action", "kind value")          # kind: start (level_tag, mode) | sound | haptic | music (name or None)

SPACE, ENTER, ESC, Q = 32, 13, 27, ord("q")
QUIT_KEYS = (ESC, Q)
BACK_KEYS = (8, 127, ord("b"))
LEFT_KEYS, RIGHT_KEYS = (2, ord(",")), (3, ord("."))               # the arrow keys come through the window as 2 and 3; , and . do the same
LEVEL_KEYS = {ord("1"): 1, ord("2"): 2, ord("3"): 3}
MUSIC = {"INTRO": "intro", "TITLE": "menu", "MODE": "menu", "OPPONENT": "menu", "RESULTS": "menu", "VS": None, "GAME": None}
CHOICES = {"MODE": 2, "OPPONENT": 3, "RESULTS": 2}  # how many things the keys can move along on a screen


class Flow:
    def __init__(self, *, intro=True, level_tag=1, mode="survival", intro_s=INTRO_S, vs_s=VS_S):
        self.mode, self.level_tag = mode, level_tag
        self.intro_s, self.vs_s = intro_s, vs_s
        self.pointer = dwell.Pointer()
        self.screen = "INTRO" if intro else "TITLE"
        self.focus = self._default_focus()
        self._t0 = self._last_ns = self._last_tick_ns = None
        self._hover = None
        self._music = "unset"

    @property
    def active(self):
        return self.screen != "GAME"

    def _default_focus(self):
        return {"MODE": 0 if self.mode == "survival" else 1, "OPPONENT": self.level_tag - 1}.get(self.screen, 0)

    # --- moving between screens ---------------------------------------------------------------------------------------------------
    def _go(self, screen, now_ns, *extra):
        self.screen, self._t0, self._hover = screen, now_ns, None
        self.pointer.reset()
        self.focus = self._default_focus()
        return list(extra) + self._music_for(screen) + ([Action("sound", "vs")] if screen == "VS" else [])

    def _music_for(self, screen):
        want = MUSIC[screen]
        if want == self._music:
            return []
        self._music = want
        return [Action("music", want)]

    def _launch(self, now_ns):
        """Start a game now with the choices so far."""
        return self._go("GAME", now_ns, Action("start", (self.level_tag, self.mode)))

    def started(self, ok):
        """The Session says whether the game really started; if it could not (the hub is lost, say) it is the title again."""
        if not ok and self.screen == "GAME":
            return self._go("TITLE", self._last_ns or 0)
        return []

    # --- the clock ---------------------------------------------------------------------------------------------------------------------
    def update(self, now_ns, ab, phase):
        """Called every frame: ab is the hand's (a, b) in the reach box (None when it is not seen), phase the game's."""
        self._last_ns = now_ns
        actions = []
        if self._t0 is None:
            self._t0 = now_ns
            actions += self._music_for(self.screen)
        if self.screen == "GAME":
            if phase == "MATCH_OVER" and now_ns - self._t0 > GAME_GRACE_S * S:
                actions += self._go("RESULTS", now_ns)
            return actions
        if self.screen == "RESULTS" and phase != "MATCH_OVER":
            return actions + self._go("GAME", now_ns)                       # something else started the next game
        pressed = self.pointer.update(now_ns, ab, menu_layout.targets(self.screen), True)
        actions += self._tick_if_new_hover(now_ns)
        if pressed:
            actions += self._press(pressed, now_ns)
        elif self.screen == "INTRO" and now_ns - self._t0 >= self.intro_s * S:
            actions += self._go("TITLE", now_ns)
        elif self.screen == "VS" and now_ns - self._t0 >= self.vs_s * S:
            actions += self._launch(now_ns)
        return actions

    def _tick_if_new_hover(self, now_ns):
        hovered = self.pointer.hovered
        fresh = hovered is not None and hovered != self._hover
        self._hover = hovered
        if fresh and (self._last_tick_ns is None or now_ns - self._last_tick_ns >= TICK_GAP_S * S):
            self._last_tick_ns = now_ns
            return [Action("sound", "menu_tick"), Action("haptic", "menu_tick")]
        return []

    def _selected(self):
        return [Action("sound", "menu_select"), Action("haptic", "menu_select")]

    def _press(self, target, now_ns):
        if target == "back":
            return self._go("TITLE" if self.screen == "MODE" else "MODE", now_ns, Action("sound", "menu_back"))
        if target == "skip":
            return self._go("TITLE", now_ns, *self._selected())
        if target == "start":
            return self._go("MODE", now_ns, *self._selected())
        if target in ("rally", "match"):
            self.mode = "survival" if target == "rally" else "match"
            return self._go("OPPONENT", now_ns, *self._selected())
        if target.startswith("opp"):
            self.level_tag = int(target[3:])
            return self._go("VS", now_ns, *self._selected())
        if target == "again":
            return self._launch(now_ns) + self._selected()
        return self._go("OPPONENT", now_ns, *self._selected())               # "change"

    # --- keys and cards ---------------------------------------------------------------------------------------------------------------------
    def on_key(self, key, now_ns):
        """-> the actions for a key the flow takes, or None for a key it leaves to the game (and to quitting)."""
        self._last_ns = now_ns
        if self.screen == "GAME" or key in QUIT_KEYS:
            return None
        if self.screen == "INTRO":
            return self._go("TITLE", now_ns)
        if self.screen == "VS":
            if key == SPACE:
                return self._launch(now_ns)
            return self._go("OPPONENT", now_ns, Action("sound", "menu_back")) if key in BACK_KEYS else []
        if key == SPACE:
            return self._launch(now_ns)
        if key == ENTER:
            return self._enter(now_ns)
        if key in BACK_KEYS:
            return self._back(now_ns)
        if key in LEFT_KEYS + RIGHT_KEYS:
            return self._move(1 if key in RIGHT_KEYS else -1)
        if key in LEVEL_KEYS:
            return self._choose_level(LEVEL_KEYS[key])
        if key == ord("m"):
            self.mode = "match" if self.mode == "survival" else "survival"
            self.focus = self._default_focus() if self.screen == "MODE" else self.focus
            return [Action("sound", "menu_tick")]
        return None

    def _enter(self, now_ns):
        if self.screen == "TITLE":
            return self._go("MODE", now_ns, *self._selected())
        if self.screen == "MODE":
            self.mode = "survival" if self.focus == 0 else "match"
            return self._go("OPPONENT", now_ns, *self._selected())
        if self.screen == "OPPONENT":
            self.level_tag = self.focus + 1
            return self._go("VS", now_ns, *self._selected())
        return self._launch(now_ns) if self.focus == 0 else self._go("OPPONENT", now_ns, *self._selected())

    def _back(self, now_ns):
        goes = {"MODE": "TITLE", "OPPONENT": "MODE", "RESULTS": "OPPONENT"}.get(self.screen)
        return [] if goes is None else self._go(goes, now_ns, Action("sound", "menu_back"))

    def _move(self, step):
        count = CHOICES.get(self.screen)
        if count is None:
            return []
        self.focus = max(0, min(count - 1, self.focus + step))
        return [Action("sound", "menu_tick")]

    def _choose_level(self, tag):
        self.level_tag = tag
        if self.screen == "OPPONENT":
            self.focus = tag - 1
        return [Action("sound", "menu_tick")]

    def on_tag(self, role, value, now_ns):
        """-> the actions for an AprilTag card the flow takes, or None while a game is on."""
        self._last_ns = now_ns
        if self.screen == "GAME":
            return None
        if role == "START":
            if self.screen == "INTRO":
                return self._go("TITLE", now_ns)
            return [] if self.screen == "VS" else self._launch(now_ns)
        if role == "LEVEL" and value in (1, 2, 3):
            return self._choose_level(value)
        return []

    # --- what the screen is told ------------------------------------------------------------------------------------------------------
    def ui_state(self, now_ns, ab):
        return UiState(screen=self.screen, t_s=0.0 if self._t0 is None else max(0.0, (now_ns - self._t0) / S), cursor=ab,
                       hover=self.pointer.hovered, progress=self.pointer.progress, focus=self.focus, mode=self.mode,
                       level_tag=self.level_tag)
