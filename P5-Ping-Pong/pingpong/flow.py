"""The way into a game, like a console sports game: an intro flying down to the court, a title, a choice of game and of
opponent, a face-off, and after the match its results.

    INTRO -> TITLE -> MODE -> OPPONENT -> VS -> (the game) -> RESULTS -> (again | OPPONENT)
    MODE -> ONLINE -> WAIT -> VS -> (the game) -> RESULTS -> (rematch | ONLINE)                       (playing a friend)

Every screen can be pointed at with the hub and held on (dwell.py), driven with a few keys, or short-cut with the AprilTag cards:
the START card, or SPACE, starts a game at once with the choices so far (the assignment's own way in), and a LEVEL card picks
the opponent.  The flow decides nothing about the game itself: it hands back Actions (start a game with this level and mode,
play this sound, buzz the hub, change the music) that the Session carries out, and it stays out of the way while a game is on.
Pure logic with the clock passed in.
"""

from collections import namedtuple

from pingpong import dwell, menu_layout
from pingpong.uistate import OnlineView, UiState

S = 1_000_000_000
INTRO_S = 10.0
VS_S = 2.4
TICK_GAP_S = 0.35                 # the hover tick (sound and buzz) is not made more often than this
GAME_GRACE_S = 0.25               # a game that has only just been started has not reached its phase yet

Action = namedtuple("Action", "kind value")          # kind: start (level_tag, mode) | sound | haptic | music (name or None) | online (verb, arg)

SPACE, ENTER, ESC, Q = 32, 13, 27, ord("q")
QUIT_KEYS = (ESC, Q)
BACK_KEYS = (8, 127, ord("b"))
LEFT_KEYS, RIGHT_KEYS = (2, ord(",")), (3, ord("."))               # the arrow keys come through the window as 2 and 3; , and . do the same
LEVEL_KEYS = {ord("1"): 1, ord("2"): 2, ord("3"): 3}
MUSIC = {"INTRO": "intro", "TITLE": "menu", "MODE": "menu", "ONLINE": "menu", "WAIT": "menu", "OPPONENT": "menu", "RESULTS": "menu",
         "VS": None, "GAME": None}
CHOICES = {"MODE": 3, "OPPONENT": 3, "RESULTS": 2}  # how many things the keys can move along on a screen (ONLINE: the host card and the games)


class Flow:
    def __init__(self, *, intro=True, level_tag=1, mode="survival", intro_s=INTRO_S, vs_s=VS_S, start=None):
        """start: a way into the game that skips the intro and the menus: ("online",) the list of friends' games, ("host", pace) a
        game opened at once, ("join", code) a game joined at once."""
        self.mode, self.level_tag = mode, level_tag
        self.intro_s, self.vs_s = intro_s, vs_s
        self.pointer = dwell.Pointer()
        self.screen = "INTRO" if intro else "TITLE"
        self._boot = []                                   # what the first frame asks for (the start's actions)
        if start:
            self.screen = {"online": "ONLINE", "host": "WAIT", "join": "WAIT"}[start[0]]
            self._boot = [Action("online", ("open", None))] + ([Action("online", tuple(start))] if start[0] != "online" else [])
            self.level_tag = start[1] if start[0] == "host" else level_tag
        self.focus = self._default_focus()
        self._t0 = self._last_ns = self._last_tick_ns = None
        self._hover, self._hover_code = None, None
        self._music = "unset"
        self.online = OnlineView()               # what the online screens show (set_online)
        self.online_game, self.opponent = False, ""         # a game with a friend: its face-off, results and rematch work differently
        self.rematch_pending = self.opponent_gone = False

    @property
    def active(self):
        return self.screen != "GAME"

    def _default_focus(self):
        return {"MODE": 0 if self.mode == "survival" else 1, "OPPONENT": self.level_tag - 1}.get(self.screen, 0)

    # --- moving between screens ---------------------------------------------------------------------------------------------------
    def _go(self, screen, now_ns, *extra):
        self.screen, self._t0, self._hover, self._hover_code = screen, now_ns, None, None
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
            return self._leave(self._last_ns or 0) if self.online_game else self._go("TITLE", self._last_ns or 0)
        return []

    # --- playing a friend: what the Session tells the flow ---------------------------------------------------------------------------
    def set_online(self, view):
        self.online = view or OnlineView()

    def paired(self, name, pace, now_ns):
        """A friend is found (they joined, or we did): the face-off, then a match at the host's pace."""
        self.online_game, self.opponent, self.level_tag, self.mode = True, name, pace, "match"
        self.rematch_pending = self.opponent_gone = False
        return self._go("VS", now_ns)

    def net_failed(self, now_ns):
        """A try to host or join came to nothing: back to the list (which says why)."""
        return self._go("ONLINE", now_ns, Action("sound", "menu_back")) if self.screen == "WAIT" else []

    def opponent_left(self, gone=True):
        self.opponent_gone = gone

    def start_rematch(self, now_ns):
        """Both players have asked for another game."""
        if self.screen != "RESULTS" or not self.online_game or not self.rematch_pending:
            return []
        self.rematch_pending = False
        return self._launch(now_ns)

    def _rematch(self):
        if self.opponent_gone or self.rematch_pending:
            return []
        self.rematch_pending = True
        return [Action("online", ("rematch", None))] + self._selected()

    def _leave(self, now_ns):
        self.online_game, self.opponent, self.rematch_pending, self.opponent_gone = False, "", False, False
        return self._go("ONLINE", now_ns, Action("online", ("leave", None)), Action("sound", "menu_back"))

    def _host(self, pace, now_ns):
        self.level_tag = pace
        return self._go("WAIT", now_ns, Action("online", ("host", pace)), *self._selected())

    def _join(self, k, now_ns):
        rooms = self.online.rooms
        if k >= len(rooms):
            return []
        return self._go("WAIT", now_ns, Action("online", ("join", rooms[k]["code"])), *self._selected())

    # --- the clock ---------------------------------------------------------------------------------------------------------------------
    def update(self, now_ns, ab, phase):
        """Called every frame: ab is the hand's (a, b) in the reach box (None when it is not seen), phase the game's."""
        self._last_ns = now_ns
        actions = []
        if self._t0 is None:
            self._t0 = now_ns
            actions += self._music_for(self.screen) + self._boot
            self._boot = []
        if self.screen == "GAME":
            if phase == "MATCH_OVER" and now_ns - self._t0 > GAME_GRACE_S * S:
                actions += self._go("RESULTS", now_ns)
            return actions
        if self.screen == "RESULTS" and phase != "MATCH_OVER":
            return actions + self._go("GAME", now_ns)                       # something else started the next game
        targets = menu_layout.targets(self.screen, len(self.online.rooms), not self.opponent_gone)
        under = dwell.hit_test(ab, targets)
        if under is not None and under == self.pointer.hovered and self._room_code(under) != self._hover_code:
            self.pointer.restart()                          # another game has taken the place of the one the hand is held on
        pressed = self.pointer.update(now_ns, ab, targets, True)
        self._hover_code = self._room_code(self.pointer.hovered)
        actions += self._tick_if_new_hover(now_ns)
        if pressed:
            actions += self._press(pressed, now_ns)
        elif self.screen == "INTRO" and now_ns - self._t0 >= self.intro_s * S:
            actions += self._go("TITLE", now_ns)
        elif self.screen == "VS" and now_ns - self._t0 >= self.vs_s * S:
            actions += self._launch(now_ns)
        return actions

    def _room_code(self, target):
        """The code of the game a row of the list stands for at this moment (None for anything else)."""
        if target is None or not target.startswith("join"):
            return None
        k = int(target[4:])
        return self.online.rooms[k]["code"] if k < len(self.online.rooms) else None

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
        if target in ("back", "cancel"):
            return self._back(now_ns)
        if target == "skip":
            return self._go("TITLE", now_ns, *self._selected())
        if target == "start":
            return self._go("MODE", now_ns, *self._selected())
        if target in ("rally", "match"):
            self.mode = "survival" if target == "rally" else "match"
            return self._go("OPPONENT", now_ns, *self._selected())
        if target == "online":
            return self._go("ONLINE", now_ns, Action("online", ("open", None)), *self._selected())
        if target.startswith("host"):
            return self._host(int(target[4:]), now_ns)
        if target.startswith("join"):
            return self._join(int(target[4:]), now_ns)
        if target.startswith("opp"):
            self.level_tag = int(target[3:])
            return self._go("VS", now_ns, *self._selected())
        if target == "again":
            return self._rematch() if self.online_game else self._launch(now_ns) + self._selected()
        return self._leave(now_ns) if self.online_game else self._go("OPPONENT", now_ns, *self._selected())               # "change"

    # --- keys and cards ---------------------------------------------------------------------------------------------------------------------
    def on_key(self, key, now_ns):
        """-> the actions for a key the flow takes, or None for a key it leaves to the game (and to quitting)."""
        self._last_ns = now_ns
        if self.screen == "GAME" or key in QUIT_KEYS:
            return None
        if self.screen == "INTRO":
            return self._go("TITLE", now_ns)
        if self.screen == "VS":
            if self.online_game:
                return []                                                   # a friend is waiting: the face-off is not ours to cut short
            if key == SPACE:
                return self._launch(now_ns)
            return self._go("OPPONENT", now_ns, Action("sound", "menu_back")) if key in BACK_KEYS else []
        if key == SPACE:
            return self._space(now_ns)
        if key == ENTER:
            return self._enter(now_ns)
        if key in BACK_KEYS:
            return self._back(now_ns)
        if key in LEFT_KEYS + RIGHT_KEYS:
            return self._move(1 if key in RIGHT_KEYS else -1)
        if key in LEVEL_KEYS:
            return self._choose_level(LEVEL_KEYS[key])
        if key == ord("m") and not self.online_game:
            self.mode = "match" if self.mode == "survival" else "survival"
            self.focus = self._default_focus() if self.screen == "MODE" else self.focus
            return [Action("sound", "menu_tick")]
        return None

    def _space(self, now_ns):
        """SPACE: start now with the choices so far, but not from the online screens, and on a friend's results it is the rematch."""
        if self.screen in ("ONLINE", "WAIT"):
            return []
        if self.screen == "RESULTS" and self.online_game:
            return self._rematch()
        return self._launch(now_ns)

    def _enter(self, now_ns):
        if self.screen == "TITLE":
            return self._go("MODE", now_ns, *self._selected())
        if self.screen == "MODE":
            if self.focus == 2:
                return self._go("ONLINE", now_ns, Action("online", ("open", None)), *self._selected())
            self.mode = "survival" if self.focus == 0 else "match"
            return self._go("OPPONENT", now_ns, *self._selected())
        if self.screen == "OPPONENT":
            self.level_tag = self.focus + 1
            return self._go("VS", now_ns, *self._selected())
        if self.screen == "ONLINE":
            return self._host(self.level_tag, now_ns) if self.focus == 0 else self._join(self.focus - 1, now_ns)
        if self.screen == "WAIT":
            return []
        if self.online_game:
            return self._rematch() if self.focus == 0 else self._leave(now_ns)
        return self._launch(now_ns) if self.focus == 0 else self._go("OPPONENT", now_ns, *self._selected())

    def _back(self, now_ns):
        if self.screen == "ONLINE":
            return self._go("MODE", now_ns, Action("online", ("close", None)), Action("sound", "menu_back"))
        if self.screen == "WAIT":
            return self._go("ONLINE", now_ns, Action("online", ("cancel", None)), Action("sound", "menu_back"))
        if self.screen == "RESULTS" and self.online_game:
            return self._leave(now_ns)
        goes = {"MODE": "TITLE", "OPPONENT": "MODE", "RESULTS": "OPPONENT"}.get(self.screen)
        return [] if goes is None else self._go(goes, now_ns, Action("sound", "menu_back"))

    def _move(self, step):
        count = 1 + min(len(self.online.rooms), menu_layout.MAX_ROOMS) if self.screen == "ONLINE" else CHOICES.get(self.screen)
        if count is None:
            return []
        self.focus = max(0, min(count - 1, self.focus + step))
        return [Action("sound", "menu_tick")]

    def _choose_level(self, tag):
        if self.online_game or self.screen == "WAIT":
            return []                                                       # the pace of a game with a friend was settled by the host
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
            if self.screen == "RESULTS" and self.online_game:
                return self._rematch()
            return [] if self.screen in ("VS", "ONLINE", "WAIT") else self._launch(now_ns)
        if role == "LEVEL" and value in (1, 2, 3):
            return self._choose_level(value)
        return []

    # --- what the screen is told ------------------------------------------------------------------------------------------------------
    def ui_state(self, now_ns, ab):
        return UiState(screen=self.screen, t_s=0.0 if self._t0 is None else max(0.0, (now_ns - self._t0) / S), cursor=ab,
                       hover=self.pointer.hovered, progress=self.pointer.progress, focus=self.focus, mode=self.mode,
                       level_tag=self.level_tag, online=self.online, opponent=self.opponent, rematch_pending=self.rematch_pending,
                       opponent_gone=self.opponent_gone)
