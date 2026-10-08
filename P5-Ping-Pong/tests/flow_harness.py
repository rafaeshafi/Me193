"""What the flow tests share: a driver that feeds a flow at 60 Hz and keeps every action it asks for, with where the hand points to press
each button, and ways to get to a screen."""

from pingpong import menu_layout
from pingpong.flow import Flow

S = 1_000_000_000
STEP = S // 60
SPACE, ENTER, BACK, ESC, Q = 32, 13, 8, 27, ord("q")
D, A = ord("."), ord(",")
ELSEWHERE = (0.5, 0.9)                           # (a, b) with b up: y = 0.1, over no button
START_AT = (0.95, 0.9)                           # the top right: the START button, and "play again"
LEFT_TOP = (0.05, 0.9)                           # the top left: "change opponent"
RALLY_AT, MATCH_AT, ONLINE_AT = (0.22, 0.45), (0.5, 0.45), (0.78, 0.45)
OPP_AT = {1: (0.22, 0.4), 2: (0.5, 0.4), 3: (0.78, 0.4)}
BACK_AT = (0.08, 0.07)
ROOM = {"code": "ABCDE", "host": "MAYA", "pace": 2, "target": 7, "t": 1}
ROOMS = (ROOM, dict(ROOM, code="FGHJK", host="LEO"), dict(ROOM, code="MNPQR", host="ZOE"), dict(ROOM, code="STUVW", host="ABE"))


def at_target(target, screen, rooms=0):
    """Where the hand points to be in the middle of a target of a screen (a, b with b up)."""
    rect = next(r for name, r, _ in menu_layout.targets(screen, rooms) if name == target)
    return (rect[0] + rect[2]) / 2, 1.0 - (rect[1] + rect[3]) / 2


class Drive:
    """Feeds a flow at 60 Hz and keeps every action it asks for, with the second it came."""

    def __init__(self, flow=None, **kw):
        self.flow, self.t, self.log = flow or Flow(**kw), 0, []

    def run(self, ab, seconds, phase="LOBBY"):
        for _ in range(round(seconds * 60)):
            self.t += STEP
            for action in self.flow.update(self.t, ab, phase):
                self.log.append((self.t / S, action))
        return self

    def key(self, key):
        actions = self.flow.on_key(key, self.t)
        for action in actions or ():
            self.log.append((self.t / S, action))
        return actions

    def tag(self, role, value=0):
        actions = self.flow.on_tag(role, value, self.t)
        for action in actions or ():
            self.log.append((self.t / S, action))
        return actions

    def started(self):
        return [a for _, a in self.log if a.kind == "start"]

    def sounds(self):
        return [a.value for _, a in self.log if a.kind == "sound"]

    def haptics(self):
        return [a.value for _, a in self.log if a.kind == "haptic"]

    def music(self):
        return [a.value for _, a in self.log if a.kind == "music"]

    def on(self, screen, ab=ELSEWHERE, seconds=0.3):
        """Settle on a screen, the hand away from every button."""
        self.run(ab, seconds)
        assert self.flow.screen == screen, (self.flow.screen, screen)
        return self


def at_title(**kw):
    return Drive(intro=False, **kw).on("TITLE")


def at_mode(**kw):
    d = at_title(**kw)
    d.run(START_AT, 1.8)
    return d.on("MODE")


def at_opponents(**kw):
    d = at_mode(**kw)
    d.run(MATCH_AT, 1.5)
    return d.on("OPPONENT")
