"""Finding a friend to play: the list of open games, hosting one, joining one, and the handshake that makes two laptops a pair.

    host(pace, target)   open a game: its room is made first, and only when the room is listening is the game put on the list
    join(code)           knock on a game's room (join, said again every second) until it answers welcome, busy or goodbye
    step(now_ns)         read what has come; -> [Paired(...)] when a pair is made, [Failed(text)] when a try comes to nothing
    cancel / leave / close   stop hosting or joining / say goodbye after a game / put everything away

A guest gives itself a token (gid) and the answer names it, so when two guests knock together exactly one gets the game and the
other is told it is full.  The host chooses the session token (sid) that every message afterwards carries.  Once paired, the room's
link belongs to the versus.Remote of the game.  `net` is anything with lobby() and room(code, role, ident=): netlink.Network over
MQTT, or a test's in-memory network.
"""

import random
from dataclasses import dataclass

from pingpong import netproto as proto
from pingpong import versus
from pingpong.uistate import OnlineView

S = 1_000_000_000
CONNECT_TIMEOUT_S = 8.0          # the server is not there if nothing has connected by now
JOIN_TIMEOUT_S = 10.0            # a game that has not answered by now is not coming
JOIN_EVERY_S = 1.0               # a join is said again this often until it is answered
TOKEN_CHARS = "abcdefghjkmnpqrstuvwxyz23456789"
SERVER_DOWN = "COULD NOT REACH THE GAME SERVER"
SERVER_TIP = "CAN'T REACH THE GAME SERVER - IS THE INTERNET ON?"


@dataclass(frozen=True)
class Paired:
    role: str                    # "host" | "guest"
    opponent: str
    pace: int
    target: int
    code: str
    remote: versus.Remote


@dataclass(frozen=True)
class Failed:
    text: str


class Online:
    def __init__(self, net, *, name, seed=None):
        self.net, self.name, self.rng = net, proto.clean_name(name), random.Random(seed)
        self.state, self.code, self.pace, self.target, self.message = "idle", "", 1, 7, ""
        self._lobby = self._room = None
        self._sid = self._gid = ""
        self._advertised, self._joined = False, None
        self._since_ns = self._down_since_ns = self._ready_ns = self._last_join_ns = None
        self._offline = False

    # --- the list -----------------------------------------------------------------------------------------------------------------------
    def open(self):
        """Start keeping the list of open games."""
        if self._lobby is None:
            self._lobby = self.net.lobby()
            self._lobby.watch()
        if self.state == "idle":
            self.state = "browsing"

    def rooms(self):
        return [] if self._lobby is None else self._lobby.rooms()

    def _token(self):
        return "".join(self.rng.choice(TOKEN_CHARS) for _ in range(8))

    # --- hosting and joining ---------------------------------------------------------------------------------------------------------
    def host(self, pace, target):
        """Open a game for a friend; -> its code."""
        if not (1 <= pace <= 3 and 1 <= target <= proto.TARGET_MAX):
            raise ValueError(f"cannot host pace {pace} to {target}")
        self.open()
        self._stop_trying()
        taken = {r["code"] for r in self.rooms()}
        code = proto.new_code(self.rng)
        while code in taken:
            code = proto.new_code(self.rng)
        self.code, self.pace, self.target, self.message = code, pace, target, ""
        self._sid, self._advertised, self._joined, self._since_ns = self._token(), False, None, None
        self._room = self.net.room(code, "host", ident={"sid": self._sid})
        self.state = "hosting"
        return code

    def join(self, code):
        """Ask to join the game with this code."""
        clean = proto.clean_code(code)
        if clean is None:
            raise ValueError(f"{code!r} is not a game code")
        self.open()
        self._stop_trying()
        self.code, self.message = clean, ""
        self._gid, self._ready_ns, self._last_join_ns, self._since_ns = self._token(), None, None, None
        self._room = self.net.room(clean, "guest", ident={"gid": self._gid})
        self.state = "joining"

    def _stop_trying(self):
        """Put away a game being hosted or joined (or played)."""
        if self._room is not None:
            self._room.close()
            self._room = None
        if self._lobby is not None:
            self._lobby.withdraw()
        self.code, self._advertised = "", False
        self.state = "browsing" if self._lobby is not None else "idle"

    def cancel(self):
        """Give up hosting or joining; the list carries on."""
        self._stop_trying()

    leave = cancel                                    # after a game: goodbye to the other, and back to the list

    def close(self):
        """Put everything away."""
        self._stop_trying()
        if self._lobby is not None:
            self._lobby.close()
            self._lobby = None
        self.state, self._offline, self._down_since_ns = "idle", False, None

    # --- the clock ----------------------------------------------------------------------------------------------------------------------
    def step(self, now_ns):
        """Read what has come: -> [Paired] or [Failed] when something has been settled, else []."""
        if self._since_ns is None:
            self._since_ns = now_ns
        self._watch_server(now_ns)
        if self.state == "hosting":
            return self._step_hosting(now_ns)
        if self.state == "joining":
            return self._step_joining(now_ns)
        return []

    def _watch_server(self, now_ns):
        if self._lobby is None or self._lobby.connected:
            self._down_since_ns, self._offline = None, False
            return
        if self._down_since_ns is None:
            self._down_since_ns = now_ns
        self._offline = now_ns - self._down_since_ns >= CONNECT_TIMEOUT_S * S

    def _fail(self, text):
        self._stop_trying()
        self.message = text
        return [Failed(text)]

    def _step_hosting(self, now_ns):
        room = self._room
        if not self._advertised:
            if room.ready:                                           # only a room that is listening can hear a join
                self._lobby.host(self.code, self.name, self.pace, self.target)
                self._advertised = True
            elif now_ns - self._since_ns >= CONNECT_TIMEOUT_S * S:
                return self._fail(SERVER_DOWN)
            return []
        events = []
        for m in room.poll():
            if m["t"] != "join":
                continue
            if self._joined is None:
                self._joined = m["gid"]
                events.append(self._host_pairs(m))
            elif m["gid"] != self._joined:
                room.send({"t": "busy", "to": m["gid"]})
        return events

    def _host_pairs(self, join):
        self._room.send({"t": "welcome", "name": self.name, "pace": self.pace, "target": self.target, "sid": self._sid, "to": join["gid"]})
        self._lobby.withdraw()                                       # it is not a game to join any more
        remote = versus.Remote(self._room, host=True, opponent_name=join["name"], sid=self._sid, gid=join["gid"], seed=self.rng.randrange(1 << 30))
        self.state = "paired"
        return Paired("host", join["name"], self.pace, self.target, self.code, remote)

    def _step_joining(self, now_ns):
        room = self._room
        if not room.ready:
            return self._fail(SERVER_DOWN) if now_ns - self._since_ns >= CONNECT_TIMEOUT_S * S else []
        if self._ready_ns is None:
            self._ready_ns = now_ns
        for m in room.poll():
            if m["t"] == "welcome" and m["to"] == self._gid:
                return [self._guest_pairs(m)]
            if m["t"] == "busy" and m["to"] == self._gid:
                return self._fail("THAT GAME IS FULL")
            if m["t"] == "bye":
                return self._fail("THAT GAME WAS CLOSED")
        if now_ns - self._ready_ns >= JOIN_TIMEOUT_S * S:
            return self._fail("COULD NOT REACH THAT GAME")
        if self._last_join_ns is None or now_ns - self._last_join_ns >= JOIN_EVERY_S * S:
            self._last_join_ns = now_ns
            room.send({"t": "join", "name": self.name, "gid": self._gid})
        return []

    def _guest_pairs(self, welcome):
        remote = versus.Remote(self._room, host=False, opponent_name=welcome["name"], sid=welcome["sid"], gid=self._gid,
                               seed=self.rng.randrange(1 << 30))
        self.state, self.pace, self.target = "paired", welcome["pace"], welcome["target"]
        return Paired("guest", welcome["name"], welcome["pace"], welcome["target"], self.code, remote)

    # --- what the screens are told ---------------------------------------------------------------------------------------------------------
    def view(self):
        status, message = self.state, self.message
        if self.state == "browsing" and self._lobby is not None and not self._lobby.connected:
            status, message = ("offline", SERVER_TIP) if self._offline else ("connecting", message)
        return OnlineView(status=status, rooms=tuple(self.rooms()), code=self.code, pace=self.pace, target=self.target, message=message,
                          live=self._advertised)
