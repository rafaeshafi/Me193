"""A whole network in memory: the lobby of open games and the rooms, with the same two calls as netlink.Network, so that online play can
be driven end to end in a test (or two sessions in one process) without a broker.

Like the broker, a room has two topics (what the host says, what the guests say) and every link in a room hears the topic the other
role speaks on, so two guests at one room both hear the host.  Messages go through the real codec and arrive after a latency (with
jitter that never reorders them); unreliable ones can be lost and reliable ones can come twice (netlink.PairLink's cable)."""

import random

from pingpong import netlink
from pingpong import netproto as proto

S = 1_000_000_000


class LoopLink(netlink.PairLink):
    """One laptop's end of a room."""

    def __init__(self, net, code, role, ident):
        super().__init__(net.clock, net.link_kw.get("latency_s", 0.0), net.link_kw.get("jitter_s", 0.0), net.link_kw.get("loss0", 0.0),
                         net.link_kw.get("dup1", 0.0), net.rng, {"cut": False}, net.link_kw.get("loss1", 0.0))
        self.net, self.code, self.role, self.ident = net, code, role, dict(ident or {})
        net.rooms.setdefault(code, {"host": [], "guest": []})[role].append(self)

    @property
    def ready(self):
        return self.alive and self.net.up

    def _hearers(self):
        return [link for link in self.net.rooms[self.code][proto.other_role(self.role)] if link.alive]

    def send(self, message, qos=1):
        if not self.alive:
            return
        wire = proto.encode(message)
        for peer in self._hearers():
            if qos == 0 and self.rng.random() < self.loss0:
                continue
            if qos == 1 and self.loss1 and self.rng.random() < self.loss1:
                continue
            for _ in range(2 if qos == 1 and self.rng.random() < self.dup1 else 1):
                peer._deliver(wire, self.clock.now_ns() + round((self.latency_s + self.rng.uniform(0.0, self.jitter_s)) * S))

    def close(self):
        if self.alive:
            self.send({"t": "bye", "reason": "left", **self.ident})
        super().close()

    def die(self):
        """The laptop is gone without a word: the broker says so for it, with the will it was given."""
        if self.alive:
            self.send({"t": "bye", "reason": "lost", **self.ident})
        super().close()


class LoopLobby:
    def __init__(self, net):
        self.net, self._own, self._code = net, None, None

    @property
    def connected(self):
        return self.net.up

    def watch(self, own_code=None):
        self._own = own_code

    def rooms(self):
        return sorted((dict(e) for code, e in self.net.entries.items() if code != self._own), key=lambda e: -e["t"])

    def host(self, code, name, pace, target):
        self._own = self._code = code
        self.net.entries[code] = {"code": code, "host": proto.clean_name(name), "pace": pace, "target": target, "t": int(self.net.clock.now_ns() / S)}

    def withdraw(self):
        if self._code is not None:
            self.net.entries.pop(self._code, None)

    def close(self):
        self.withdraw()


class LoopNet:
    def __init__(self, clock, *, rng=None, **link_kw):
        self.clock, self.link_kw, self.rng = clock, link_kw, rng or random.Random(0)
        self.entries, self.rooms, self.up = {}, {}, True                 # open games by code, the links of each room, "the server is there"

    def lobby(self):
        return LoopLobby(self)

    def room(self, code, role, ident=None):
        return LoopLink(self, code, role, ident)
