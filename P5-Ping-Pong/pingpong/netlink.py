"""The channel between two laptops: the real thing over MQTT, and an in-memory pair for tests.

A Link is anything with send(message, qos), poll() -> the messages that have arrived (oldest first), alive, ready and close().
RoomLink is the private channel of one game (what the host says goes on one topic, what the guest says on another; a will
tells the other side at once if this laptop dies, and a goodbye says it left on purpose).  Lobby is the list of open games:
a host's game is one retained message that its own last will clears if it dies, and a watcher keeps the list up to date from
the retained messages.  Everything stays in the game's own corner of the broker: check_topic() refuses anything else, so
this code cannot touch the official score topic whatever it is handed.
"""

import heapq
import itertools
import queue
import random
import threading
import time
import uuid

import config
from pingpong import netproto as proto

LOBBY_MAX_AGE_S = 3 * 3600          # a game that has been open longer than this is somebody's forgotten one
S = 1_000_000_000


def check_topic(topic):
    """The topic itself if it is in the game's own corner of the broker; anything else (the score topic, a wildcard that could
    reach it, a path that climbs out) is a ValueError."""
    if not isinstance(topic, str) or not topic.startswith(proto.NAMESPACE + "/") or ".." in topic or "\x00" in topic:
        raise ValueError(f"{topic!r} is not a topic of this game")
    return topic


def make_client(prefix="pp-net"):
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{prefix}-{uuid.uuid4().hex[:8]}")
    client.reconnect_delay_set(1, 5)                  # the public broker drops an attempt now and then: try again briskly
    return client


# --- an in-memory pair, for tests ---------------------------------------------------------------------------------------------------------
class PairLink:
    """One end of a cable between two games in the same process: the messages go through the real codec, arrive after a latency
    (with jitter that never reorders them), unreliable ones can be lost, reliable ones can come twice, and the cable can be cut."""

    def __init__(self, clock, latency_s, jitter_s, loss0, dup1, rng, shared):
        self.clock, self.latency_s, self.jitter_s, self.loss0, self.dup1, self.rng = clock, latency_s, jitter_s, loss0, dup1, rng
        self._shared, self._inbox, self._last_due, self._seq, self._closed, self._peer = shared, [], 0, itertools.count(), False, None

    @classmethod
    def pair(cls, clock, latency_s=0.0, jitter_s=0.0, loss0=0.0, dup1=0.0, rng=None):
        rng, shared = rng or random.Random(0), {"cut": False}
        a, b = cls(clock, latency_s, jitter_s, loss0, dup1, rng, shared), cls(clock, latency_s, jitter_s, loss0, dup1, rng, shared)
        a._peer, b._peer = b, a
        return a, b

    @property
    def alive(self):
        return not self._shared["cut"] and not self._closed

    ready = alive

    def cut(self):
        self._shared["cut"] = True

    def mend(self):
        self._shared["cut"] = False

    def send(self, message, qos=1):
        if not self.alive:
            return
        wire = proto.encode(message)
        if qos == 0 and self.rng.random() < self.loss0:
            return
        for _ in range(2 if qos == 1 and self.rng.random() < self.dup1 else 1):
            self._peer._deliver(wire, self.clock.now_ns() + round((self.latency_s + self.rng.uniform(0.0, self.jitter_s)) * S))

    def _deliver(self, wire, due_ns):
        due_ns = max(due_ns, self._last_due)                                         # a cable never reorders
        self._last_due = due_ns
        heapq.heappush(self._inbox, (due_ns, next(self._seq), wire))

    def inject_raw(self, raw):
        """What a stranger could put on the wire: bytes that arrive as they are."""
        self._deliver(raw, self.clock.now_ns())

    def poll(self):
        out = []
        while self._inbox and self._inbox[0][0] <= self.clock.now_ns() and not self._closed:
            message = proto.decode(heapq.heappop(self._inbox)[2])
            if message is not None:
                out.append(message)
        return out

    def close(self):
        self._closed = True


# --- the room of one game over MQTT --------------------------------------------------------------------------------------------------------
class RoomLink:
    def __init__(self, code, role, *, client=None, ident=None):
        """ident: the tokens that say who this end is (a guest's gid, a host's sid): its goodbye and its will carry them, so the
        other end can tell its own partner leaving from a guest that was turned away."""
        if proto.clean_code(code) != code or role not in proto.ROLES:
            raise ValueError(f"cannot make a room link for {code!r} as {role!r}")
        self.code, self.role, self.ident = code, role, dict(ident or {})
        self.topic_out = check_topic(proto.room_topic(code, role))
        self.topic_in = check_topic(proto.room_topic(code, proto.other_role(role)))
        self._inbox, self._ready = queue.SimpleQueue(), threading.Event()
        self.client = client = client or make_client()
        client.will_set(self.topic_out, self._bye("lost"), qos=1)                    # if this laptop dies, the other knows at once
        client.on_connect, client.on_subscribe, client.on_message = self._on_connect, self._on_subscribe, self._on_message
        client.connect_async(*config.net_broker(), config.KEEPALIVE_S)
        client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self._ready.clear()
        client.subscribe(self.topic_in, qos=1)                                       # (a reconnect forgets subscriptions: do it every time)

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties=None):
        self._ready.set()

    def _on_message(self, client, userdata, message):
        decoded = proto.decode(message.payload)
        if decoded is not None:
            self._inbox.put(decoded)

    def _bye(self, reason):
        return proto.encode({"t": "bye", "reason": reason, **self.ident})

    @property
    def ready(self):
        """Connected and listening: only now can what the other side says to us arrive."""
        return self._ready.is_set() and self.client.is_connected()

    @property
    def alive(self):
        return self.client.is_connected()

    def send(self, message, qos=1):
        self.client.publish(self.topic_out, proto.encode(message), qos=qos)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        try:
            if self.client.is_connected():
                self.client.publish(self.topic_out, self._bye("left"), qos=1).wait_for_publish(1.0)
            self.client.disconnect()                                                  # a clean goodbye: the will stays unsaid
        except Exception:
            pass
        finally:
            self.client.loop_stop()


# --- the list of open games ----------------------------------------------------------------------------------------------------------------------
class Lobby:
    def __init__(self, now=time.time):
        self._now, self._rooms, self._lock = now, {}, threading.Lock()
        self._watch_client = self._host_client = self._own = self._host_topic = None

    def watch(self, own_code=None):
        """Start keeping the list of open games (not counting your own, if you are hosting one)."""
        self._own = own_code
        client = self._watch_client = make_client("pp-lobby")
        client.on_connect = lambda c, userdata, flags, reason_code, properties=None: c.subscribe(check_topic(proto.lobby_filter()), qos=1)
        client.on_message = self._on_message
        client.connect_async(*config.net_broker(), config.KEEPALIVE_S)
        client.loop_start()

    def _on_message(self, client, userdata, message):
        code = proto.code_from_topic(message.topic)
        if code is None:
            return
        entry = proto.read_lobby(message.payload) if message.payload else None
        with self._lock:
            if not message.payload:
                self._rooms.pop(code, None)                                           # the host closed it (or its will did)
            elif entry is not None and entry["code"] == code:                         # a message that names another room than its topic lies
                self._rooms[code] = entry

    def rooms(self):
        """The open games, newest first, as dicts of code, host, pace, target and t."""
        cutoff = self._now() - LOBBY_MAX_AGE_S
        with self._lock:
            return sorted((dict(e) for e in self._rooms.values() if e["t"] >= cutoff and e["code"] != self._own), key=lambda e: -e["t"])

    def host(self, code, name, pace, target):
        """Open a game for others to find; it closes when withdraw() is called, or at once if this laptop dies (a game already
        open is withdrawn first: a laptop hosts one game at a time)."""
        self.withdraw()
        self._own = code
        topic = self._host_topic = check_topic(proto.lobby_topic(code))
        entry = proto.lobby_entry(code, name, pace, target, now=self._now())
        client = self._host_client = make_client("pp-host")
        client.will_set(topic, b"", qos=1, retain=True)
        client.on_connect = lambda c, userdata, flags, reason_code, properties=None: c.publish(topic, entry, qos=1, retain=True)
        client.connect_async(*config.net_broker(), config.KEEPALIVE_S)
        client.loop_start()

    def withdraw(self):
        """Close the game you opened: it leaves everyone's list, and its connection goes."""
        client, topic, self._host_client, self._host_topic = self._host_client, self._host_topic, None, None
        if client is None:
            return
        try:
            if client.is_connected():
                client.publish(topic, b"", qos=1, retain=True).wait_for_publish(1.0)
            client.disconnect()
        except Exception:
            pass
        finally:
            client.loop_stop()

    @property
    def connected(self):
        return any(c is not None and c.is_connected() for c in (self._watch_client, self._host_client))

    def close(self):
        self.withdraw()
        client, self._watch_client = self._watch_client, None
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
            client.loop_stop()


class Network:
    """What online.Online plays over: the lobby of open games and the rooms, over the MQTT broker."""

    def lobby(self):
        return Lobby()

    def room(self, code, role, ident=None):
        return RoomLink(code, role, ident=ident)
