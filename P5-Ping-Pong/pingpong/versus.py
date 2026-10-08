"""Playing another person: the other end of the cable, as the game core sees it.

Each laptop runs its own complete game.  When you hit the ball your game sends what you did (the ball leaves from here, at this
speed, with this spin, aimed there); the other game starts that same ball flying at its player, mirrored (the table is turned
about the net, physics.plan_mirrored), and decides for itself whether its player returns it.  If they do, their game sends the
ball back the same way; if they do not, it says so, and both games give you the point.  No clock is shared, and none is
needed: a ball starts flying when its message is read, so the time each player has to react is the ball's own flight, however
long the message took (the delay only stretches the rally).

GameCore calls into a Remote at a handful of places (game.remote is None for every other game):
    tick      step(): read the messages, keep the pings going, and wait or give up if the other end goes quiet
    _serve    serve(): the server's serve (automatic: a slow clean ball); the receiver waits for it
    _launch   shape() the speed to the match's pace, plan() the flight to the other person, send_hit() what you did
    misses    sent_miss(): tell the other game you missed (or faulted): the point is theirs
"""

import dataclasses
import math
import random

from pingpong import physics
from pingpong.events import GameEvent
from pingpong.judge import BallWindow

S = 1_000_000_000
POS_EVERY_S = 1.0 / 15
PING_EVERY_S = 1.0
STALE_S = 4.0                    # no word for this long and the game waits for them
LOST_S = 20.0                    # ... and for this long and the match is over
PADDLE_SMOOTH_S = 0.08           # how fast the drawn opponent follows the paddle position it was told
READY_BEFORE_SERVE_S = 0.5       # a serve that arrives this close to the end of the countdown starts the ball at once
ACCEPTS = ("hit", "miss", "pos", "ping", "pong", "rematch")


def versus_speed(level, strength):
    """The speed of a ball in a game with another person: the match's pace, a little faster for a harder hit.  (Against the
    computer a hard hit can be as fast as 14 m/s, and the computer's paddle copes; a person cannot.)"""
    return level.v_tier * (0.8 + 0.5 * strength)


class Remote:
    def __init__(self, link, *, host, opponent_name, sid, seed=0):
        self.link, self.host, self.sid, self.opponent_name = link, host, sid, opponent_name
        self.rng = random.Random(seed)
        self.paddle_x = 0.0                      # where the other person's paddle is across the table, as seen from this end
        self.ping_ms = None
        self.gone, self.reason = False, ""
        self.rematch_seen = False
        self.awaiting = False                    # my ball is out and their answer has not come
        self._n, self._seen_n = 0, 0
        self._heard = self._last_ping = self._last_pos = self._last_step = None
        self._my_x, self._x_target, self._early, self._out = None, 0.0, None, None
        self._pings = 0

    # --- what the screen asks -------------------------------------------------------------------------------------------------------------------
    def set_paddle(self, x_m):
        """Where my own paddle is across the table (sent to the other game now and then)."""
        self._my_x = x_m

    def on_start(self, now_ns):
        self.awaiting, self._heard = False, now_ns                  # the wait before a game is not silence

    # --- sending ----------------------------------------------------------------------------------------------------------------------------------
    def _send(self, message, qos=1):
        self.link.send(dict(message, sid=self.sid), qos=qos)

    def _next(self):
        self._n += 1
        return self._n

    def _score(self, game):
        """(host's points, guest's points): the score the same way round for both ends."""
        return [game.player_points, game.cpu_points] if self.host else [game.cpu_points, game.player_points]

    def shape(self, game, sp, strength):
        return dataclasses.replace(sp, v_out=versus_speed(game.level, strength))

    def plan(self, game, contact_ns, sp, contact, stroke):
        aim = (0.5 + sp.aim_a / 1.6, 0.5)
        self._out = {"v": sp.v_out, "start": [float(c) for c in contact], "aim": list(aim), "top": sp.T, "side": sp.S, "loft": stroke.loft_m}
        return physics.plan_return(contact_ns, sp.v_out, contact, aim, topspin=sp.T, sidespin=sp.S, fault=sp.fault, loft_m=stroke.loft_m,
                                   end_z=physics.TABLE_LEN_M - physics.HIT_Z_M)

    def send_hit(self, game, now_ns, *, serve=False):
        self._send(dict(self._out, t="hit", n=self._next(), serve=serve, score=self._score(game), rally=game.tracker.streak))
        self.awaiting = True

    def sent_miss(self, game, reason, now_ns):
        self._send({"t": "miss", "n": self._next(), "reason": reason, "score": self._score(game)})
        self.awaiting = False

    def serve(self, game, t0_ns):
        """A point starts: whose serve it is depends on how many points have been played (the host serves first, then one each)."""
        game.phase, game.incoming, game.incoming_leg, game._hand, game._cpu_at = "RALLY", None, None, None, None
        if (game.player_points + game.cpu_points) % 2 == (0 if self.host else 1):
            aim = (self.rng.uniform(0.3, 0.7), 0.5)
            v = game.level.v_tier
            game.outgoing_leg = physics.plan_serve_out(t0_ns, v, 0.0, aim)
            self._out = {"v": v, "start": [0.0, physics.STRIKE_Y_M, physics.HIT_Z_M], "aim": list(aim), "top": 0.0, "side": 0.0, "loft": 0.0}
            self.send_hit(game, t0_ns, serve=True)
        else:
            game.outgoing_leg = None                                       # the other person serves: wait for the ball
        return []

    # --- the clock: read the messages ----------------------------------------------------------------------------------------------------------------
    def step(self, game, now_ns):
        events = []
        if self._heard is None:
            self._heard = now_ns
        dt = 0.0 if self._last_step is None else max(0.0, (now_ns - self._last_step) / S)
        self._last_step = now_ns
        for m in self.link.poll():
            if m["t"] == "bye":
                if m.get("sid", self.sid) == self.sid:
                    self._end(m["reason"] or "left")
                continue
            if m["t"] not in ACCEPTS or m.get("sid") != self.sid or self.gone:
                continue
            self._heard = now_ns
            events += self._read(game, m, now_ns)
        if self._early is not None and self._ready(game, now_ns):
            held, self._early = self._early, None
            events += self._receive_hit(game, held, now_ns)
        self._x_shown(dt)
        self._keep_alive(game, now_ns)
        return events

    def _read(self, game, m, now_ns):
        kind = m["t"]
        if kind == "pos":
            self._x_target = -m["x"]
        elif kind == "ping":
            self._send({"t": "pong", "k": m["k"], "ts": m["ts"]}, qos=0)
        elif kind == "pong":
            rtt = (now_ns / S - m["ts"]) * 1000.0
            self.ping_ms = rtt if self.ping_ms is None else 0.7 * self.ping_ms + 0.3 * rtt
        elif kind == "rematch":
            self.rematch_seen = True
        elif kind in ("hit", "miss"):
            if m["n"] <= self._seen_n:
                return []                                                        # a copy of one already read
            self._seen_n = m["n"]
            return self._receive_hit(game, m, now_ns) if kind == "hit" else self._receive_miss(game, m, now_ns)
        return []

    def _x_shown(self, dt):
        self.paddle_x += (self._x_target - self.paddle_x) * (1.0 - math.exp(-dt / PADDLE_SMOOTH_S))

    def _keep_alive(self, game, now_ns):
        if self._last_ping is None or now_ns - self._last_ping >= PING_EVERY_S * S:
            self._last_ping = now_ns
            self._pings += 1
            self._send({"t": "ping", "k": self._pings, "ts": now_ns / S}, qos=0)
        if self._my_x is not None and (self._last_pos is None or now_ns - self._last_pos >= POS_EVERY_S * S):
            self._last_pos = now_ns
            self._send({"t": "pos", "x": self._my_x}, qos=0)
        silent = (now_ns - self._heard) / S
        if self.gone:
            return
        if silent > STALE_S:
            if "opponent" not in game.pause_reasons:
                game.set_pause("opponent", True, now_ns)
            if silent > LOST_S:
                self._end("lost")
        elif "opponent" in game.pause_reasons:
            game.set_pause("opponent", False, now_ns)

    def _end(self, reason):
        self.gone, self.reason = True, reason

    # --- the other person's ball ------------------------------------------------------------------------------------------------------------------------
    def _ready(self, game, now_ns):
        if game.phase in ("RALLY", "POINT_OVER"):
            return True
        remaining = game.seconds_to_serve(now_ns)
        return remaining is not None and remaining <= READY_BEFORE_SERVE_S

    def _receive_hit(self, game, m, now_ns):
        """Their ball comes at my player, from where they struck it, starting now."""
        if game.phase in ("LOBBY", "MATCH_OVER") or not self._ready(game, now_ns):
            self._early = m if game.phase != "MATCH_OVER" else None
            return []
        if game.incoming is not None or game._pending is not None or game.held is not None:
            return []                                                            # a ball is already on its way to me
        leg = physics.plan_mirrored(now_ns, m["v"], m["start"], m["aim"], m["top"], m["side"], loft_m=m["loft"])
        game._ball_id += 1
        game.incoming, game.incoming_leg, game.outgoing_leg = BallWindow(game._ball_id, leg.arrival_ns, leg.aim_ab, game.level, leg), leg, None
        game.phase, game._cpu_at, game._hand, game.point_over_until_ns = "RALLY", None, None, None
        self.awaiting = False
        self._adopt_score(game, m.get("score"), now_ns)
        return [GameEvent("serve", now_ns, {"ball_id": game._ball_id, "v": leg.v, "aim_ab": leg.aim_ab, "arrival_ns": leg.arrival_ns,
                                            "special": False, "remote": True})]

    def _receive_miss(self, game, m, now_ns):
        """They missed (or faulted): the point is mine."""
        if game.phase != "RALLY" or game.incoming is not None or game._pending is not None:
            return []
        leg = game.outgoing_leg
        if leg is not None and now_ns > leg.arrival_ns:                          # the ball waited at their end: now it flies on past them
            game.outgoing_leg = dataclasses.replace(leg, t0_ns=leg.t0_ns + (now_ns - leg.arrival_ns))
        self.awaiting = False
        events = game._end_rally("cpu_miss", now_ns)
        return events + self._adopt_score(game, m["score"], now_ns)

    def _adopt_score(self, game, score, now_ns):
        """The host's count of the points is the one that stands: a guest that disagrees takes it."""
        if self.host or score is None:
            return []
        points = (score[1], score[0])                                              # (mine, theirs): I am the guest
        if (game.player_points, game.cpu_points) == points:
            return []
        game.player_points, game.cpu_points = points
        winner = game._winner()
        if winner and game.phase != "MATCH_OVER":
            game.phase = "MATCH_OVER"
            game._end_game()
            return [GameEvent("match_over", now_ns, {"winner": winner, "player_points": points[0], "cpu_points": points[1]})]
        return []
