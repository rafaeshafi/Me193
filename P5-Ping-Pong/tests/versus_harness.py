"""Two complete games on one fake clock, joined by a link with real latency: the way to test online play without a network.

Each side is an app.Session (a GameCore, its view, its judge) with a versus.Remote on its end of the cable and a scripted player.
A "perfect" player meets every ball; "idle" never moves, so every ball it gets is a miss.
"""

import random

from pingpong import app, contact, netlink, versus
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose, SwingEvent

S = 1_000_000_000


class Player:
    """A scripted hand and wrist (as app.play_until's, one frame at a time)."""

    def __init__(self, session, skill, rng=None):
        self.session, self.skill, self._swung, self._step = session, skill, None, 0
        self.rng, self._mind = rng or random.Random(0), {}            # "flaky" meets about two balls in three, at random

    def _meets(self, ball):
        if self.skill == "perfect":
            return True
        if self.skill == "idle":
            return False
        return self._mind.setdefault(ball.ball_id, self.rng.random() < 0.65)

    def act(self):
        """-> the events this frame's hand and wrist made."""
        s, game = self.session, self.session.game
        ball, events = game.incoming, []
        if ball is None or not self._meets(ball):
            return events
        self._step += 1
        now = s.clock.now_ns()
        if ball.t_c_ns - int(0.5 * S) <= now <= ball.t_c_ns + int(0.05 * S) and self._step % 3 == 0:
            u, v = game.judge.box.to_uv(ball.aim_ab[0], 0.5)
            events += s.on_pose(PaddlePose(t_scene_ns=now, u=u, v=v, conf=0.9, hand="right"))
        if now >= ball.t_c_ns and self._swung != ball.ball_id:
            self._swung = ball.ball_id
            peak = ball.t_c_ns - round(game.judge.contact_lag_s * S)
            events += s.on_swing(SwingEvent(kind="IMPACT", t_ns=peak, w_pk=600.0, dur_ms=150.0, n_reversals=0, axis_unit=(1, 0, 0),
                                  net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1), clipped=False, feat=(0.0,) * 12))
        return events


class ContactPlayer:
    """A hand the camera always sees: level with each ball it means to meet (a still hand blocks, the softest return) and well away
    from the ones it does not ("idle" never meets one, "flaky" about two in three)."""

    def __init__(self, session, skill, rng=None):
        self.session, self.skill, self._step, self.rng, self._mind = session, skill, 0, rng or random.Random(0), {}

    def _meets(self, ball):
        if self.skill == "perfect":
            return True
        return self.skill == "flaky" and self._mind.setdefault(ball.ball_id, self.rng.random() < 0.65)

    def act(self):
        s, game = self.session, self.session.game
        self._step += 1
        if self._step % 3:
            return []                                                       # the camera: one reading in three frames
        now = s.clock.now_ns()
        ball, leg = game.incoming, game.incoming_leg
        u = 0.0
        if ball is not None and leg is not None and ball.t_c_ns - int(0.7 * S) <= now <= ball.t_c_ns + int(0.4 * S):
            u = contact.ball_u(game.judge.box, leg.position(leg.arrival_ns)[0]) + (0.0 if self._meets(ball) else 1.4)
        return s.on_pose(PaddlePose(t_scene_ns=now, u=u, v=0.0, conf=0.9, hand="right"))


class Side:
    def __init__(self, clock, link, *, host, skill, level, target, seed, name, sid="abc123", gid="k3x9", hit_mode="swing"):
        self.session = app.make_session(level=level, mode="match", target=target, clock=clock, hit_mode=hit_mode, seed=seed)
        self.link, self.events = link, []
        self.player = (ContactPlayer if hit_mode == "contact" else Player)(self.session, skill, random.Random(seed * 31 + 7))
        self.remote = versus.Remote(link, host=host, opponent_name=name, sid=sid, gid=gid, seed=seed)
        self.session.game.remote = self.remote

    @property
    def game(self):
        return self.session.game

    def tick(self):
        self.events += self.player.act()
        self.events += self.session.tick()


class Rig:
    def __init__(self, *, latency_s=0.03, target=3, level=1, host_skill="perfect", guest_skill="perfect", seed=1, hit_mode="swing", **link_kw):
        self.clock = FakeClock(start_ns=1_000_000_000)
        link_kw.setdefault("rng", random.Random(seed))
        a, b = netlink.PairLink.pair(self.clock, latency_s=latency_s, **link_kw)
        self.link_a, self.link_b = a, b
        self.host = Side(self.clock, a, host=True, skill=host_skill, level=level, target=target, seed=seed, name="MAYA", hit_mode=hit_mode)
        self.guest = Side(self.clock, b, host=False, skill=guest_skill, level=level, target=target, seed=seed + 1, name="RAFAE", hit_mode=hit_mode)

    def start(self):
        self.host.session.on_start()
        self.guest.session.on_start()
        return self

    def run(self, seconds, until=None, dt=0.01):
        end = self.clock.now_ns() + round(seconds * S)
        while self.clock.now_ns() < end:
            self.clock.advance_s(dt)
            self.host.tick()
            self.guest.tick()
            if until is not None and until(self):
                return True
        return until is None or until(self)

    @property
    def sides(self):
        return self.host, self.guest

    def scores(self):
        """((host's points, guest's points) as the host sees them, as the guest sees them)."""
        h, g = self.host.game, self.guest.game
        return (h.player_points, h.cpu_points), (g.cpu_points, g.player_points)
