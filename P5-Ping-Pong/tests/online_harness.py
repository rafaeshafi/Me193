"""Two laptops in one test: each a Session with its flow (the screens), an Online (finding a friend) and a scripted hand, on one fake
clock and one in-memory network.  Their hands point through the menus like the hub does and then play the match like a scripted player."""

import random

from pingpong import app, online
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose
from pingpong.flow import Flow
from pingpong import menu_layout
from pingpong.loopnet import LoopNet
from tests.versus_harness import Player

S = 1_000_000_000
ELSEWHERE = (0.5, 0.9)


def center(target, screen, rooms=0, can_again=True):
    """Where the hand points to be in the middle of a target of a screen (a, b with b up)."""
    rect = next(r for name, r, _ in menu_layout.targets(screen, rooms, can_again) if name == target)
    return (rect[0] + rect[2]) / 2, 1.0 - (rect[1] + rect[3]) / 2


class Laptop:
    def __init__(self, clock, net, name, *, skill="perfect", seed=1, target=3, level=1, start=None, vs_s=None, countdown_s=None, **session_kw):
        self.name = name
        self.online = online.Online(net, name=name, seed=seed)
        flow = Flow(intro=False, level_tag=level, start=start, **({} if vs_s is None else {"vs_s": vs_s}))
        self.session = app.make_session(level=level, mode="survival", target=target, clock=clock, hit_mode="swing", seed=seed,
                                        flow=flow, online=self.online, **session_kw)
        if countdown_s is not None:
            self.session.game.countdown_s = countdown_s
        self.session.player = name
        self.player = Player(self.session, skill, random.Random(seed))
        self.hand = ELSEWHERE                            # where the hand points while the screens are up (None: not seen)
        self.events = []
        self._pose_at = -1

    @property
    def game(self):
        return self.session.game

    @property
    def flow(self):
        return self.session.flow

    @property
    def screen(self):
        return self.session.hud_state().screen

    def step(self, now_ns):
        s = self.session
        if self.flow.screen == "GAME" or self.flow.screen == "RESULTS":
            self.events += self.player.act()
        if self.hand is not None and now_ns - self._pose_at >= S // 30 and self.flow.screen != "GAME":
            self._pose_at = now_ns
            u, v = self.game.judge.box.to_uv(*self.hand)
            self.events += s.on_pose(PaddlePose(t_scene_ns=now_ns, u=u, v=v, conf=0.9, hand="right"))
        self.events += s.tick()


class Pair:
    def __init__(self, *, host_skill="perfect", guest_skill="perfect", target=3, guest_target=None, latency_s=0.03, host_level=2, **net_kw):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.net = LoopNet(self.clock, latency_s=latency_s, **net_kw)
        self.a = Laptop(self.clock, self.net, "MAYA", skill=host_skill, seed=1, target=target, level=host_level)
        self.b = Laptop(self.clock, self.net, "RAFAE", skill=guest_skill, seed=2, target=guest_target or target, level=1)

    @property
    def laptops(self):
        return self.a, self.b

    def run(self, seconds, until=None, dt=0.01):
        end = self.clock.now_ns() + round(seconds * S)
        while self.clock.now_ns() < end:
            self.clock.advance_s(dt)
            for laptop in self.laptops:
                laptop.step(self.clock.now_ns())
            if until is not None and until(self):
                return True
        return until is None or until(self)

    def point(self, laptop, target, seconds, screen=None, rooms=0, can_again=True):
        """The hand of one laptop points at a target for a while (the other keeps its hand where it was)."""
        laptop.hand = target if isinstance(target, tuple) else center(target, screen or laptop.flow.screen, rooms, can_again)
        self.run(seconds)
        laptop.hand = ELSEWHERE

    def settle(self, seconds=0.4):
        for laptop in self.laptops:
            laptop.hand = ELSEWHERE
        self.run(seconds)

    def both_to(self, screen_name, **kw):
        """Both laptops from the title to the screen: START, then the third card (playing a friend)."""
        for laptop in self.laptops:
            laptop.flow.screen = "TITLE"
        self.settle(0.5)
        for laptop in self.laptops:
            laptop.hand = (0.95, 0.9)
        self.run(1.8)
        self.settle(0.5)
        for laptop in self.laptops:
            laptop.hand = center("online", "MODE")
        self.run(1.6)
        self.settle(0.5)
        return self

    def start_hosting(self, pace=2):
        self.point(self.a, f"host{pace}", 1.6)
        self.settle(0.5)

    def join_first(self):
        self.point(self.b, "join0", 1.6, rooms=len(self.b.session.hud_state().ui.online.rooms))
