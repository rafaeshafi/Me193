"""Two whole live pipelines playing each other: each a FakeRig (fake hub, camera and tags through the real wiring of live.assemble) with
its own flow, its own online manager and a scripted player, joined by an in-memory network on one shared clock.  The hub, the camera
and the broker are fake; everything between is what a player runs.  Used by the tests and by `./pp play --selftest`."""

from pingpong import online
from pingpong.clock import FakeClock
from pingpong.fakerig import FakeRig
from pingpong.flow import Flow
from pingpong.loopnet import LoopNet

S = 1_000_000_000


class Match:
    """A host's rig and a guest's, on one clock; the guest joins the host's game by its code."""

    def __init__(self, *, target=3, pace=2, host_idle=False, guest_idle=False, record_dirs=(None, None), **net_kw):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.net = LoopNet(self.clock, **net_kw)
        self.a = FakeRig(clock=self.clock, level=1, mode="survival", target=target, seed=1, cards=[],
                         flow=Flow(intro=False, level_tag=pace, start=("host", pace)), online=online.Online(self.net, name="MAYA", seed=1),
                         record_dir=record_dirs[0])
        self.a.run(until=lambda: self.a.session.online.code, seconds=0.5, max_s=1)
        self.code = self.a.session.online.code
        self.b = FakeRig(clock=self.clock, level=1, mode="survival", target=target + 4, seed=2, cards=[],
                         flow=Flow(intro=False, start=("join", self.code)), online=online.Online(self.net, name="RAFAE", seed=2),
                         record_dir=record_dirs[1])
        self.a.player.idle, self.b.player.idle = host_idle, guest_idle

    def run(self, seconds, until=None):
        end = self.clock.now_ns() + round(seconds * S)
        while self.clock.now_ns() < end:
            self.a.step()
            self.b.step(advance=False)
            if until is not None and until(self):
                return True
        return until is None

    def screens(self):
        return self.a.session.hud_state().screen, self.b.session.hud_state().screen

    def close(self):
        self.a.close()
        self.b.close()


def selftest():
    """A host and a guest pair through the list and play a match to 3-0; -> a line saying so (AssertionError if they did not)."""
    m = Match(target=3, guest_idle=True, latency_s=0.04)
    try:
        assert m.run(20, until=lambda m: m.screens() == ("VS", "VS")), m.screens()
        assert m.run(150, until=lambda m: m.screens() == ("RESULTS", "RESULTS")), m.screens()
        host, guest = m.a.game, m.b.game
        assert (host.player_points, host.cpu_points) == (3, 0) == (guest.cpu_points, guest.player_points), (host.player_points, guest.player_points)
        assert m.a.client.published == [] and m.b.client.published == []                  # nothing a friend plays is ever published
        return f"two live pipelines paired through the list and played a match to {host.player_points}-{host.cpu_points} on both in {m.a.now_s():.0f} simulated s"
    finally:
        m.close()
