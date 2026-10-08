"""Two whole live pipelines playing each other: each a FakeRig (fake hub, camera and tags through the real wiring of live.assemble) with
its own flow, its own online manager and a scripted player, joined by an in-memory network on one shared clock.  The hub, the camera
and the broker are fake; everything between is what a player runs."""

from pingpong import online
from pingpong.clock import FakeClock
from pingpong.fakerig import FakeRig
from pingpong.flow import Flow
from pingpong.loopnet import LoopNet

S = 1_000_000_000


class Match:
    """A host's rig and a guest's, on one clock; the guest joins the host's game by its code."""

    def __init__(self, *, target=3, pace=2, host_idle=False, guest_idle=False, **net_kw):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.net = LoopNet(self.clock, **net_kw)
        self.a = FakeRig(clock=self.clock, level=1, mode="survival", target=target, seed=1, cards=[],
                         flow=Flow(intro=False, level_tag=pace, start=("host", pace)), online=online.Online(self.net, name="MAYA", seed=1))
        self.a.run(until=lambda: self.a.session.online.code, seconds=0.5, max_s=1)
        self.code = self.a.session.online.code
        self.b = FakeRig(clock=self.clock, level=1, mode="survival", target=target + 4, seed=2, cards=[],
                         flow=Flow(intro=False, start=("join", self.code)), online=online.Online(self.net, name="RAFAE", seed=2))
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


def test_two_live_pipelines_pair_through_the_list_and_play_a_match_to_its_end_with_the_same_score():
    m = Match(target=3, guest_idle=True, latency_s=0.04)
    try:
        assert m.run(20, until=lambda m: m.screens() == ("VS", "VS"))
        assert m.run(10, until=lambda m: m.screens() == ("GAME", "GAME"))
        assert m.run(120, until=lambda m: m.screens() == ("RESULTS", "RESULTS"))
        for rig in (m.a, m.b):
            assert rig.game.target_points == 3 and rig.game.level.tag == 2 and rig.game.mode == "match"
        assert (m.a.game.player_points, m.a.game.cpu_points) == (3, 0) == (m.b.game.cpu_points, m.b.game.player_points)
        assert m.a.session.hud_state().results.title == "YOU WIN!" and m.b.session.hud_state().results.title == "GOOD GAME!"
    finally:
        m.close()


def test_nothing_is_published_to_any_broker_by_a_friends_game():
    m = Match(target=2, guest_idle=True)
    try:
        assert m.run(120, until=lambda m: m.screens() == ("RESULTS", "RESULTS"))
        assert m.a.client.published == [] and m.b.client.published == []                 # neither the score topic nor anything else
        assert m.a.game.publisher is None and m.b.game.publisher is None
    finally:
        m.close()


def test_a_long_rally_between_two_players_over_a_bad_network_goes_on_and_the_match_still_ends_in_agreement():
    m = Match(target=3, latency_s=0.15, jitter_s=0.08, loss0=0.3, dup1=0.2)
    try:
        assert m.run(40, until=lambda m: m.a.game.tracker.streak >= 5)
        m.b.player.idle = True                                                          # now the guest stops returning balls
        assert m.run(300, until=lambda m: m.screens() == ("RESULTS", "RESULTS"))
        assert (m.a.game.player_points, m.a.game.cpu_points) == (m.b.game.cpu_points, m.b.game.player_points)
        assert m.a.game.player_points == 3 and not m.a.game.paused and not m.b.game.paused
    finally:
        m.close()


def test_closing_one_laptop_in_the_middle_of_a_match_ends_it_for_the_other_with_no_winner_named():
    m = Match(target=9)
    try:
        assert m.run(60, until=lambda m: m.a.game.tracker.streak >= 3)
        m.b.close()                                                                     # their window closes: goodbye, then the list
        assert m.run(5, until=lambda m: m.a.session.hud_state().screen == "RESULTS")
        state = m.a.session.hud_state()
        assert state.results.title == "THEY LEFT" and state.ui.opponent_gone and m.a.game.phase == "MATCH_OVER"
        assert m.net.entries == {}
    finally:
        m.a.close()
