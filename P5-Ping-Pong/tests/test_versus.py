"""Playing another person: two complete games, each on its own laptop, joined by a cable with latency.

Each game runs its own ball and decides for itself whether the ball it was sent was returned.  What crosses the cable is what the
other person did (I hit it like this, I missed), so latency can slow a rally but never changes how long anyone has to react."""

import dataclasses
import random

import pytest

from pingpong import levels, netproto as proto, physics, versus
from tests.versus_harness import Rig, S

SID = "abc123"
HIT = {"t": "hit", "n": 1, "v": 3.0, "start": [0.2, 0.22, 0.3], "aim": [0.3, 0.5], "top": 0.0, "side": 0.0, "loft": 0.0, "serve": False,
       "sid": SID}


def mirror(point):
    x, y, z = point
    return -x, y, physics.TABLE_LEN_M - z


def kinds(side):
    return [e.kind for e in side.events]


# --- the pace -------------------------------------------------------------------------------------------------------------------------------
def test_the_pace_of_a_match_sets_how_fast_the_ball_goes_and_a_harder_hit_makes_it_a_little_faster():
    for tag in (1, 2, 3):
        level = levels.LEVELS[tag]
        soft, hard = versus.versus_speed(level, 0.0), versus.versus_speed(level, 1.0)
        assert 0.75 * level.v_tier <= soft < hard <= 1.35 * level.v_tier
    assert versus.versus_speed(levels.LEVELS[3], 0.0) > versus.versus_speed(levels.LEVELS[1], 1.0)


# --- one ball across the cable ---------------------------------------------------------------------------------------------------------------
def test_the_host_serves_first_and_the_ball_comes_at_the_guest_as_the_mirror_image_of_the_one_the_host_sent():
    rig = Rig(latency_s=0.05, host_skill="idle", guest_skill="idle").start()
    assert rig.run(6, until=lambda r: r.guest.game.incoming_leg is not None)
    out, inc = rig.host.game.outgoing_leg, rig.guest.game.incoming_leg
    assert out.z_end == pytest.approx(physics.TABLE_LEN_M - physics.HIT_Z_M) and inc.z_end == pytest.approx(physics.HIT_Z_M)
    for k in range(0, 11):
        dt = round(out.flight_s * 1e9 * k / 10)
        assert inc.position(inc.t0_ns + dt) == pytest.approx(mirror(out.position(out.t0_ns + dt)), abs=1e-6)
    assert inc.t0_ns - out.t0_ns == pytest.approx(0.05 * S, abs=0.011 * S)                        # it came after the latency, no sooner


def guest_waiting_for_a_serve():
    """Only the guest is playing (the host is the test): it counts down, and then waits for the other person's serve."""
    rig = Rig(host_skill="idle", guest_skill="idle")
    rig.guest.session.on_start()
    rig.run(3.2)
    return rig


def test_a_ball_arrives_with_the_spin_the_other_person_gave_it_turned_round():
    rig = guest_waiting_for_a_serve()
    assert rig.guest.game.phase == "RALLY" and rig.guest.game.incoming_leg is None
    rig.link_a.send(dict(HIT, side=0.5, top=-0.25, v=4.0, serve=True))
    rig.run(0.2)
    leg = rig.guest.game.incoming_leg
    assert leg is not None and leg.sidespin == pytest.approx(-0.5) and leg.topspin == pytest.approx(-0.25) and leg.v == pytest.approx(4.0)
    assert [e.kind for e in rig.guest.events if e.kind == "serve"] == ["serve"]
    assert rig.guest.game.phase == "RALLY" and rig.guest.game.incoming.level is rig.guest.game.level


def test_the_balls_a_person_hits_count_as_hits_with_the_usual_fuss_and_leave_at_the_pace_of_the_match():
    rig = Rig(level=2, host_skill="perfect", guest_skill="perfect", target=7).start()
    assert rig.run(25, until=lambda r: r.guest.game.tracker.streak >= 2 and r.host.game.tracker.streak >= 2)
    hits = [e for e in rig.guest.events if e.kind == "hit"]
    level = levels.LEVELS[2]
    assert hits and all(0.75 * level.v_tier <= e.data["v_out"] <= 1.35 * level.v_tier for e in hits)
    assert all("contact_ns" in e.data for e in hits)


# --- points ----------------------------------------------------------------------------------------------------------------------------------------
def test_a_ball_nobody_returns_is_a_point_for_the_server_on_both_screens_and_then_the_other_one_serves():
    rig = Rig(host_skill="idle", guest_skill="idle").start()
    assert rig.run(20, until=lambda r: r.host.game.player_points == 1 and r.guest.game.cpu_points == 1)
    assert rig.scores() == ((1, 0), (1, 0))
    assert "miss" in kinds(rig.guest) and "point" in kinds(rig.host) and "point" in kinds(rig.guest)
    assert [e.data["scorer"] for e in rig.host.events if e.kind == "point"] == ["player"]
    assert [e.data["scorer"] for e in rig.guest.events if e.kind == "point"] == ["cpu"]
    assert rig.run(12, until=lambda r: r.guest.game.outgoing_leg is not None and r.guest.game.phase == "RALLY")      # the guest serves the next
    assert rig.run(12, until=lambda r: r.host.game.incoming_leg is not None)


def test_a_match_to_three_ends_on_both_screens_with_the_same_winner_and_the_same_score():
    rig = Rig(host_skill="perfect", guest_skill="idle", target=3).start()
    assert rig.run(120, until=lambda r: r.host.game.phase == "MATCH_OVER" and r.guest.game.phase == "MATCH_OVER")
    assert rig.scores() == ((3, 0), (3, 0))
    assert [e.data["winner"] for e in rig.host.events if e.kind == "match_over"] == ["player"]
    assert [e.data["winner"] for e in rig.guest.events if e.kind == "match_over"] == ["cpu"]


def test_the_server_alternates_one_point_each():
    rig = Rig(host_skill="idle", guest_skill="idle", target=5).start()
    servers = []

    def watch(r):
        for name, side in (("host", r.host), ("guest", r.guest)):
            if side.game.phase == "RALLY" and side.game.outgoing_leg is not None and side.game.incoming is None and side.game.outgoing_leg.t0_ns not in seen:
                seen.add(side.game.outgoing_leg.t0_ns)
                servers.append(name)
        return len(servers) >= 4

    seen = set()
    assert rig.run(60, until=watch)
    assert servers[:4] == ["host", "guest", "host", "guest"]


# --- the cable ----------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("latency", [0.0, 0.08, 0.25, 0.6])
def test_a_long_delay_slows_a_rally_but_nobody_loses_a_ball_they_could_have_reached(latency):
    rig = Rig(latency_s=latency, level=1, host_skill="perfect", guest_skill="perfect", target=7).start()
    rig.run(45)
    assert rig.scores() == ((0, 0), (0, 0))
    assert rig.host.game.tracker.streak >= 3 and rig.guest.game.tracker.streak >= 3


def test_losing_unreliable_messages_changes_nothing_about_the_game():
    rig = Rig(loss0=0.9, host_skill="perfect", guest_skill="idle", target=2).start()
    for side in rig.sides:
        side.remote.set_paddle(0.3)
    assert rig.run(90, until=lambda r: r.host.game.phase == "MATCH_OVER" and r.guest.game.phase == "MATCH_OVER")
    assert rig.scores() == ((2, 0), (2, 0))


def test_a_reliable_message_that_arrives_twice_makes_one_ball_and_one_point():
    rig = Rig(dup1=1.0, host_skill="perfect", guest_skill="idle", target=2).start()
    assert rig.run(90, until=lambda r: r.host.game.phase == "MATCH_OVER" and r.guest.game.phase == "MATCH_OVER")
    assert rig.scores() == ((2, 0), (2, 0))
    assert kinds(rig.guest).count("miss") == 2 and kinds(rig.host).count("point") == 2


def test_the_two_screens_never_disagree_about_the_score_for_longer_than_a_message_takes():
    rig = Rig(latency_s=0.04, host_skill="perfect", guest_skill="idle", target=3).start()
    worst = 0
    while rig.clock.now_ns() < 150 * S and not (rig.host.game.phase == "MATCH_OVER" and rig.guest.game.phase == "MATCH_OVER"):
        rig.run(0.01)
        (hh, hg), (gh, gg) = rig.scores()
        if (hh, hg) != (gh, gg):
            worst += 1
    assert rig.scores() == ((3, 0), (3, 0)) and worst <= 3 * (4 + 2)                                # a message's flight and a frame or two, per point


def test_when_the_cable_goes_both_games_wait_and_when_it_comes_back_they_carry_on():
    rig = Rig(latency_s=0.03, host_skill="perfect", guest_skill="perfect").start()
    rig.run(8)
    rig.link_a.cut()
    rig.run(6)
    assert rig.host.game.paused and rig.guest.game.paused
    assert "opponent" in rig.host.game.pause_reasons and rig.host.remote.gone is False
    rig.link_a.mend()
    assert rig.run(6, until=lambda r: not r.host.game.paused and not r.guest.game.paused)
    rig.run(20)
    assert rig.scores() == ((0, 0), (0, 0)) and rig.host.game.tracker.streak >= 2


def test_a_cable_that_stays_cut_ends_the_match_for_both_after_twenty_seconds_as_lost():
    rig = Rig(host_skill="perfect", guest_skill="perfect").start()
    rig.run(5)
    rig.link_a.cut()
    assert not rig.run(15, until=lambda r: r.host.remote.gone)
    assert rig.run(8, until=lambda r: r.host.remote.gone and r.guest.remote.gone)
    assert rig.host.remote.reason == "lost" and rig.guest.remote.reason == "lost"


def test_a_goodbye_ends_it_at_once_and_says_the_other_left():
    rig = Rig().start()
    rig.run(1)
    rig.link_a.send({"t": "bye", "reason": "left"})
    rig.run(0.2)
    assert rig.guest.remote.gone and rig.guest.remote.reason == "left"


def test_a_lost_laptop_the_brokers_will_announces_is_noticed_without_waiting():
    rig = Rig().start()
    rig.run(1)
    rig.link_a.send({"t": "bye", "reason": "lost"})
    rig.run(0.2)
    assert rig.guest.remote.gone and rig.guest.remote.reason == "lost"


def test_a_goodbye_that_names_another_guest_or_another_session_is_not_my_partner_leaving():
    rig = Rig().start()
    rig.run(1)
    rig.link_b.send({"t": "bye", "reason": "left", "gid": "someoneelse"})                  # a guest the host turned away, going
    rig.link_a.send({"t": "bye", "reason": "left", "sid": "othersession"})
    rig.run(0.3)
    assert not rig.host.remote.gone and not rig.guest.remote.gone
    rig.link_b.send({"t": "bye", "reason": "left", "gid": "k3x9"})                         # my partner, by its token
    rig.link_a.send({"t": "bye", "reason": "lost", "sid": SID})                            # and the host, by the session
    rig.run(0.3)
    assert rig.host.remote.gone and rig.guest.remote.gone


def test_a_second_guest_who_asks_to_join_a_game_that_has_begun_is_told_it_is_busy():
    rig = Rig().start()
    rig.run(0.5)
    rig.link_b.poll()
    rig.link_b.send({"t": "join", "name": "LATE", "gid": "zz11"})
    rig.link_b.send({"t": "join", "name": "MAYA", "gid": "k3x9"})                          # (my own guest asking again: no answer)
    for _ in range(30):
        rig.clock.advance_s(0.01)
        rig.host.tick()
    answers = [m for m in rig.link_b.poll() if m["t"] == "busy"]
    assert [m["to"] for m in answers] == ["zz11"]
    assert not rig.host.remote.gone


def test_a_rematch_asked_for_is_seen_once_and_a_copy_of_it_later_is_not_a_second_one():
    rig = Rig().start()
    rig.run(1)
    assert not rig.guest.remote.rematch_seen
    rig.host.remote.send_rematch()
    rig.run(0.3)
    assert rig.guest.remote.rematch_seen
    rig.guest.remote.rematch_seen = False                                                  # (a new game started and forgot it)
    rig.link_a.send({"t": "rematch", "n": rig.host.remote._n, "sid": SID})                 # the broker delivers it twice
    rig.run(0.3)
    assert not rig.guest.remote.rematch_seen
    rig.host.remote.send_rematch()
    rig.run(0.3)
    assert rig.guest.remote.rematch_seen


def test_a_new_game_forgets_a_rematch_that_started_it():
    rig = Rig().start()
    rig.guest.remote.rematch_seen = True
    rig.guest.session.game.phase = "MATCH_OVER"
    rig.guest.session.on_start()
    assert not rig.guest.remote.rematch_seen


def test_when_the_other_person_leaves_the_game_ends_where_it_stands_for_the_one_who_stays():
    for word in ("left", "lost"):
        rig = Rig(host_skill="perfect", guest_skill="perfect", target=7).start()
        rig.run(7)
        assert rig.guest.game.phase in ("RALLY", "POINT_OVER")
        rig.link_a.send({"t": "bye", "reason": word})
        rig.run(0.3)
        game = rig.guest.game
        assert game.phase == "MATCH_OVER" and not game.paused and game.incoming is None and game.outgoing_leg is None
        over = [e for e in rig.guest.events if e.kind == "match_over"]
        assert len(over) == 1 and over[0].data["walkover"] is True and over[0].data["winner"] is None


def test_a_silent_cable_pauses_the_game_and_then_ends_it_as_a_walkover_with_no_pause_left_behind():
    rig = Rig(host_skill="perfect", guest_skill="perfect").start()
    rig.run(6)
    rig.link_a.cut()
    assert rig.run(30, until=lambda r: r.guest.game.phase == "MATCH_OVER")
    assert not rig.guest.game.paused and rig.guest.remote.reason == "lost"
    assert [e.data["walkover"] for e in rig.guest.events if e.kind == "match_over"] == [True]


def test_a_game_that_is_already_over_has_no_walkover():
    rig = Rig(host_skill="perfect", guest_skill="idle", target=1).start()
    assert rig.run(30, until=lambda r: r.guest.game.phase == "MATCH_OVER")
    before = len([e for e in rig.guest.events if e.kind == "match_over"])
    rig.link_a.send({"t": "bye", "reason": "left"})
    rig.run(0.3)
    assert len([e for e in rig.guest.events if e.kind == "match_over"]) == before == 1


# --- trust nothing -------------------------------------------------------------------------------------------------------------------------------------
def test_junk_and_messages_from_another_session_do_nothing():
    rig = Rig(host_skill="idle", guest_skill="idle").start()
    rig.run(0.2)
    for junk in (b"", b"garbage", b'{"ver":1,"t":"hit"}', b"\xff" * 50, proto.encode({"t": "pos", "x": 9e9})):
        rig.link_b.inject_raw(junk)
    rig.link_a.send(dict(HIT, sid="zzzzzz"))
    rig.link_a.send({"t": "miss", "n": 1, "reason": "miss", "score": [5, 0], "sid": "zzzzzz"})
    rig.run(0.5)
    assert rig.guest.game.incoming_leg is None and rig.scores() == ((0, 0), (0, 0)) and not rig.guest.remote.gone


def test_a_ball_while_one_is_already_coming_is_ignored_and_a_repeat_by_number_is_ignored():
    rig = guest_waiting_for_a_serve()
    rig.link_a.send(dict(HIT, n=1, v=3.0))
    rig.run(0.2)
    first = rig.guest.game.incoming_leg
    assert first is not None
    rig.link_a.send(dict(HIT, n=2, v=9.0))
    rig.link_a.send(dict(HIT, n=1, v=9.0))
    rig.run(0.3)
    assert rig.guest.game.incoming_leg is first


def test_a_wrong_score_is_put_right_by_the_hosts_next_word():
    rig = Rig(host_skill="idle", guest_skill="idle", target=9).start()
    rig.run(1)
    rig.guest.game.cpu_points, rig.guest.game.player_points = 5, 5
    assert rig.run(30, until=lambda r: r.host.game.player_points == 1 and r.guest.game.cpu_points == 1)
    rig.run(1)
    assert rig.scores() == ((1, 0), (1, 0))


# --- who serves, and what is drawn ----------------------------------------------------------------------------------------------------------------------
def test_a_serve_that_arrives_before_the_other_game_has_started_waits_for_it():
    rig = Rig(host_skill="idle", guest_skill="idle")
    rig.host.session.on_start()
    rig.run(4.5)
    assert rig.guest.game.phase == "LOBBY" and rig.guest.game.incoming_leg is None
    rig.guest.session.on_start()
    assert rig.run(8, until=lambda r: r.guest.game.incoming_leg is not None)
    assert rig.guest.game.phase == "RALLY"


def test_the_opponents_paddle_is_where_they_hold_it_turned_round_for_the_far_end():
    rig = Rig(latency_s=0.04).start()
    rig.host.remote.set_paddle(0.4)
    rig.run(1)
    assert rig.guest.remote.paddle_x == pytest.approx(-0.4, abs=0.05)
    rig.host.remote.set_paddle(-0.2)
    rig.run(1)
    assert rig.guest.remote.paddle_x == pytest.approx(0.2, abs=0.05)


def test_the_round_trip_is_measured_for_the_screen():
    rig = Rig(latency_s=0.07).start()
    rig.run(4)
    assert rig.host.remote.ping_ms == pytest.approx(140, abs=25) and rig.guest.remote.ping_ms == pytest.approx(140, abs=25)


def test_the_opponent_has_a_name_for_the_screen():
    rig = Rig()
    assert rig.host.remote.opponent_name == "MAYA" and rig.guest.remote.opponent_name == "RAFAE"


def test_the_serve_is_a_slow_clean_ball_to_somewhere_in_the_middle_of_the_table():
    rig = Rig(level=2, host_skill="idle", guest_skill="idle").start()
    assert rig.run(6, until=lambda r: r.guest.game.incoming_leg is not None)
    leg = rig.guest.game.incoming_leg
    assert leg.v == pytest.approx(levels.LEVELS[2].v_tier) and leg.topspin == 0.0 and leg.sidespin == 0.0
    assert abs(leg.x_end) < 0.6 * physics.HALF_WIDTH_M


def test_the_same_seed_serves_the_same_balls():
    def serves(seed):
        rig = Rig(seed=seed, host_skill="idle", guest_skill="idle").start()
        rig.run(6, until=lambda r: r.guest.game.incoming_leg is not None)
        return rig.guest.game.incoming_leg.x_end

    assert serves(3) == serves(3) and serves(3) != serves(4)


def test_nothing_in_a_game_with_another_person_ever_touches_the_score_topic():
    rig = Rig(host_skill="perfect", guest_skill="idle", target=1).start()
    rig.run(60, until=lambda r: r.host.game.phase == "MATCH_OVER")
    assert rig.host.game.publisher is None and rig.guest.game.publisher is None


# --- many matches, a bad network and players who miss at random ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_whatever_the_network_and_the_players_a_match_ends_with_both_screens_agreeing(seed):
    rnd = random.Random(seed)
    rig = Rig(seed=seed, latency_s=rnd.choice([0.0, 0.02, 0.08, 0.2, 0.45]), jitter_s=rnd.choice([0.0, 0.03, 0.12]),
              loss0=rnd.choice([0.0, 0.3]), dup1=rnd.choice([0.0, 0.5]), level=rnd.choice([1, 2, 3]), target=rnd.choice([2, 3]),
              host_skill="flaky", guest_skill="flaky").start()
    done = rig.run(400, until=lambda r: r.host.game.phase == "MATCH_OVER" and r.guest.game.phase == "MATCH_OVER")
    assert done, rig.scores()
    rig.run(1.0)
    assert rig.scores()[0] == rig.scores()[1]
    (h, g), _ = rig.scores()
    assert max(h, g) >= rig.host.game.target_points and abs(h - g) >= 1
    winners = ([e.data["winner"] for e in rig.host.events if e.kind == "match_over"], [e.data["winner"] for e in rig.guest.events if e.kind == "match_over"])
    assert winners[0] == ["player" if h > g else "cpu"] and winners[1] == ["cpu" if h > g else "player"]
    for side in rig.sides:
        assert not side.game.paused and side.game.publisher is None


# --- the hand into the ball (the way the live game is played) --------------------------------------------------------------------------------------
def test_a_match_in_contact_mode_is_played_to_the_end_across_the_cable_with_the_same_score_on_both():
    rig = Rig(hit_mode="contact", host_skill="perfect", guest_skill="idle", target=3, latency_s=0.06).start()
    assert rig.run(90, until=lambda r: r.host.game.phase == "MATCH_OVER" and r.guest.game.phase == "MATCH_OVER")
    assert rig.scores() == ((3, 0), (3, 0))
    assert "hit" in kinds(rig.host) and "hit" not in kinds(rig.guest)


def test_two_hands_into_the_ball_keep_a_rally_going_across_the_cable_and_a_flaky_one_loses_points_to_the_other():
    rig = Rig(hit_mode="contact", host_skill="perfect", guest_skill="perfect", target=9, latency_s=0.05).start()
    assert rig.run(60, until=lambda r: r.host.game.tracker.streak >= 6)
    rig = Rig(hit_mode="contact", host_skill="perfect", guest_skill="flaky", target=4, latency_s=0.05, seed=3).start()
    assert rig.run(240, until=lambda r: r.host.game.phase == "MATCH_OVER" and r.guest.game.phase == "MATCH_OVER")
    as_host_sees, as_guest_sees = rig.scores()
    assert as_host_sees == as_guest_sees and max(as_host_sees) == 4
