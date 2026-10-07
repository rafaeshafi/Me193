"""GameCore: the rally lifecycle end to end, with real judge/shot/physics/policy/scoring."""

import random

import pytest

import config
from pingpong import levels
from pingpong.events import PaddlePose, SwingEvent
from pingpong.judge import HitJudge
from pingpong.mqtt_pub import ScorePublisher
from pingpong.paddle import ReachBox
from pingpong.policy import CpuPolicy
from pingpong.rules import GameCore
from pingpong.scoring import ScoreTracker
from pingpong.sources_fake import FakeMqttClient

S = 1_000_000_000
BOX = ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5)


class Script:
    """Deterministic CPU: returns the next scripted decision (Match tests)."""

    def __init__(self, decisions, inner=None):
        self.decisions = list(decisions)
        self.inner = inner or CpuPolicy(random.Random(1))

    def serve(self, *a, **kw):
        return self.inner.serve(*a, **kw)

    def returns(self, *a, **kw):
        return self.decisions.pop(0) if self.decisions else True


def make(mode="survival", level=1, target=7, policy=None, seed=1, scope="record_session"):
    client = FakeMqttClient()
    pub = ScorePublisher(client, scope=scope)
    game = GameCore(judge=HitJudge(BOX, t_pk=250.0), tracker=ScoreTracker(scope=scope), publisher=pub,
                    policy=policy or CpuPolicy(random.Random(seed)), level=levels.LEVELS[level], mode=mode,
                    target_points=target, omega_lo=300.0, omega_hi=1200.0)
    return game, client


def sent(client):
    return [p["payload"] for p in client.published if p["topic"] == config.SCORE_TOPIC]


def swing(t_ns, w_pk=600.0):
    return SwingEvent(kind="IMPACT", t_ns=t_ns, w_pk=w_pk, dur_ms=150.0, n_reversals=0,
                      axis_unit=(1, 0, 0), net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1),
                      clipped=False, feat=(0.0,) * 12)


def poses(t_i, ab, du=0.0, conf=0.9):
    u, v = BOX.to_uv(*ab)
    pts = [PaddlePose(t_scene_ns=int(t_i - 0.30 * S + k * 0.05 * S), u=u + du, v=v, conf=conf, hand="right")
           for k in range(6)]
    pts.append(PaddlePose(t_scene_ns=int(t_i + 0.02 * S), u=u + du, v=v, conf=conf, hand="right"))
    return pts


def start_rally(game, t0=0):
    game.start(t0)
    game.tick(t0 + 3 * S)
    assert game.phase == "RALLY"
    return t0 + 3 * S


def good_hit(game, w_pk=600.0, now=None):
    ball = game.incoming
    t = ball.t_c_ns
    return game.on_swing(swing(t, w_pk), poses(t, ball.aim_ab), now if now is not None else t)


def kinds(events):
    return [e.kind for e in events]


def next_serve(game):
    """Advance time to the CPU's return after a hit and return the events."""
    return game.tick(game.outgoing_leg.arrival_ns)


def test_start_runs_a_three_second_countdown_then_serves():
    game, _ = make()
    assert game.phase == "LOBBY"
    assert game.start(0) is True and game.phase == "COUNTDOWN"
    assert game.tick(int(2.9 * S)) == []
    events = game.tick(3 * S)
    assert "serve" in kinds(events) and game.phase == "RALLY"
    assert game.incoming_leg.flight_s == pytest.approx(3.0 / 3.5, rel=0.01)   # Rookie, s_prev 0.5


def test_start_is_ignored_mid_rally_and_level_changes_only_in_the_lobby():
    game, _ = make()
    assert game.set_level(levels.LEVELS[2]) is True and game.level.name == "Club"
    start_rally(game)
    assert game.start(10 * S) is False
    assert game.set_level(levels.LEVELS[3]) is False and game.level.name == "Club"


def test_a_good_swing_scores_publishes_and_the_cpu_returns_the_ball():
    game, client = make()
    start_rally(game)
    events = good_hit(game)
    assert "hit" in kinds(events)
    assert game.tracker.streak == 1 and sent(client) == ["1.0"]
    assert game.outgoing_leg is not None
    served = next_serve(game)
    assert "serve" in kinds(served) and game.incoming.ball_id == 2


def test_five_consecutive_hits_tick_the_score_up_live():
    game, client = make()
    start_rally(game)
    for _ in range(5):
        good_hit(game)
        next_serve(game)
    assert sent(client) == ["1.0", "2.0", "3.0", "4.0", "5.0"]
    assert game.tracker.record == 5


def test_a_miss_ends_survival_and_the_record_holds_on_the_broker():
    game, client = make()
    start_rally(game)
    for _ in range(2):
        good_hit(game)
        next_serve(game)
    events = game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)
    assert kinds(events)[:2] == ["miss", "rally_end"] and "game_over" in kinds(events)
    assert game.phase == "MATCH_OVER"
    assert sent(client) == ["1.0", "2.0"]
    over = [e for e in events if e.kind == "game_over"][0]
    assert over.data["streak"] == 2 and over.data["record"] == 2


def test_a_weak_swing_is_rejected_but_the_ball_stays_live_for_a_proper_one():
    game, client = make()
    start_rally(game)
    t = game.incoming.t_c_ns
    rejected = game.on_swing(swing(t, w_pk=100.0), poses(t, game.incoming.aim_ab), t)
    assert "rejected" in kinds(rejected) and game.tracker.streak == 0
    assert "hit" in kinds(good_hit(game))


def test_an_early_practice_swing_is_ignored_silently():
    game, _ = make()
    start_rally(game)
    t = game.incoming.t_c_ns - 1 * S
    events = game.on_swing(swing(t), poses(t, game.incoming.aim_ab), t)
    assert "hit" not in kinds(events) and game.tracker.streak == 0 and game.phase == "RALLY"


def test_swings_outside_a_rally_do_nothing():
    game, client = make()
    assert game.on_swing(swing(5 * S), [], 5 * S) == []
    assert sent(client) == []


def test_a_hard_sloppy_swing_faults_and_does_not_score():
    game, client = make()
    start_rally(game)
    ball = game.incoming
    R, E = ball.level.radius_sw, ball.level.early_s
    t = ball.t_c_ns - int(0.9 * E * S)                          # near the early edge ...
    events = game.on_swing(swing(t, w_pk=1500.0), poses(t, ball.aim_ab, du=0.9 * R), t)   # ... and far off
    assert "fault" in kinds(events)
    assert [e for e in events if e.kind == "fault"][0].data["fault"] == "out"
    assert game.tracker.streak == 0 and sent(client) == []
    assert game.phase == "MATCH_OVER"


def test_next_serve_speed_follows_the_players_last_swing_strength():
    soft, _ = make(level=2)
    start_rally(soft)
    good_hit(soft, w_pk=350.0)
    next_serve(soft)
    hard, _ = make(level=2)
    start_rally(hard)
    good_hit(hard, w_pk=1500.0)
    next_serve(hard)
    assert hard.incoming_leg.v > soft.incoming_leg.v
    # full-strength swing: v_tier * (0.9 + 0.2 * 1.0) with the Survival ramp after one hit
    expected = levels.incoming_speed(levels.LEVELS[2], 1.0, ramp=levels.survival_ramp(1))
    assert hard.incoming_leg.v == pytest.approx(expected)


def test_survival_ramps_the_ball_speed_over_the_rally():
    game, _ = make(level=1)
    start_rally(game)
    first = game.incoming_leg.v
    for _ in range(14):
        good_hit(game)
        next_serve(game)
    assert game.incoming_leg.v > first * 1.25


def test_the_first_rally_only_builds_the_best_so_nothing_is_announced_as_a_record():
    # Every hit of the first rally is technically a new best; announcing each one would buzz the
    # motors on every hit and hide the timing cues the player needs.
    game, _ = make()
    start_rally(game)
    flagged = []
    for _ in range(4):
        flagged += [e for e in good_hit(game) if e.kind == "record"]
        next_serve(game)
    assert flagged == [] and game.tracker.record == 4


def test_passing_the_previous_best_is_announced_once_per_rally_not_for_every_hit_after_it():
    game, _ = make()
    start_rally(game)
    for _ in range(2):                                           # best so far: 2
        good_hit(game)
        next_serve(game)
    game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)    # a miss ends the game
    assert game.phase == "MATCH_OVER"
    start_rally(game, 100 * S)                                   # a second game
    flagged = []
    for _ in range(4):
        flagged += [e for e in good_hit(game) if e.kind == "record"]
        next_serve(game)
    assert [e.data["value"] for e in flagged] == [3]             # 3 passes 2; 4, 5 and 6 are not re-announced


def test_match_a_cpu_miss_gives_the_player_the_point_and_the_cpu_serves_again():
    game, client = make(mode="match", target=2, policy=Script([False, True, False]))
    start_rally(game)
    good_hit(game)
    events = next_serve(game)                                   # CPU misses: point to the player
    assert "point" in kinds(events) and game.player_points == 1 and game.cpu_points == 0
    assert game.phase == "POINT_OVER"
    again = game.tick(game.point_over_until_ns)
    assert "serve" in kinds(again) and game.phase == "RALLY"     # receive-only: CPU serves every point


def test_match_player_miss_gives_the_cpu_a_point_and_the_match_ends_at_the_target():
    game, _ = make(mode="match", target=2, policy=Script([]))
    start_rally(game)
    game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)
    assert game.cpu_points == 1 and game.phase == "POINT_OVER"
    game.tick(game.point_over_until_ns)
    events = game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)
    assert game.cpu_points == 2 and game.phase == "MATCH_OVER"
    over = [e for e in events if e.kind == "match_over"][0]
    assert over.data["winner"] == "cpu"


def test_match_to_eleven_must_be_won_by_two():
    game, _ = make(mode="match", target=11, policy=Script([]))
    game.player_points, game.cpu_points = 10, 10
    start_rally(game)
    game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)    # 10-11: not over
    assert game.phase == "POINT_OVER"
    game.tick(game.point_over_until_ns)
    game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)    # 10-12: over
    assert game.phase == "MATCH_OVER"


def test_starting_again_after_game_over_resets_the_scoreboard_but_not_the_record():
    game, client = make()
    start_rally(game)
    good_hit(game)
    game.tick(game.outgoing_leg.arrival_ns)
    game.tick(game.judge.miss_deadline_ns(game.incoming) + 1)
    assert game.phase == "MATCH_OVER"
    assert game.start(100 * S) is True and game.phase == "COUNTDOWN"
    assert game.tracker.streak == 0 and game.tracker.record == 1


def test_the_hit_event_carries_what_the_hud_and_haptics_need():
    game, _ = make()
    start_rally(game)
    hit = [e for e in good_hit(game, w_pk=900.0) if e.kind == "hit"][0]
    for key in ("label", "v_out", "kmh", "topspin", "sidespin", "q_total", "streak", "gates"):
        assert key in hit.data
    assert hit.data["label"] in ("perfect", "good", "early", "late")
    assert hit.data["kmh"] == pytest.approx(3.6 * hit.data["v_out"])


# --- PAUSED: a silent hub or lost camera must never cause a hit or a fault -------------------------

def test_pausing_freezes_the_ball_clock_and_resuming_shifts_every_deadline():
    game, client = make()
    start_rally(game)
    t_c0 = game.incoming.t_c_ns
    deadline0 = game.judge.miss_deadline_ns(game.incoming)
    t_pause = t_c0 - S // 2
    game.set_pause("hub", True, t_pause)
    assert game.paused is True
    assert game.tick(deadline0 + 10 * S) == []                      # no miss while paused
    assert game.on_swing(swing(t_c0), poses(t_c0, game.incoming.aim_ab), t_c0) == []
    t_resume = deadline0 + 10 * S
    game.set_pause("hub", False, t_resume)
    assert game.paused is False
    assert game.incoming.t_c_ns == t_c0 + (t_resume - t_pause)
    assert game.tick(t_resume) == []                                # the ball is live again, not missed
    t = game.incoming.t_c_ns
    assert "hit" in kinds(game.on_swing(swing(t), poses(t, game.incoming.aim_ab), t))


def test_pause_reasons_combine_and_every_one_must_clear_before_play_resumes():
    game, _ = make()
    start_rally(game)
    t = game.incoming.t_c_ns - S // 2
    game.set_pause("hub", True, t)
    game.set_pause("pose", True, t + 1000)
    game.set_pause("hub", False, t + S)
    assert game.paused is True and game.pause_reasons == {"pose"}
    game.set_pause("pose", False, t + 2 * S)
    assert game.paused is False


def test_pausing_during_the_countdown_and_the_cpu_return_also_shifts_the_timers():
    game, _ = make()
    game.start(0)
    game.set_pause("hub", True, 1 * S)
    assert game.tick(10 * S) == []                                   # the serve does not happen while paused
    game.set_pause("hub", False, 10 * S)
    assert game.tick(10 * S + 1) == []
    assert "serve" in kinds(game.tick(10 * S + 2 * S + 1))           # 2 s of the countdown were left


def test_pausing_outside_a_rally_is_harmless():
    game, _ = make()
    game.set_pause("hub", True, 0)
    assert game.start(1 * S) is False or game.phase in ("LOBBY", "COUNTDOWN")
    game.set_pause("hub", False, 2 * S)
    assert game.paused is False


def test_a_miss_waits_until_the_imu_data_has_caught_up_with_the_deadline():
    # "No swing" is only a fact once the IMU stream has reached the deadline.  If the hub goes
    # quiet around the swing window the ball must not be called a miss by the wall clock alone.
    game, _ = make()
    start_rally(game)
    ball = game.incoming
    deadline = game.judge.miss_deadline_ns(ball)
    quiet_since = ball.t_c_ns - int(0.1 * S)                     # newest IMU sample before the gap
    assert game.tick(deadline + int(0.2 * S), data_ns=quiet_since) == []
    assert game.phase == "RALLY"
    assert "miss" in kinds(game.tick(deadline + int(0.2 * S), data_ns=deadline + 1))


def test_without_a_data_horizon_the_wall_clock_decides_as_before():
    game, _ = make()
    start_rally(game)
    assert "miss" in kinds(game.tick(game.judge.miss_deadline_ns(game.incoming) + 1))


def test_the_game_remembers_exactly_when_it_was_started_so_a_replay_can_reproduce_it():
    game, _ = make()
    assert game.started_at_ns is None
    game.start(123_456_789)
    assert game.started_at_ns == 123_456_789
