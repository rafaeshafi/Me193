"""PD control: the computer's paddle chases the predicted landing point after a reaction delay."""

import pytest

from pingpong import app, latency, levels, pd, physics, policy

CLUB, PRO, ROOKIE = levels.LEVELS[2], levels.LEVELS[3], levels.LEVELS[1]
LIVE = latency.Latency(display_s=0.0, loop_s=0.0)         # (the screen draws ahead by the display's delay: not what these test)


def test_the_output_is_proportional_plus_a_damping_term_and_is_saturated_at_the_speed_limit():
    c = pd.PDController(kp=10.0, kd=0.0, limit=3.0)
    assert c.step(0.1, 0.01) == pytest.approx(1.0)
    assert c.step(-0.1, 0.01) == pytest.approx(-1.0)
    assert c.step(5.0, 0.01) == 3.0 and c.step(-5.0, 0.01) == -3.0                    # saturated both ways
    damped = pd.PDController(kp=10.0, kd=0.5, limit=9.0, d_filter=1.0)
    damped.step(0.5, 0.01)
    assert damped.step(0.4, 0.01) < 10.0 * 0.4                                          # closing fast: braking


def test_reset_forgets_the_derivative_history():
    c = pd.PDController(kp=1.0, kd=1.0, limit=9.0, d_filter=1.0)
    c.step(1.0, 0.1)
    c.step(0.0, 0.1)
    c.reset()
    assert c.step(0.5, 0.1) == pytest.approx(0.5)                                       # no stale derivative


def test_the_paddle_stays_put_for_the_reaction_time_then_chases_the_target():
    x0 = 0.1
    assert pd.paddle_x(CLUB, 0.6, x0, CLUB.tau_s * 0.9) == pytest.approx(x0)
    assert pd.paddle_x(CLUB, 0.6, x0, CLUB.tau_s + 0.15) > x0 + 0.05


def test_the_paddle_never_moves_faster_than_the_levels_speed_limit():
    prev = pd.paddle_x(PRO, 1.4, 0.0, 0.0)
    for k in range(1, 100):
        t = k * 0.01
        x = pd.paddle_x(PRO, 1.4, 0.0, t)
        assert abs(x - prev) <= PRO.cpu_speed_ms * 0.01 + 1e-9
        prev = x


def test_a_target_beyond_reach_leaves_exactly_the_distance_the_paddle_could_not_cover():
    flight = 0.6
    reach = CLUB.cpu_speed_ms * (flight - CLUB.tau_s)
    assert pd.paddle_error_m(CLUB, reach + 0.3, 0.0, flight) == pytest.approx(0.3, abs=0.01)
    assert pd.paddle_error_m(CLUB, 0.5, 0.0, CLUB.tau_s) == pytest.approx(0.5)           # no time to move at all


def test_a_comfortably_reachable_ball_leaves_almost_no_error_even_at_pro_speed():
    for level in (ROOKIE, CLUB, PRO):
        flight = physics.D_M / level.v_tier
        reach = level.cpu_speed_ms * (flight - level.tau_s)
        assert pd.paddle_error_m(level, 0.5 * reach, 0.0, flight) < 0.03, level.name
        assert pd.paddle_error_m(level, 0.0, 0.0, flight) == 0.0


def test_the_approach_is_smooth_with_no_overshoot_past_the_target():
    xs = [pd.paddle_x(ROOKIE, 0.4, 0.0, t * 0.02) for t in range(60)]
    assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])) and max(xs) <= 0.4 + 1e-9


def test_the_policy_reach_deficit_is_the_pd_error():
    for x_land in (0.0, 0.3, 0.9, 1.4):
        assert policy.reach_deficit_m(CLUB, x_land, 0.0, 0.6) == pytest.approx(pd.paddle_error_m(CLUB, x_land, 0.0, 0.6))


# --- the HUD shows the paddle chasing your shot -----------------------------------------------------------------
def _wide_shot(mode, w_pk, min_x=0.3, level=1):
    """Play until a return that lands at least min_x metres from the centre; -> (session, the outgoing leg)."""
    for seed in range(1, 60):
        session = app.make_session(level=level, mode=mode, target=100, seed=seed, latency=LIVE)
        app.play_until_hits(session, 1, w_pk=w_pk)
        leg = session.game.outgoing_leg
        if abs(leg.x_end) >= min_x:
            return session, leg
    raise AssertionError("no wide shot found")


def _advance_to_arrival(session, leg, margin_s=0.005):
    session.clock.advance_s(max(0.0, (leg.arrival_ns - session.clock.now_ns()) / 1e9) - margin_s)


def test_during_your_shots_flight_the_computers_paddle_moves_toward_where_it_will_land():
    session, leg = _wide_shot("survival", 700.0)
    early = session.hud_state().cpu_x_m
    session.clock.advance_s(leg.flight_s * 0.8)
    late = session.hud_state().cpu_x_m
    assert abs(late - leg.x_end) < abs(early - leg.x_end)


def test_in_survival_the_paddle_always_gets_there_but_in_match_a_hard_shot_beats_it():
    survival, leg = _wide_shot("survival", 1200.0)
    _advance_to_arrival(survival, leg)
    assert abs(survival.hud_state().cpu_x_m - leg.x_end) < 0.06                         # "never misses": within the ring
    match, leg = _wide_shot("match", 1200.0)                        # Rookie: 0.4 s to react, the smash takes 0.3 s
    _advance_to_arrival(match, leg)
    assert abs(match.hud_state().cpu_x_m - leg.x_end) > 0.1                             # a smash it cannot reach


def test_after_its_return_the_computers_paddle_is_where_it_hit_and_drifts_back_to_the_middle():
    session, leg = _wide_shot("survival", 700.0)
    session.clock.advance_s(max(0.0, (leg.arrival_ns - session.clock.now_ns()) / 1e9) + 0.001)
    session.tick()                                                       # the computer hits the ball back
    assert session.game.incoming_leg.x_start == pytest.approx(leg.x_end)
    assert session.hud_state().cpu_x_m == pytest.approx(leg.x_end, abs=0.03)
    session.clock.advance_s(0.5)
    assert abs(session.hud_state().cpu_x_m) < 0.5 * abs(leg.x_end)       # on its way back


def test_the_computer_swings_when_it_serves_or_returns_for_about_a_third_of_a_second():
    session = app.make_session(level=1, latency=LIVE)
    assert session.hud_state().cpu_swing is None
    session.on_start()
    session.clock.advance_s(3.0)
    session.tick()                                                       # the first serve
    assert session.hud_state().cpu_swing == pytest.approx(0.0, abs=0.05)
    session.clock.advance_s(0.15)
    assert 0.4 < session.hud_state().cpu_swing < 0.6
    session.clock.advance_s(0.5)
    assert session.hud_state().cpu_swing is None


def test_the_paddle_is_back_in_the_centre_when_the_computer_serves():
    session = app.make_session(level=1)
    session.on_start()
    session.clock.advance_s(3.2)
    session.tick()
    assert session.game.incoming is not None and session.hud_state().cpu_x_m == 0.0
