"""Analytic ball legs in 3-D: exact arrival time and point, a real bounce, stylised spin (no integrator).

World metres: x across the table (positive to the player's right), y up from the table top, z along the table from the
player's edge (0) to the computer's (2.74); the net stands at z = 1.37.
"""

import pytest

from pingpong import physics

S = 1_000_000_000


def leg(v=5.0, top=0.0, side=0.0, x_start=0.0, aim=(0.5, 0.5), fault=None):
    return physics.plan_leg(t0_ns=0, v=v, x_start=x_start, aim_ab=aim, topspin=top, sidespin=side, fault=fault)


def ret(v=5.0, start=(0.2, 0.22, 0.3), aim=(0.5, 0.5), top=0.0, side=0.0, fault=None):
    return physics.plan_return(t0_ns=0, v=v, start=start, aim_ab=aim, topspin=top, sidespin=side, fault=fault)


def track(l, n=200):
    """(x, y, z) of the ball at n+1 even steps from launch to arrival."""
    return [l.position(round(i * l.flight_s / n * S)) for i in range(n + 1)]


# --- time ---------------------------------------------------------------------------------------------------------
def test_flat_leg_arrives_exactly_distance_over_speed_later():
    l = leg(v=5.0)
    assert l.flight_s == pytest.approx(0.6)
    assert l.arrival_ns == 600_000_000


def test_flight_is_never_shorter_than_0_30_s_for_any_spin_or_speed():
    for v in (3.0, 7.0, 12.0, 20.0, 40.0):
        for top in (-1.0, -0.5, 0.0, 0.5, 1.0):
            assert leg(v=v, top=top).flight_s >= 0.30 - 1e-12


def test_topspin_arrives_sooner_and_backspin_later_than_flat():
    flat, top, back = leg(v=4.0).flight_s, leg(v=4.0, top=1.0).flight_s, leg(v=4.0, top=-1.0).flight_s
    assert top < flat < back


def test_progress_is_monotone_and_reaches_one_at_arrival():
    l = leg(v=5.0, top=0.7)
    ps = [l.progress(int(i * l.flight_s / 50 * S)) for i in range(51)]
    assert all(b >= a for a, b in zip(ps, ps[1:]))
    assert ps[0] == 0.0 and ps[-1] == pytest.approx(1.0)


def test_time_at_progress_inverts_progress_even_after_the_flight_floor_stretched_it():
    for l in (leg(v=5.0), leg(v=40.0, top=1.0), leg(v=3.0, top=-1.0)):
        for p in (0.0, 0.3, l.p_bounce, 0.9, 1.0):
            assert l.progress(l.time_at_progress(p)) == pytest.approx(p, abs=1e-6)


# --- where the ball is --------------------------------------------------------------------------------------------
def test_an_incoming_ball_starts_at_the_computers_end_and_arrives_over_the_players_end_at_the_aimed_point():
    l = leg(aim=(0.85, 0.5), x_start=-0.3)
    x0, y0, z0 = l.position(0)
    x1, y1, z1 = l.position(l.arrival_ns)
    assert (x0, y0, z0) == pytest.approx((-0.3, physics.STRIKE_Y_M, physics.CPU_Z_M))
    assert (x1, y1, z1) == pytest.approx((physics.x_of_a(0.85), physics.STRIKE_Y_M, physics.HIT_Z_M))


def test_arrival_point_is_the_aim_even_with_sidespin():
    l = leg(aim=(0.85, 0.5), side=0.8, x_start=-0.3)
    assert l.position(l.arrival_ns)[0] == pytest.approx(l.x_end)


def test_sidespin_bends_the_path_midway_but_not_without_spin():
    straight, curved = leg(side=0.0, x_start=-0.5), leg(side=1.0, x_start=-0.5)
    t_mid = round(straight.flight_s * 0.5 * S)
    assert abs(curved.position(t_mid)[0] - straight.position(t_mid)[0]) > 0.05


def test_aim_cells_map_to_lateral_meters_symmetrically():
    left, mid, right = leg(aim=(0.15, 0.5)).x_end, leg(aim=(0.5, 0.5)).x_end, leg(aim=(0.85, 0.5)).x_end
    assert mid == pytest.approx(0.0)
    assert left == pytest.approx(-right)
    assert abs(right) <= physics.HALF_WIDTH_M


def test_a_of_x_inverts_x_of_a():
    for a in (0.0, 0.2, 0.5, 0.77, 1.0):
        assert physics.a_of_x(physics.x_of_a(a)) == pytest.approx(a)
    assert physics.a_of_x(5.0) == 1.0 and physics.a_of_x(-5.0) == 0.0         # clamped to the box


# --- the bounce and the net -----------------------------------------------------------------------------------------
def test_the_ball_never_goes_below_the_table_and_touches_it_exactly_once_at_the_bounce():
    l = leg(v=5.0)
    ys = [y for _, y, _ in track(l, 600)]
    assert min(ys) >= -1e-9
    assert sum(1 for y in ys if y < 0.004) <= 2            # the ball is off the table at every sample but the bounce
    bx, by, bz = l.position(l.bounce_ns)
    assert by == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < bz < physics.NET_Z_M                      # on the receiver's half: the player's, for an incoming ball


def test_a_returned_ball_bounces_on_the_computers_half():
    l = ret()
    _, by, bz = l.position(l.bounce_ns)
    assert by == pytest.approx(0.0, abs=1e-6)
    assert physics.NET_Z_M < bz < physics.TABLE_LEN_M


def test_the_ball_rises_again_after_the_bounce():
    l = leg(v=3.0)
    after = [y for _, y, _ in (l.position(l.bounce_ns + round(k * 0.05 * S)) for k in range(1, 4))]
    assert after[0] > 0.0 and max(after) > 0.15


@pytest.mark.parametrize("v", [2.5, 3.5, 5.0, 7.0, 9.5, 14.0, 40.0])
@pytest.mark.parametrize("aim", [(0.15, 0.15), (0.5, 0.5), (0.85, 0.85)])
@pytest.mark.parametrize("top", [-1.0, 0.0, 1.0])
def test_every_ball_clears_the_net(v, aim, top):
    for l in (leg(v=v, aim=aim, top=top), ret(v=v, aim=aim, top=top)):
        crossing = min((p for p in track(l, 800)), key=lambda p: abs(p[2] - physics.NET_Z_M))
        assert crossing[1] >= physics.NET_H_M + 0.04, crossing


def test_short_balls_bounce_nearer_the_net_and_deep_balls_nearer_the_players_edge():
    short, mid, deep = (leg(aim=(0.5, b)) for b in (0.15, 0.5, 0.85))
    zs = [l.position(l.bounce_ns)[2] for l in (short, mid, deep)]
    assert zs[0] > zs[1] > zs[2] > 0.0


def test_a_returned_ball_starts_where_it_was_hit_and_ends_at_the_computers_paddle():
    l = ret(start=(0.35, 0.18, 0.55), aim=(0.15, 0.5))
    assert l.position(0) == pytest.approx((0.35, 0.18, 0.55))
    x, y, z = l.position(l.arrival_ns)
    assert (x, y, z) == pytest.approx((physics.x_of_a(0.15), physics.STRIKE_Y_M, physics.CPU_Z_M))


def test_the_flight_of_a_return_takes_the_same_time_as_an_incoming_ball_of_that_speed():
    assert ret(v=5.0).flight_s == pytest.approx(leg(v=5.0).flight_s)


# --- faults ---------------------------------------------------------------------------------------------------------
def test_net_fault_ends_early_in_the_net_and_out_fault_lands_beyond_the_table():
    ok, net, out = ret(), ret(fault="net"), ret(fault="out")
    assert ok.terminal == "arrive" and net.terminal == "net" and out.terminal == "out"
    assert net.end_ns < ok.end_ns < out.end_ns
    nx, ny, nz = net.position(net.end_ns)
    assert nz == pytest.approx(physics.NET_Z_M, abs=0.05) and ny < physics.NET_H_M
    ox, oy, oz = out.position(out.end_ns)
    assert oz > physics.TABLE_LEN_M


def test_same_inputs_give_the_same_leg():
    assert leg(v=6.3, top=0.4, side=-0.2) == leg(v=6.3, top=0.4, side=-0.2)


def test_a_ball_with_no_speed_is_refused():
    with pytest.raises(ValueError):
        leg(v=0.0)


# --- a ball that is not hit ---------------------------------------------------------------------------------------
def test_a_ball_that_is_not_hit_flies_on_past_the_player_instead_of_hanging_over_the_sweet_spot():
    l = leg(v=2.5, aim=(0.71, 0.5))
    at = l.position(l.arrival_ns)
    later = l.position(l.arrival_ns + round(0.2 * S))
    assert later[2] < at[2] - 0.2                                   # still travelling towards you ...
    assert later[0] > at[0]                                         # ... and sideways, the way it was going
    assert l.position(l.arrival_ns + round(5.0 * S))[2] == pytest.approx(l.position(l.arrival_ns + round(1.0 * S))[2])  # (for a while)


def test_it_drops_below_the_table_top_after_the_edge_but_never_absurdly_far():
    l = leg(v=2.5)
    ys = [l.position(l.arrival_ns + round(k * 0.05 * S))[1] for k in range(21)]
    assert min(ys) >= -0.6 and ys[-1] < physics.STRIKE_Y_M


# --- when the ball is somewhere ---------------------------------------------------------------------------------------------
def test_time_at_z_says_when_the_ball_is_over_a_given_depth_and_inverts_position():
    l = leg(v=2.5, aim=(0.71, 0.5))
    for z in (2.5, 1.6, 1.0, 0.55, 0.3, 0.05):
        assert l.position(l.time_at_z(z))[2] == pytest.approx(z, abs=1e-3)
    assert l.time_at_z(physics.HIT_Z_M) == l.arrival_ns
    assert l.time_at_z(0.5) < l.time_at_z(0.1)                          # it gets to the more distant depth first


def test_the_same_holds_for_a_ball_going_the_other_way():
    l = ret(v=5.0, start=(0.0, 0.2, 0.4))
    assert l.time_at_z(2.0) > l.time_at_z(1.0) > 0
    assert l.position(l.time_at_z(2.0))[2] == pytest.approx(2.0, abs=1e-3)
