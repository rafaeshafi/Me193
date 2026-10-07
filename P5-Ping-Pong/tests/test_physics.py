"""Analytic ball legs: exact arrival time and point, stylised spin (no integrator)."""

import pytest

from pingpong import physics

S = 1_000_000_000


def leg(v=5.0, top=0.0, side=0.0, x_start=0.0, aim=(0.5, 0.5), fault=None):
    return physics.plan_leg(t0_ns=0, v=v, x_start=x_start, aim_ab=aim, topspin=top, sidespin=side, fault=fault)


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


def test_bounce_is_at_55_percent_of_the_flat_flight():
    l = leg(v=5.0)
    assert l.p_bounce == 0.55
    assert l.progress(int(0.55 * 0.6 * S)) == pytest.approx(0.55)


def test_progress_is_monotone_and_reaches_one_at_arrival():
    l = leg(v=5.0, top=0.7)
    ps = [l.progress(int(i * l.flight_s / 50 * S)) for i in range(51)]
    assert all(b >= a for a, b in zip(ps, ps[1:]))
    assert ps[0] == 0.0 and ps[-1] == pytest.approx(1.0)


def test_arrival_point_is_the_aim_even_with_sidespin():
    l = leg(aim=(0.85, 0.5), side=0.8, x_start=-0.3)
    x, p, h = l.position(l.arrival_ns)
    assert p == pytest.approx(1.0)
    assert x == pytest.approx(l.x_end)


def test_sidespin_bends_the_path_midway_but_not_without_spin():
    straight = leg(aim=(0.5, 0.5), side=0.0, x_start=-0.5)
    curved = leg(aim=(0.5, 0.5), side=1.0, x_start=-0.5)
    t_mid = int(straight.flight_s * 0.5 * S)
    x_straight = straight.position(t_mid)[0]
    x_curved = curved.position(int(curved.flight_s * 0.5 * S))[0]
    assert abs(x_curved - x_straight) > 0.05


def test_aim_cells_map_to_lateral_meters_symmetrically():
    left, mid, right = leg(aim=(0.15, 0.5)).x_end, leg(aim=(0.5, 0.5)).x_end, leg(aim=(0.85, 0.5)).x_end
    assert mid == pytest.approx(0.0)
    assert left == pytest.approx(-right)
    assert abs(right) <= physics.HALF_WIDTH_M


def test_net_fault_ends_early_and_out_fault_lands_long():
    ok, net, out = leg(), leg(fault="net"), leg(fault="out")
    assert ok.terminal == "arrive" and net.terminal == "net" and out.terminal == "out"
    assert net.end_ns < ok.end_ns < out.end_ns
    assert out.position(out.end_ns)[1] > 1.0


def test_same_inputs_give_the_same_leg():
    assert leg(v=6.3, top=0.4, side=-0.2) == leg(v=6.3, top=0.4, side=-0.2)


def test_apex_is_capped_so_fast_balls_do_not_imply_absurd_arcs():
    assert leg(v=3.0).apex_m <= 0.30
    assert leg(v=12.0).apex_m <= leg(v=3.0).apex_m
