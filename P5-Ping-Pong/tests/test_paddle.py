import pytest

from pingpong.paddle import ReachBox


def test_corners_map_to_the_unit_square_and_back():
    box = ReachBox(u_min=-0.8, u_max=1.2, v_min=-0.4, v_max=0.6)
    assert box.to_ab(-0.8, -0.4) == pytest.approx((0.0, 0.0))
    assert box.to_ab(1.2, 0.6) == pytest.approx((1.0, 1.0))
    assert box.to_uv(0.5, 0.5) == pytest.approx((0.2, 0.1))
    a, b = box.to_ab(*box.to_uv(0.15, 0.85))
    assert (a, b) == pytest.approx((0.15, 0.85))


def test_fit_from_four_corner_samples_uses_the_extremes_with_a_small_margin():
    corners = [(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)]
    box = ReachBox.fit(corners, margin=0.0)
    assert (box.u_min, box.u_max, box.v_min, box.v_max) == (-1.0, 1.0, -0.5, 0.5)
    padded = ReachBox.fit(corners, margin=0.1)
    assert padded.u_min == pytest.approx(-1.1) and padded.v_max == pytest.approx(0.55)


def test_degenerate_boxes_are_rejected():
    with pytest.raises(ValueError):
        ReachBox(u_min=0.0, u_max=0.0, v_min=0.0, v_max=1.0)
    with pytest.raises(ValueError):
        ReachBox.fit([(0, 0)], margin=0.0)
