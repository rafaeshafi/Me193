"""The pose model: how the hand's readings are filtered and where the hand will be when the frame is seen.

Both parts are learned from the player's own recorded tracks (pingpong/posetrain.py).  The defaults are what the game
used before: a One-Euro filter at fixed settings and an extrapolation by the hand's recent speed.
"""

import math

import pytest

from pingpong import posemodel

S = 1_000_000_000
DT = 1 / 30


class P:
    def __init__(self, t_s, u, v=0.0, conf=0.9):
        self.t_scene_ns, self.u, self.v, self.conf = round(t_s * S), u, v, conf


def track(fn, seconds=2.0, hz=30.0):
    return [P(k / hz, *fn(k / hz)) for k in range(int(seconds * hz))]


# --- the filter ---------------------------------------------------------------------------------------------------------
def test_the_default_filter_is_the_one_the_game_used_before():
    from pingpong.oneeuro import OneEuro2D

    f, g = posemodel.PoseFilter(), OneEuro2D()
    for k in range(60):
        xy = (math.sin(k / 9.0), 0.2 * math.cos(k / 7.0))
        out = f(xy, k * DT)
        assert out == pytest.approx(g(xy, k * DT))


def test_a_single_frame_glitch_is_dropped_but_a_real_jump_that_lasts_is_followed():
    f = posemodel.PoseFilter()
    for k in range(30):
        f((0.5, 0.0), k * DT)
    spike = f((3.5, 0.0), 30 * DT)                                         # one frame 3 shoulder widths away: a landmark flip
    assert spike[0] == pytest.approx(0.5, abs=0.05)
    after = f((0.5, 0.0), 31 * DT)
    assert after[0] == pytest.approx(0.5, abs=0.05)
    g = posemodel.PoseFilter()
    for k in range(30):
        g((0.5, 0.0), k * DT)
    out = [g((3.5, 0.0), (30 + j) * DT)[0] for j in range(8)]                # it stays there: the hand really is there
    assert out[-1] > 2.0 and out[0] < 1.0


def test_the_filter_parameters_survive_json():
    params = posemodel.FilterParams(min_cutoff=0.8, beta=12.0, d_cutoff=1.7, gate_speed=33.0)
    assert posemodel.FilterParams.from_json(params.to_json()) == params


# --- the predictor ---------------------------------------------------------------------------------------------------------
def linear_model(gain=1.0):
    """A predictor that extrapolates by the fitted slope: weights that put `gain` x the lead on the velocity."""
    k = posemodel.K
    # slope through the last two samples: (x_t - x_{t-1}) / STEP  ->  d_1 = x_{t-1} - x_t = -slope * STEP
    a = tuple(-gain * posemodel.H0 / posemodel.STEP_S if i == 0 else 0.0 for i in range(k))
    b = tuple(-gain / posemodel.STEP_S if i == 0 else 0.0 for i in range(k))
    return posemodel.HandPredictor(a_u=a, b_u=b, a_v=a, b_v=b, alpha=1.0)


def test_a_predictor_with_slope_weights_extrapolates_a_steady_hand():
    ps = track(lambda t: (1.5 * t, 0.0), seconds=1.5)
    model = linear_model()
    for lead in (0.10, 0.15, 0.20):
        u, v = model.predict(ps, lead)
        assert u == pytest.approx(ps[-1].u + 1.5 * lead, abs=1e-6) and v == pytest.approx(0.0, abs=1e-9)


def test_the_weights_depend_on_the_lead_linearly():
    model = linear_model()
    ps = track(lambda t: (2.0 * t, -1.0 * t), seconds=1.5)
    a, b = model.predict(ps, 0.10), model.predict(ps, 0.20)
    assert b[0] - ps[-1].u == pytest.approx(2 * (a[0] - ps[-1].u), rel=1e-6) and b[1] - ps[-1].v == pytest.approx(
        2 * (a[1] - ps[-1].v), rel=1e-6)


def test_a_hand_at_rest_is_predicted_where_it_is_whatever_the_weights():
    ps = track(lambda t: (0.4, -0.2), seconds=1.0)
    u, v = linear_model(gain=5.0).predict(ps, 0.2)
    assert (u, v) == pytest.approx((0.4, -0.2))


def test_a_track_with_a_hole_or_too_short_gives_no_prediction_so_the_caller_can_fall_back():
    model = linear_model()
    ps = track(lambda t: (t, 0.0), seconds=1.0)
    assert model.predict(ps[-3:], 0.15) is None                                 # not enough history
    holed = ps[:12] + ps[20:]                                                    # a quarter second with no reading
    assert model.predict(holed, 0.15) is None
    low = [P(p.t_scene_ns / S, p.u, p.v, conf=0.2) for p in ps]
    assert model.predict(low, 0.15) is None


def test_the_smoother_in_front_of_the_weights_averages_the_noise_of_the_newest_readings():
    import random

    rnd = random.Random(2)
    ps = [P(k * DT, 0.5 + rnd.gauss(0, 0.03), 0.0) for k in range(40)]
    smooth = posemodel.HandPredictor(a_u=(0.0,) * 6, b_u=(0.0,) * 6, a_v=(0.0,) * 6, b_v=(0.0,) * 6, alpha=0.5)
    outs = [smooth.predict(ps[: n + 1], 0.15)[0] for n in range(15, 40)]
    raw = [p.u for p in ps[15:]]

    def sd(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    assert sd(outs) < 0.8 * sd(raw)


def test_a_pose_model_survives_json_and_a_player_folder(tmp_path):
    model = posemodel.PoseModel(filter=posemodel.FilterParams(min_cutoff=0.9), predictor=linear_model(0.7),
                                meta={"trained_on": 3, "note": "x"})
    assert posemodel.PoseModel.from_json(model.to_json()) == model
    posemodel.save_for("Rafae Shafi", model, root=tmp_path)
    assert posemodel.load_for("Rafae Shafi", root=tmp_path) == model
    assert posemodel.load_for("someone-else", root=tmp_path) is None
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "pose_model.json").write_text("{not json")
    assert posemodel.load_for("broken", root=tmp_path) is None                  # a damaged file must not stop the game


def test_no_model_means_the_game_keeps_its_old_behaviour():
    assert posemodel.PoseModel.default().predictor is None
    assert posemodel.PoseModel.default().filter == posemodel.FilterParams()


def test_the_pose_model_the_player_runs_is_part_of_the_trained_model_and_defaults_to_the_light_one(tmp_path):
    assert posemodel.PoseModel.default().landmarker == "lite"
    model = posemodel.PoseModel(landmarker="full", meta={"why": "less noise"})
    assert model.to_json()["landmarker"] == "full"
    assert posemodel.PoseModel.from_json(model.to_json()) == model
    posemodel.save_for("rafae", model, root=tmp_path)
    assert posemodel.load_for("rafae", root=tmp_path).landmarker == "full"
    assert posemodel.PoseModel.from_json({"filter": {}}).landmarker == "lite"           # a file written before the choice existed
    assert posemodel.PoseModel.from_json({"landmarker": "heavy"}).landmarker == "lite"  # a name this game does not know
