"""Training the pose model on recorded tracks: a predictor that beats extrapolation on tracks it never saw, a filter tuned
against a zero-phase smoothing of the raw readings, and the plumbing that reads a player's recorded sessions.

The tracks here are synthetic (smooth reaching moves between random targets, pauses, and fast swings, plus noise,
dropouts and landmark glitches) so the truth is known; tests/data/ carries nothing of the player's."""

import json
import math
import random

import numpy as np
import pytest

from pingpong import posemodel, posetrain

HZ = 30.0


def truth(seed, seconds=70.0, kind="rally"):
    """The hand's true (u, v) on a fine grid.

    "rally": swings back and forth at a rhythm that drifts (0.6-1.5 Hz) with a drifting reach (0.4-1.2 shoulder widths),
    now and then a pause, like a rally; "reach": min-jerk moves between random targets, whose timing nobody can predict."""
    rnd = random.Random(seed)
    fine = 240
    n = int(seconds * fine)
    if kind == "reach":
        us, vs, x, y = [], [], 0.0, 0.0
        while len(us) < n:
            dur = rnd.choice([0.15, 0.2, 0.3, 0.5, 0.8, 1.2])
            tx, ty = rnd.uniform(-1.2, 1.2), rnd.uniform(-0.8, 0.8)
            m_n = int(dur * fine)
            for i in range(m_n):
                s_ = i / m_n
                m = 10 * s_ ** 3 - 15 * s_ ** 4 + 6 * s_ ** 5
                us.append(x + (tx - x) * m)
                vs.append(y + (ty - y) * m)
            x, y = tx, ty
            us += [x] * int(rnd.choice([0.0, 0.2, 0.5, 1.0]) * fine)
            vs += [y] * len(us[len(vs):])
        return np.array(us[:n]), np.array(vs[:n]), fine
    phase, f, amp, us, vs = rnd.uniform(0, 6.28), 1.0, 0.8, [], []
    pause = 0
    for i in range(n):
        f = min(1.5, max(0.6, f + rnd.gauss(0, 0.0015)))
        amp = min(1.2, max(0.4, amp + rnd.gauss(0, 0.0012)))
        phase += 2 * math.pi * f / fine
        if pause == 0 and rnd.random() < 0.0004:
            pause = int(rnd.uniform(0.3, 1.0) * fine)
        if pause:
            pause -= 1
            amp *= 0.998
        us.append(amp * math.sin(phase))
        vs.append(0.35 * amp * math.sin(phase * 0.5 + 1.0))
    return np.array(us), np.array(vs), fine


def raw_track(seed, noise=0.02, drop=0.0, glitches=0, seconds=70.0, kind="rally"):
    """(t, u, v) samples at ~30 Hz with jitter in time, noise, dropped frames and one-frame landmark flips."""
    tu, tv, fine = truth(seed, seconds, kind)
    rnd = random.Random(seed + 100)
    t, u, v = [], [], []
    k = 0
    while True:
        ts = k / HZ + rnd.gauss(0, 0.002)
        i = int(ts * fine)
        if i >= len(tu):
            break
        k += 1
        if rnd.random() < drop:
            continue
        t.append(ts)
        u.append(tu[i] + rnd.gauss(0, noise))
        v.append(tv[i] + rnd.gauss(0, noise))
    for _ in range(glitches):
        j = rnd.randrange(10, len(t) - 10)
        u[j] += rnd.choice([-1, 1]) * rnd.uniform(1.0, 2.5)
    return posetrain.Track(t=np.array(t), u=np.array(u), v=np.array(v), ru=np.array(u), rv=np.array(v), name=f"s{seed}")


def filtered(track, params=None):
    f = posemodel.PoseFilter(params)
    out = [f((a, b), ts) for ts, a, b in zip(track.t, track.ru, track.rv)]
    return posetrain.Track(t=track.t, u=np.array([o[0] for o in out]), v=np.array([o[1] for o in out]), ru=track.ru,
                           rv=track.rv, name=track.name)


# --- the grid and the teacher ---------------------------------------------------------------------------------------------
def test_the_grid_resamples_onto_a_uniform_clock_and_marks_holes_it_will_not_bridge():
    t = np.array([0.0, 0.034, 0.066, 0.1, 0.4, 0.434])
    g, x = posetrain.on_grid(t, t * 2.0)
    assert g[1] - g[0] == pytest.approx(posemodel.STEP_S)
    assert x[1] == pytest.approx(2 * g[1], abs=1e-9)
    assert np.isnan(x[(g > 0.13) & (g < 0.37)]).all() and np.isfinite(x[g < 0.1]).all()


def test_the_teacher_is_smooth_and_follows_a_real_move_without_lag_and_drops_a_glitch():
    t = np.arange(0, 6, 1 / HZ)
    clean = np.where(t < 3.0, 0.0, 1.0) * 1.0
    clean = np.convolve(clean, np.ones(5) / 5, mode="same")                    # a move over a sixth of a second
    rnd = np.random.default_rng(1)
    noisy = clean + rnd.normal(0, 0.03, len(t))
    noisy[40] += 2.0                                                           # a landmark flip
    teach = posetrain.teacher(noisy)
    assert abs(teach[40] - clean[40]) < 0.08                                   # the flip is gone
    assert np.sqrt(np.mean((teach - clean)[5:-5] ** 2)) < 0.04                 # and the track is close to the truth
    assert np.argmax(np.abs(np.diff(teach))) in range(88, 93)                  # the step is where it was: no lag


# --- the predictor -----------------------------------------------------------------------------------------------------------
def test_a_predictor_trained_on_some_tracks_is_better_than_holding_and_as_good_as_extrapolating_with_far_less_shimmer():
    # (extrapolating the raw speed is as accurate as it gets here but shivers three times as much as the track it is
    # added to; the trained predictor is held to 1.5 times)
    train = [filtered(raw_track(s)) for s in (1, 2, 3, 4)]
    test = [filtered(raw_track(s)) for s in (11, 12)]
    model, report = posetrain.fit_predictor(train)
    scores = posetrain.evaluate(model, test, lead_s=5 * posemodel.STEP_S)
    assert scores["rmse_model"] < 0.87 * scores["rmse_hold"]
    assert scores["rmse_model"] <= 1.12 * scores["rmse_extrapolation"]
    assert scores["step_model"] < 0.65 * scores["step_extrapolation"]
    assert report["lambda"] in posetrain.LAMBDAS and report["sessions"] == 4


def test_where_the_future_cannot_be_predicted_the_model_is_not_worse_than_extrapolating():
    train = [filtered(raw_track(s, kind="reach")) for s in (1, 2, 3)]
    test = [filtered(raw_track(s, kind="reach")) for s in (11, 12)]
    model, _ = posetrain.fit_predictor(train)
    scores = posetrain.evaluate(model, test, lead_s=5 * posemodel.STEP_S)
    assert scores["rmse_model"] <= 1.05 * scores["rmse_extrapolation"]


def test_the_predictor_is_not_much_rougher_than_the_filtered_track_it_is_added_to():
    train = [filtered(raw_track(s)) for s in (1, 2, 3)]
    test = [filtered(raw_track(s)) for s in (21,)]
    model, _ = posetrain.fit_predictor(train)
    scores = posetrain.evaluate(model, test, lead_s=5 * posemodel.STEP_S)
    assert scores["step_model"] < 2.0 * scores["step_track"]                    # smooth enough not to shimmer on screen
    assert scores["step_model"] < scores["step_extrapolation"] * 1.2 or scores["step_model"] < 0.03


def test_training_uses_the_same_features_the_game_will_so_there_is_no_train_serve_skew():
    track = filtered(raw_track(5, seconds=20.0))
    model, _ = posetrain.fit_predictor([track, filtered(raw_track(6, seconds=20.0))])

    class P:
        def __init__(self, t, u, v):
            self.t_scene_ns, self.u, self.v, self.conf = round(t * 1e9), u, v, 0.9

    g, ug = posetrain.on_grid(track.t, track.u)
    _, vg = posetrain.on_grid(track.t, track.v)
    i = next(k for k in range(60, len(g) - 10) if np.isfinite(ug[k - 12:k + 8]).all())
    poses = [P(g[k], ug[k], vg[k]) for k in range(i - 40, i + 1)]
    offline = posetrain.predict_on_grid(model, ug, vg, i, lead_s=0.15)
    online = model.predict(poses, 0.15)
    assert online == pytest.approx(offline, abs=1e-9)


def test_the_predictor_ships_only_when_it_clearly_beats_holding_the_hand_and_does_not_shimmer():
    good = {"rmse_hold": 0.30, "rmse_model": 0.25, "step_track": 0.010, "step_model": 0.013}
    ok, why = posetrain.worth_shipping(good)
    assert ok and "17%" in why
    ok, why = posetrain.worth_shipping(dict(good, rmse_model=0.285))                 # 5% better: not worth the risk
    assert not ok and "5%" in why and "10%" in why
    ok, why = posetrain.worth_shipping(dict(good, step_model=0.020))                 # 2x the track's shimmer
    assert not ok and "shimmer" in why
    ok, why = posetrain.worth_shipping(dict(good, rmse_model=0.31))                  # worse than holding still
    assert not ok and "does not beat" in why and "-" not in why.split("beat")[0]
    ok, why = posetrain.worth_shipping(dict(good, rmse_model=0.2995))                # the real sessions: no better at all
    assert not ok and "does not beat" in why


def test_a_model_trained_on_nothing_useful_is_refused_not_shipped():
    with pytest.raises(ValueError):
        posetrain.fit_predictor([])
    short = posetrain.Track(t=np.arange(0, 2, 1 / HZ), u=np.zeros(60), v=np.zeros(60), ru=None, rv=None, name="short")
    with pytest.raises(ValueError):
        posetrain.fit_predictor([short])


# --- the filter --------------------------------------------------------------------------------------------------------------
def test_a_tuned_filter_follows_the_teacher_better_than_the_default_and_never_worse():
    tracks = [raw_track(s, noise=0.04, glitches=3) for s in (1, 2, 3)]
    params, report = posetrain.tune_filter(tracks)
    assert report["rmse_tuned"] <= report["rmse_default"] + 1e-12
    assert report["rmse_tuned"] < 0.97 * report["rmse_default"]
    assert isinstance(params, posemodel.FilterParams) and params.gate_speed == posemodel.FilterParams().gate_speed


def test_tuning_needs_the_unfiltered_readings():
    track = raw_track(1)
    with pytest.raises(ValueError):
        posetrain.tune_filter([posetrain.Track(t=track.t, u=track.u, v=track.v, ru=None, rv=None, name="x")])


# --- reading a player's sessions ------------------------------------------------------------------------------------------------
def write_session(root, name, track, source="live", hand="right", with_raw=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"source": source, "player": "rafae"}))
    with (d / "pose.jsonl").open("w") as f:
        for ts, u, v, ru, rv in zip(track.t, track.u, track.v, track.ru, track.rv):
            row = {"t": int(ts * 1e9) + 10**12, "u": float(u), "v": float(v), "c": 0.9, "h": hand}
            if with_raw:
                row["r"] = [float(ru), float(rv)]
            f.write(json.dumps(row) + "\n")


def test_a_players_live_sessions_become_tracks_and_other_sources_and_other_players_do_not(tmp_path):
    t1, t2 = filtered(raw_track(1, seconds=20.0)), filtered(raw_track(2, seconds=20.0))
    write_session(tmp_path, "20261007-100000-rafae", t1)
    write_session(tmp_path, "20261007-110000-rafae", t2, with_raw=False)
    write_session(tmp_path, "20261007-120000-rafae", t1, source="replay")
    write_session(tmp_path, "20261007-130000-maya", t1)
    tracks = posetrain.load_tracks("rafae", root=tmp_path)
    assert [x.name for x in tracks] == ["20261007-100000-rafae", "20261007-110000-rafae"]
    assert tracks[0].ru is not None and tracks[1].ru is None
    assert tracks[0].t[0] == pytest.approx(0.0, abs=1e-6) or tracks[0].t[0] > 0                 # seconds, any origin
    assert tracks[0].u == pytest.approx(t1.u, abs=1e-9)


def test_short_sessions_are_not_training_data(tmp_path):
    write_session(tmp_path, "20261007-100000-rafae", filtered(raw_track(1, seconds=4.0)))
    assert posetrain.load_tracks("rafae", root=tmp_path) == []
