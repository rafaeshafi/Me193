"""Training the pose model on the player's recorded tracks (`./pp train_pose`).

What is learned, and against what:

  * PREDICTOR.  Where will the hand be `lead` seconds from the newest reading?  Learned by ridge regression on the
    player's own recorded tracks: the features are the last 0.2 s of the (lightly smoothed) track relative to its
    present, the target is where the track really went.  That needs no ground truth beyond the track itself.  The ridge
    strength is chosen by leave-one-session-out cross-validation: the most accurate predictor whose paddle shimmers at
    most SHIMMER_CAP times as much as the filtered track (a predictor that differentiates noise is more accurate and
    would make the paddle shiver).
  * FILTER.  The One-Euro settings are tuned so the causal filter's output follows a ZERO-PHASE smoothing (Hampel,
    then a Gaussian, over past and future) of the player's raw readings: the smoothing a filter would give if it could
    see the future, with the glitches removed.  Needs the unfiltered readings (sessions record them now).

Everything here is plain numpy and deterministic; the game only ever loads the result (posemodel.py).
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pingpong import posemodel, profile
from pingpong.posemodel import H0, K, MAX_GAP_S, STEP_S, WARM, FilterParams, GlitchGate, HandPredictor, PoseFilter

LAMBDAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)       # ridge strengths tried
LEAD_STEPS = (3, 4, 5, 6, 7)                         # grid steps trained on: 0.10 - 0.23 s
EVAL_STEPS = 5                                       # ... and reported at 0.17 s
ALPHA = 0.5                                          # the predictor's smoother
SHIMMER_CAP = 1.5                                    # the predicted paddle may shimmer this many times the filtered track
MIN_GAIN = 0.10                                      # the predictor ships only if it cuts the held-out error this much
MIN_TRACK_S = 10.0
MIN_WINDOWS = 300
CALM = 0.03                                          # shoulder widths per frame: below this the hand is not moving fast
OLD_GAIN = 0.8                                       # the extrapolation the game used before the model (latency.GAIN)
WINDOW = K + WARM + 1

# The filter settings tried.  The first search stopped at the top of every range (4.5, 20, 2.5): matching the zero-phase
# reference alone is best served by letting everything through.  Now a setting also has to jitter a still hand no more than
# the default does, and the ranges reach well past where the best one sits (a d_cutoff above ~10 only lets the noise in the
# speed open the filter).
CUTOFFS = (0.15, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0)
BETAS = (0.0, 1.0, 2.5, 5.0, 7.0, 10.0, 14.0, 20.0, 40.0, 80.0, 160.0)
D_CUTOFFS = (0.7, 1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0)
JITTER_CAP = 1.0                                     # a tuned filter may jitter a still hand this many times as much as the default
STILL_STEP = 0.01                                    # shoulder widths per frame (0.3 a second): below it the hand counts as still
CHUNK = 256                                          # filter settings run in parallel per pass (keeps long sessions in memory)


@dataclass
class Track:
    t: np.ndarray                  # seconds
    u: np.ndarray                  # the filtered readings, as recorded
    v: np.ndarray
    ru: Optional[np.ndarray] = None    # the readings before the filter (None for sessions recorded before they were kept)
    rv: Optional[np.ndarray] = None
    name: str = ""


# --- reading sessions ----------------------------------------------------------------------------------------------------------
def load_tracks(player, root=None, min_s=MIN_TRACK_S):
    """The player's recorded LIVE sessions as tracks (replays and other players' sessions are not training data)."""
    from pingpong import recorder

    root = root or recorder.default_root()
    suffix = "-" + profile.slug(player)
    tracks = []
    for d in sorted(p for p in root.glob("*" + suffix) if p.is_dir()):
        try:
            if json.loads((d / "session.json").read_text()).get("source") != "live":
                continue
            rows = [json.loads(line) for line in (d / "pose.jsonl").read_text().splitlines() if line]
        except (OSError, ValueError):
            continue
        if len(rows) < 30 or (rows[-1]["t"] - rows[0]["t"]) / 1e9 < min_s:
            continue
        t = np.array([r["t"] for r in rows], dtype=float)
        t = (t - t[0]) / 1e9
        raw = all("r" in r for r in rows)
        tracks.append(Track(t=t, u=np.array([r["u"] for r in rows]), v=np.array([r["v"] for r in rows]),
                            ru=np.array([r["r"][0] for r in rows]) if raw else None,
                            rv=np.array([r["r"][1] for r in rows]) if raw else None, name=d.name))
    return tracks


# --- the grid and the teacher ---------------------------------------------------------------------------------------------------
def on_grid(t, x, step=STEP_S, max_gap=MAX_GAP_S):
    """The track on a uniform clock; NaN where the readings around a grid time are further apart than max_gap."""
    g = np.arange(t[0], t[-1], step)
    xi = np.interp(g, t, x)
    idx = np.searchsorted(t, g)
    hi, lo = np.clip(idx, 0, len(t) - 1), np.clip(idx - 1, 0, len(t) - 1)
    xi[((t[hi] - t[lo]) > max_gap) & (np.abs(t[hi] - g) > 1e-9)] = np.nan
    return g, xi


def _segments(x):
    """(start, stop) of every run of finite values."""
    finite = np.isfinite(x)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], finite.view(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def teacher(x, sigma=1.5, window=7, k=3.5):
    """A zero-phase smoothing of a uniformly sampled track: Hampel (a reading far from its neighbours' median is
    replaced by it) then a Gaussian over past and future.  What a filter would give if it could see the future."""
    out = np.full_like(x, np.nan, dtype=float)
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    half = window // 2
    for a, b in _segments(x):
        seg = x[a:b].astype(float)
        if len(seg) < 3:
            out[a:b] = seg
            continue
        padded = np.pad(seg, half, mode="edge")
        med = np.array([np.median(padded[i:i + window]) for i in range(len(seg))])
        mad = np.array([np.median(np.abs(padded[i:i + window] - med[i])) for i in range(len(seg))]) * 1.4826
        seg = np.where(np.abs(seg - med) > k * np.maximum(mad, 0.005), med, seg)
        out[a:b] = np.convolve(np.pad(seg, radius, mode="edge"), kernel, mode="valid")
    return out


# --- the predictor ----------------------------------------------------------------------------------------------------------------
def _windows(xg, alpha=ALPHA):
    """For every grid index with a finite window behind it: (index, smoothed present, d_1..d_K), like HandPredictor does."""
    n = len(xg)
    if n < WINDOW:
        return np.array([], dtype=int), np.zeros(0), np.zeros((0, K))
    idx = np.array([i for i in range(WINDOW - 1, n) if np.isfinite(xg[i - WINDOW + 1:i + 1]).all()], dtype=int)
    if not len(idx):
        return idx, np.zeros(0), np.zeros((0, K))
    w = np.stack([xg[i - WINDOW + 1:i + 1] for i in idx])
    s = w[:, 0].copy()
    cols = [s.copy()]
    for c in range(1, WINDOW):
        s = s + alpha * (w[:, c] - s)
        cols.append(s.copy())
    sm = np.stack(cols, axis=1)                                # (rows, WINDOW): newest last
    present = sm[:, -1]
    d = np.stack([sm[:, -1 - k] - present for k in range(1, K + 1)], axis=1)
    return idx, present, d


def _design(track_grids, steps, alpha=ALPHA):
    """Per axis: (X, y) with X = [d, (h - H0) d] and y = future - smoothed present, for every lead in steps."""
    out = []
    for xg in track_grids:
        idx, present, d = _windows(xg, alpha)
        rows, ys = [], []
        for h_steps in steps:
            ok = idx + h_steps < len(xg)
            fut = np.full(len(idx), np.nan)
            fut[ok] = xg[idx[ok] + h_steps]
            good = np.isfinite(fut)
            h = h_steps * STEP_S - H0
            rows.append(np.hstack([d[good], h * d[good]]))
            ys.append(fut[good] - present[good])
        out.append((np.vstack(rows) if rows else np.zeros((0, 2 * K)), np.concatenate(ys) if ys else np.zeros(0)))
    return out


def _ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ y)


def _fit_axis(grids, lam, alpha):
    parts = [p for p in _design(grids, LEAD_STEPS, alpha) if len(p[1])]
    if not parts:
        raise ValueError("no usable stretch of track (it needs 0.4 s of unbroken readings)")
    X, y = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    w = _ridge(X, y, lam)
    return tuple(float(a) for a in w[:K]), tuple(float(b) for b in w[K:])


def _grids(tracks):
    gu, gv = [], []
    for tr in tracks:
        _, u = on_grid(tr.t, tr.u)
        _, v = on_grid(tr.t, tr.v)
        gu.append(u)
        gv.append(v)
    return gu, gv


def _usable(tracks):
    keep = []
    for tr in tracks:
        if tr.t[-1] - tr.t[0] >= MIN_TRACK_S and len(_windows(on_grid(tr.t, tr.u)[1])[0]) >= MIN_WINDOWS:
            keep.append(tr)
    return keep


def _fit(tracks, lam, alpha):
    gu, gv = _grids(tracks)
    a_u, b_u = _fit_axis(gu, lam, alpha)
    a_v, b_v = _fit_axis(gv, lam, alpha)
    return HandPredictor(a_u=a_u, b_u=b_u, a_v=a_v, b_v=b_v, alpha=alpha)


def predict_on_grid(model, ug, vg, i, lead_s):
    """What the model says at grid index i: the same arithmetic the game runs, from the grid's window."""
    lead = max(0.0, min(posemodel.MAX_LEAD_S, lead_s))
    return (model._axis(list(ug[i - WINDOW + 1:i + 1]), model.a_u, model.b_u, lead),
            model._axis(list(vg[i - WINDOW + 1:i + 1]), model.a_v, model.b_v, lead))


def evaluate(model, tracks, lead_s=EVAL_STEPS * STEP_S):
    """How well `model` predicts where the hand goes `lead_s` ahead on `tracks` it was not trained on, against holding the
    hand where it is and against extrapolating its recent speed (what the game did before): position error in shoulder
    widths, and the shimmer (frame-to-frame step on calm stretches) of the track the player would see."""
    h = max(1, round(lead_s / STEP_S))
    err = {"hold": [], "extrapolation": [], "model": []}
    steps = {"track": [], "extrapolation": [], "model": []}
    for tr in tracks:
        for xg, ag, bg in ((on_grid(tr.t, tr.u)[1], model.a_u, model.b_u), (on_grid(tr.t, tr.v)[1], model.a_v, model.b_v)):
            idx, present, d = _windows(xg, model.alpha)
            ok = idx + h < len(xg)
            idx, present, d = idx[ok], present[ok], d[ok]
            fut = xg[idx + h]
            good = np.isfinite(fut)
            idx, present, d, fut = idx[good], present[good], d[good], fut[good]
            if not len(idx):
                continue
            now = xg[idx]
            lead = h * STEP_S
            weights = np.array(ag) + (lead - H0) * np.array(bg)
            shown = present + d @ weights
            kk = np.arange(1, 5) * STEP_S                      # the old rule: a least-squares slope over 0.13 s, damped
            back = np.stack([xg[np.clip(idx - k, 0, None)] - now for k in range(1, 5)], axis=1)
            slope = -(back @ kk) / (kk @ kk)
            extrap = now + OLD_GAIN * slope * lead
            err["hold"].append(fut - now)
            err["extrapolation"].append(fut - extrap)
            err["model"].append(fut - shown)
            consecutive = np.diff(idx) == 1
            calm = np.abs(np.diff(now)) < CALM
            for name, series in (("track", now), ("extrapolation", extrap), ("model", shown)):
                steps[name].append(np.diff(series)[consecutive & calm])
    if not err["model"]:
        raise ValueError("nothing to evaluate on")
    rms = lambda parts: float(np.sqrt(np.mean(np.concatenate(parts) ** 2))) if sum(len(p) for p in parts) else float("nan")  # noqa: E731
    return {"lead_s": h * STEP_S, "n": int(sum(len(e) for e in err["model"]) // 1),
            "rmse_hold": rms(err["hold"]), "rmse_extrapolation": rms(err["extrapolation"]), "rmse_model": rms(err["model"]),
            "step_track": rms(steps["track"]), "step_extrapolation": rms(steps["extrapolation"]),
            "step_model": rms(steps["model"])}


def worth_shipping(cv):
    """-> (bool, why): is a cross-validated predictor good enough for the game?  It has to cut the error of simply holding
    the hand where it was by MIN_GAIN, and not make the paddle shimmer more than SHIMMER_CAP times the filtered track."""
    gain = 1.0 - cv["rmse_model"] / cv["rmse_hold"]
    shimmer = cv["step_model"] / cv["step_track"] if cv["step_track"] > 0 else float("inf")
    if not shimmer <= SHIMMER_CAP:
        return False, f"it would make the paddle shimmer {shimmer:.1f}x the filtered track (the limit is {SHIMMER_CAP}x)"
    if gain < 0.005:
        return False, "it does not beat holding the hand where it was"
    if gain < MIN_GAIN:
        return False, f"it cuts the error by {gain:.0%} against holding the hand where it was; it has to cut it by {MIN_GAIN:.0%}"
    return True, f"it cuts the error by {gain:.0%} against holding the hand where it was, with {shimmer:.1f}x the track's shimmer"


def _folds(tracks):
    """Leave-one-session-out; a single session is cut in two by time."""
    if len(tracks) >= 2:
        return [([t for j, t in enumerate(tracks) if j != i], [tracks[i]]) for i in range(len(tracks))]
    tr = tracks[0]
    mid = len(tr.t) // 2

    def cut(a, b):
        return Track(t=tr.t[a:b], u=tr.u[a:b], v=tr.v[a:b], name=tr.name)

    return [([cut(0, mid)], [cut(mid, None)]), ([cut(mid, None)], [cut(0, mid)])]


def fit_predictor(tracks, lambdas=LAMBDAS, alpha=ALPHA):
    """-> (HandPredictor, report).  The ridge strength is the one with the best held-out error + shimmer."""
    tracks = _usable(tracks)
    if not tracks:
        raise ValueError(f"not enough recorded hand tracking to train on (need a session of {MIN_TRACK_S:.0f} s or more)")
    folds = _folds(tracks)
    scores = {}
    for lam in lambdas:
        fold_scores = []
        for train, held in folds:
            try:
                fold_scores.append(evaluate(_fit(train, lam, alpha), held))
            except ValueError:
                continue
        if fold_scores:
            scores[lam] = {k: float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}
    if not scores:
        raise ValueError("not enough unbroken tracking to cross-validate on")
    smooth_enough = [lam for lam in scores if scores[lam]["step_model"] <= SHIMMER_CAP * scores[lam]["step_track"]]
    best = min(smooth_enough, key=lambda lam: scores[lam]["rmse_model"]) if smooth_enough else \
        min(scores, key=lambda lam: scores[lam]["step_model"])             # nothing is smooth enough: take the smoothest
    report = {"lambda": best, "cv": scores[best], "cv_by_lambda": scores, "sessions": len(tracks),
              "readings": int(sum(len(t.t) for t in tracks))}
    return _fit(tracks, best, alpha), report


# --- the filter -------------------------------------------------------------------------------------------------------------------
def _filtered(track, params):
    f = PoseFilter(params)
    out = [f((a, b), ts) for ts, a, b in zip(track.t, track.ru, track.rv)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def refilter(track, params):
    """The track as the game will see it after the filter `params`, from the raw readings it was recorded with."""
    u, v = _filtered(track, params)
    return Track(t=track.t, u=u, v=v, ru=track.ru, rv=track.rv, name=track.name)


def gated(t, u, v, gate_speed):
    """The readings after the glitch gate alone: it does not depend on the One-Euro settings, so it is done once."""
    gate = GlitchGate(gate_speed)
    out = [gate((a, b), ts) for ts, a, b in zip(t, u, v)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def filter_many(t, x, settings):
    """One axis through the One-Euro filter for every (min_cutoff, beta, d_cutoff) of `settings` at once -> (settings, samples).

    The recursion of oneeuro.OneEuro, run on all the settings in parallel (a sample at or before the last one changes nothing)."""
    cutoff, beta, d_cutoff = (np.array(column, dtype=float) for column in zip(*settings))
    out = np.empty((len(cutoff), len(x)))
    xh, dxh = np.full(len(cutoff), float(x[0])), np.zeros(len(cutoff))
    out[:, 0] = xh
    tau_d = 1.0 / (2.0 * np.pi * d_cutoff)
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            a_d = 1.0 / (1.0 + tau_d / dt)
            dxh = a_d * (x[i] - xh) / dt + (1.0 - a_d) * dxh
            a = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * (cutoff + beta * np.abs(dxh))) / dt)
            xh = a * x[i] + (1.0 - a) * xh
        out[:, i] = xh
    return out


def tune_filter(tracks, cutoffs=CUTOFFS, betas=BETAS, d_cutoffs=D_CUTOFFS, cap=JITTER_CAP):
    """-> (FilterParams, report): the One-Euro settings whose output follows the teacher best on the raw readings, among the
    settings that jitter a still hand no more than `cap` times as much as the default settings do.

    Following the teacher alone is best served by passing every reading through (clean readings need no filter to match a
    smoothing of themselves), which shows all of the noise on a hand that is still: the paddle would shiver.  So the jitter is
    capped at the default's, and what is left to gain is lag: the best settings smooth a still hand as much and follow a moving
    one more closely."""
    raw = [t for t in tracks if t.ru is not None]
    if not raw:
        raise ValueError("no raw hand readings recorded yet: play a game (sessions keep them now) and train again")
    base = FilterParams()
    settings = [(base.min_cutoff, base.beta, base.d_cutoff)] + [(c, b, d) for c in cutoffs for b in betas for d in d_cutoffs]
    err2, step2 = np.zeros(len(settings)), np.zeros(len(settings))
    n_err = n_still = 0
    for tr in raw:
        g, ru = on_grid(tr.t, tr.ru)
        _, rv = on_grid(tr.t, tr.rv)
        teach = (teacher(ru), teacher(rv))
        speed = np.hypot(np.diff(teach[0]), np.diff(teach[1]))
        still = np.concatenate([[False], speed < STILL_STEP])             # False where the teacher has a hole (NaN < x is False)
        gated_uv = gated(tr.t, tr.ru, tr.rv, base.gate_speed)
        n_still += int(still.sum())
        for axis, gated_axis in zip(teach, gated_uv):
            known = np.isfinite(axis)
            n_err += int(known.sum())
            for lo in range(0, len(settings), CHUNK):
                out = filter_many(tr.t, gated_axis, settings[lo:lo + CHUNK])
                shown = np.stack([np.interp(g, tr.t, row) for row in out])
                err2[lo:lo + CHUNK] += ((shown[:, known] - axis[known]) ** 2).sum(axis=1)
                step2[lo:lo + CHUNK] += (np.diff(shown, axis=1)[:, still[1:]] ** 2).sum(axis=1)
    if not n_still:
        raise ValueError("the hand was never still in the recorded readings, so there is nothing to judge a filter's jitter on")
    rmse, shimmer = np.sqrt(err2 / n_err), np.sqrt(step2 / n_still)
    allowed = np.flatnonzero(shimmer <= cap * shimmer[0] * (1.0 + 1e-9))      # the default is always allowed
    best = int(allowed[np.argmin(rmse[allowed])])
    if rmse[best] >= rmse[0] - 1e-12:
        best = 0
    c, b, d = settings[best]
    return (FilterParams(min_cutoff=c, beta=b, d_cutoff=d, gate_speed=base.gate_speed),
            {"rmse_default": float(rmse[0]), "rmse_tuned": float(rmse[best]), "shimmer_default": float(shimmer[0]),
             "shimmer_tuned": float(shimmer[best]), "sessions": len(raw), "readings": int(sum(len(t.t) for t in raw))})
