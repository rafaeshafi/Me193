"""The pose model: how the hand's readings are filtered and where the hand will be when the frame is seen.

Both parts are LEARNED from the player's own recorded tracks (pingpong/posetrain.py, `./pp train_pose`):

  * the filter is a One-Euro filter (smooth when the hand is still, hardly delayed when it moves) whose three settings
    are tuned to follow a zero-phase smoothing of the player's own raw readings, behind a gate that drops the
    single-frame glitches a landmark flip makes;
  * the predictor is a linear model (ridge regression) from the last 0.2 s of the smoothed track to where the hand
    will be `lead` seconds later, so the paddle on screen can be drawn where the hand will be when the frame is seen
    (the camera pipeline, the hub's stamps and the screen all delay what the player sees; see latency.py).

The defaults are what the game used before any training: the fixed One-Euro settings and no predictor (the caller then
extrapolates by the hand's recent speed).  A damaged or missing model file never stops the game.
"""

import bisect
import json
import math
from dataclasses import asdict, dataclass, field

from pingpong import profile
from pingpong.oneeuro import OneEuro2D
from pingpong.pose_features import LANDMARKERS

VERSION = 1
FILE = "pose_model.json"
STEP_S = 1 / 30           # the predictor sees the track on a uniform grid of this period
K = 6                     # lags it looks back over (0.2 s) ...
WARM = 6                  # ... plus this many older grid points that only warm its smoother up
H0 = 0.15                 # the lead the weights are centred on (they are linear in lead - H0)
MAX_GAP_S = 0.12          # neighbouring readings further apart than this are not bridged
MAX_LEAD_S = 0.30


@dataclass(frozen=True)
class FilterParams:
    min_cutoff: float = 1.2       # Hz: how hard a still hand is smoothed
    beta: float = 5.0             # how fast the cutoff rises with the hand's speed
    d_cutoff: float = 1.0         # Hz: smoothing of the speed estimate itself
    gate_speed: float = 40.0      # shoulder widths per second: a reading implying more is a landmark flip, not a hand

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**{k: float(d[k]) for k in ("min_cutoff", "beta", "d_cutoff", "gate_speed") if k in d})


class PoseFilter:
    """The filter the camera worker runs on every reading: a One-Euro filter behind a glitch gate."""

    def __init__(self, params=None):
        self.params = params or FilterParams()
        self._f = OneEuro2D(min_cutoff=self.params.min_cutoff, beta=self.params.beta, d_cutoff=self.params.d_cutoff)
        self._last = self._t = None
        self._glitches = 0

    def reset(self):
        self._f.reset()
        self._last = self._t = None
        self._glitches = 0

    def __call__(self, xy, t_s):
        if self._last is not None and t_s > self._t:
            speed = math.hypot(xy[0] - self._last[0], xy[1] - self._last[1]) / (t_s - self._t)
            if speed > self.params.gate_speed and self._glitches < 2:        # a flip lasts a frame; a real jump goes on
                self._glitches += 1
                xy = self._last
            else:
                self._glitches = 0
        self._last, self._t = (xy[0], xy[1]), t_s
        return self._f(xy, t_s)


# --- the predictor -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HandPredictor:
    a_u: tuple
    b_u: tuple
    a_v: tuple
    b_v: tuple
    alpha: float = 0.5            # the smoother in front of the weights: a first-order low-pass per grid step

    def _axis(self, grid, a, b, lead_s):
        s = grid[0]
        smooth = []
        for g in grid:
            s = g if not smooth else s + self.alpha * (g - s)
            smooth.append(s)
        present = smooth[-1]
        h = lead_s - H0
        return present + sum((a[k - 1] + h * b[k - 1]) * (smooth[-1 - k] - present) for k in range(1, K + 1))

    def predict(self, poses, lead_s, min_conf=0.5):
        """(u, v) of the hand `lead_s` after the newest reading, or None when the readings do not cover the last
        0.4 s without a hole (the caller then falls back to extrapolating by the hand's speed)."""
        good = [p for p in poses if p.conf >= min_conf]
        if not good:
            return None
        n = K + WARM + 1
        t_n = good[-1].t_scene_ns / 1e9
        times = [p.t_scene_ns / 1e9 for p in good]
        grid_u, grid_v = [], []
        for j in range(n - 1, -1, -1):                                    # oldest first
            t = t_n - j * STEP_S
            i = bisect.bisect_left(times, t - 1e-9)
            if i >= len(times) or (times[i] > t + 1e-9 and (i == 0 or times[i] - times[i - 1] > MAX_GAP_S)):
                return None
            if abs(times[i] - t) <= 1e-9:
                u, v = good[i].u, good[i].v
            else:
                f = (t - times[i - 1]) / (times[i] - times[i - 1])
                u = good[i - 1].u + f * (good[i].u - good[i - 1].u)
                v = good[i - 1].v + f * (good[i].v - good[i - 1].v)
            grid_u.append(u)
            grid_v.append(v)
        lead = max(0.0, min(MAX_LEAD_S, lead_s))
        return self._axis(grid_u, self.a_u, self.b_u, lead), self._axis(grid_v, self.a_v, self.b_v, lead)

    def to_json(self):
        return {"alpha": self.alpha, "a_u": list(self.a_u), "b_u": list(self.b_u), "a_v": list(self.a_v),
                "b_v": list(self.b_v)}

    @classmethod
    def from_json(cls, d):
        return cls(a_u=tuple(d["a_u"]), b_u=tuple(d["b_u"]), a_v=tuple(d["a_v"]), b_v=tuple(d["b_v"]),
                   alpha=float(d.get("alpha", 0.5)))


@dataclass(frozen=True)
class PoseModel:
    filter: FilterParams = field(default_factory=FilterParams)
    predictor: HandPredictor = None
    meta: dict = field(default_factory=dict)
    landmarker: str = "lite"      # which pose model reads the camera frames (pose_features.LANDMARKERS)

    @classmethod
    def default(cls):
        return cls()

    def to_json(self):
        return {"version": VERSION, "filter": self.filter.to_json(), "landmarker": self.landmarker,
                "predictor": None if self.predictor is None else self.predictor.to_json(), "meta": dict(self.meta)}

    @classmethod
    def from_json(cls, d):
        return cls(filter=FilterParams.from_json(d.get("filter", {})),
                   predictor=None if d.get("predictor") is None else HandPredictor.from_json(d["predictor"]),
                   meta=dict(d.get("meta", {})), landmarker=d["landmarker"] if d.get("landmarker") in LANDMARKERS else "lite")


def path_for(player, root=None):
    return (root or profile.default_root()) / profile.slug(player) / FILE


def save_for(player, model, root=None):
    path = path_for(player, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=1))
    return path


def load_for(player, root=None):
    """The player's trained model, or None (no file, or one the game cannot read: it carries on with the defaults)."""
    try:
        return PoseModel.from_json(json.loads(path_for(player, root).read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return None
