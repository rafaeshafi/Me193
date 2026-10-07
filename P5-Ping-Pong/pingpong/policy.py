"""The computer opponent: what it serves and whether it returns (answers "describe the policy").

Perceive -> choose a target zone by a softmax over a utility that wrong-foots the
player's tracked hand -> in Match, return or miss with a probability that grows
with the shot's speed, spin and the CPU paddle's physical reach deficit; in
Survival the CPU never misses and only the ramp (speed, spin, wide balls,
every 10th ball a special) changes.
"""

import math
import random
from dataclasses import dataclass

from pingpong import levels, pd, physics, qbandit

ZONES = tuple((a, b) for b in (0.15, 0.5, 0.85) for a in (0.15, 0.5, 0.85))
V_MAX = 14.0
K_V = {1: 0.10, 2: 0.08, 3: 0.06, 4: 0.05}      # per m/s above the level's v_ref
K_W = {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15}      # per unit of spin above the level's spin_ref
K_D = 1.5                                         # per metre the CPU paddle cannot reach


def in_reach(zone, reach):
    """A zone's (a, b) pulled toward the middle of the reach box: the level decides how much of the box the balls use.

    Only the served aim moves; the zone labels the learner and the softmax work with stay on the 0.15 / 0.5 / 0.85 grid."""
    if reach == 1.0:
        return zone
    return tuple(0.5 + (c - 0.5) * reach for c in zone)


@dataclass(frozen=True)
class ServePlan:
    v: float
    aim_ab: tuple
    topspin: float
    sidespin: float
    special: bool


def zone_utility(zone, player_a, v, v_max, player_va):
    """Far from the player's hand, away from where it is heading, and not too risky at speed."""
    a_z, _ = zone
    x_z, x_p = physics.x_of_a(a_z), physics.x_of_a(player_a)
    width = 2 * physics.HALF_WIDTH_M
    lead = (-math.copysign(1.0, player_va) if player_va else 0.0) * (x_z - x_p)
    edge = 1.0 if a_z != 0.5 else 0.0
    risk = 0.05 + 0.10 * (v / v_max) ** 2 + 0.15 * edge
    return abs(x_z - x_p) / width + 0.6 * lead / width - risk


def miss_probability(level, v, spin_amp, reach_deficit_m):
    p = (level.p0 + K_V[level.tag] * max(0.0, v - level.v_ref)
         + K_W[level.tag] * max(0.0, spin_amp - level.spin_ref) + K_D * reach_deficit_m)
    return max(0.0, min(0.95, p))


def reach_deficit_m(level, x_land_m, x_cpu_m, flight_s):
    """Metres between the CPU paddle and the landing point when the ball arrives (a PD controller chases it)."""
    return pd.paddle_error_m(level, x_land_m, x_cpu_m, flight_s)


class CpuPolicy:
    def __init__(self, rng=None, learner=None):
        self.rng = rng or random.Random(0)
        self.learner = learner                       # an optional qbandit.QBandit (--learn)
        self._pending, self._last_col = None, 1

    def serve(self, level, s_prev, n_hits, player_a, survival, player_va=0.0):
        ramp = levels.survival_ramp(n_hits) if survival else 1.0
        v = levels.incoming_speed(level, s_prev, ramp)
        special = bool(survival and n_hits > 0 and n_hits % 10 == 0)
        if special:                                   # a drop or a screamer
            v *= self.rng.choice([0.7, 1.3])
        candidates = ZONES
        if survival and self.rng.random() < min(0.8, 0.3 + 0.01 * n_hits):
            candidates = tuple(z for z in ZONES if z[0] != 0.5)       # wide ball
        aim = self._pick_zone(candidates, level, player_a, v, player_va)
        amp = min(1.0, 0.10 + 0.03 * n_hits) if survival else level.spin_variety
        top = amp * self.rng.uniform(-1.0, 1.0)
        side = 0.5 * amp * self.rng.uniform(-1.0, 1.0)
        return ServePlan(v=v, aim_ab=in_reach(aim, level.reach), topspin=0.0 if special else top,
                         sidespin=0.0 if special else side, special=special)

    def observe(self, reward, terminal=False):
        """The game reports how the ball just served went for the player (1 = they failed); the learner learns."""
        p = self._pending
        if self.learner is None or p is None:
            return
        if terminal:
            self.learner.update(p["state"], p["action"], reward, None)
            self._pending = None
        else:
            p["reward"] = reward

    def end_game(self):
        if self.learner is not None:
            self.learner.end_game()

    def _pick_learned(self, candidates, level, player_a, v, player_va):
        q, state = self.learner, self.learner.state(player_a, self._last_col)
        p = self._pending
        if p is not None and p["reward"] is not None:
            q.update(p["state"], p["action"], p["reward"], state)      # the previous ball's outcome, now with its s'
        temp = level.softmax_temp
        base = [zone_utility(z, player_a, v, V_MAX, player_va) for z in candidates]
        action = q.choose(state, [ZONES.index(z) for z in candidates], weight=qbandit.weight(level), base=base,
                          temp=temp)
        self._pending = {"state": state, "action": action, "reward": None}
        self._last_col = {0.15: 0, 0.5: 1, 0.85: 2}[ZONES[action][0]]
        return ZONES[action]

    def _pick_zone(self, candidates, level, player_a, v, player_va):
        if self.learner is not None:
            return self._pick_learned(candidates, level, player_a, v, player_va)
        temp = level.softmax_temp
        if math.isinf(temp):
            return self.rng.choice(candidates)
        utils = [zone_utility(z, player_a, v, V_MAX, player_va) for z in candidates]
        top = max(utils)
        weights = [math.exp((u - top) / temp) for u in utils]
        return self.rng.choices(candidates, weights)[0]

    def returns(self, level, v, spin_amp, reach_deficit_m, survival):
        if survival:
            return True
        return self.rng.random() >= miss_probability(level, v, spin_amp, reach_deficit_m)
