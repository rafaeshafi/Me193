"""A tabular Q-learning opponent that learns where YOU fail (opt-in with --learn, per player, kept between games).

State  = where your hand is (the left, middle or right third of your reach box) x the column of the zone the
         computer served to last: 3 x 3 = 9 states.
Action = which of the nine target zones to serve to.
Reward = 1 when you miss or fault, otherwise a small "how hard was that" signal, 0.5 * (1 - hit quality).

    Q(s, a) += alpha * (reward + gamma * max Q(s', .) - Q(s, a))

Choosing a zone adds weight x Q(s, a) to the utility the softmax policy already uses (so the computer still
wrong-foots your hand) and explores with probability epsilon, which shrinks each game down to a floor.  The
weight depends on the level: 0 at Rookie (it learns there but does not use it yet), growing to Pro.  The
constants alpha 0.1, gamma 0.9, decay 0.98 and floor 0.05 come from the class Q-learning example
(scripts/rl_straight.py).
"""

import math
import random
from pathlib import Path

import numpy as np

from pingpong import profile

N = 9
WEIGHTS = {1: 0.0, 2: 0.4, 3: 0.8, 4: 1.0}


def weight(level):
    return WEIGHTS[level.tag]


class QBandit:
    def __init__(self, *, alpha=0.1, gamma=0.9, epsilon=0.15, decay=0.98, floor=0.05, rng=None):
        self.alpha, self.gamma, self.epsilon, self.decay, self.floor = alpha, gamma, epsilon, decay, floor
        self.rng = rng or random.Random()
        self.table, self.games = np.zeros((N, N)), 0

    def state(self, hand_a, last_col):
        return min(2, max(0, int(hand_a * 3))) * 3 + last_col

    def choose(self, state, candidates, *, weight, base, temp):
        """The action (an index 0..8) among `candidates`, given the softmax utilities `base` for each."""
        if weight > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)
        if math.isinf(temp):
            return self.rng.choice(candidates)
        utils = [b + weight * self.table[state, a] for a, b in zip(candidates, base)]
        top = max(utils)
        return self.rng.choices(candidates, [math.exp((u - top) / temp) for u in utils])[0]

    def update(self, state, action, reward, next_state):
        target = reward + (0.0 if next_state is None else self.gamma * float(self.table[next_state].max()))
        self.table[state, action] += self.alpha * (target - self.table[state, action])

    def end_game(self):
        self.games += 1
        self.epsilon = max(self.floor, self.epsilon * self.decay)


def _path(name, root):
    return Path(root or profile.default_root()) / profile.slug(name) / "qtable.npz"


def save_for(name, q, root=None):
    path = _path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, table=q.table, games=q.games, epsilon=q.epsilon)
    return path


def load_for(name, root=None, **kw):
    """The player's saved table, or None if they have none yet."""
    path = _path(name, root)
    if not path.exists():
        return None
    try:
        data = np.load(path)
        q = QBandit(**kw)
        q.table, q.games, q.epsilon = data["table"], int(data["games"]), float(data["epsilon"])
        assert q.table.shape == (N, N)
    except Exception as exc:
        raise ValueError(f"the qtable file {path} is damaged ({exc}); delete it to start learning afresh") from exc
    return q
