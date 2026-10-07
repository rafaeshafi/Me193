"""Difficulty levels (starting values; tools/sim.py and play-testing tune them).

The AprilTag sets the level and the level sets the ball speed: the player's own
swing can only modulate the incoming speed by +-10%, so adjacent levels never
overlap.  Insane (key/tag 4) is a config row only -- it stacks detection and
camera lag against a 0.32 s flight and is probably unplayable.
"""

from dataclasses import dataclass

INF = float("inf")


@dataclass(frozen=True)
class Level:
    name: str
    tag: int
    v_tier: float        # m/s, CPU -> player
    early_s: float       # hit window before arrival
    late_s: float        # hit window after arrival
    radius_sw: float     # paddle must be this close to the ball (shoulder widths)
    tau_s: float         # CPU reaction delay
    cpu_speed_ms: float  # CPU paddle speed limit
    p0: float            # base miss probability (Match)
    aim_sigma_m: float
    softmax_temp: float
    spin_variety: float
    v_ref: float         # speed above which the CPU starts missing
    spin_ref: float
    fault_th: float      # player fault threshold on s * (1 - q_total)
    reach: float = 1.0   # the share of the reach box the balls arrive in (0.6: the middle 60%), so an easy level
                         # never asks for the corners of a box that was stretched to reach


LEVELS = {
    1: Level("Rookie", 1, 3.5, 0.30, 0.18, 0.55, 0.40, 1.2, 0.10, 0.25, INF, 0.0, 5.0, 0.20, 0.60, 0.6),
    2: Level("Club", 2, 5.0, 0.22, 0.14, 0.45, 0.28, 2.2, 0.06, 0.15, 1.0, 0.2, 7.0, 0.35, 0.50, 0.8),
    3: Level("Pro", 3, 7.0, 0.16, 0.10, 0.35, 0.18, 3.5, 0.03, 0.08, 0.4, 0.4, 9.0, 0.50, 0.42),
    4: Level("Insane", 4, 9.5, 0.12, 0.07, 0.28, 0.10, 5.0, 0.01, 0.04, 0.15, 0.7, 11.0, 0.65, 0.35),
}


def level_for_tag(tag):
    """Tags 1-3 are the printed LEVEL cards; 0 is START and 4 is not printed."""
    return LEVELS[tag] if tag in (1, 2, 3) else None


def incoming_speed(level, s_prev, ramp=1.0):
    """CPU -> player ball speed: the tag level dominates, the last swing modulates +-10%."""
    return level.v_tier * (0.9 + 0.2 * s_prev) * ramp


def survival_ramp(n_hits):
    return min(1.0 + 0.03 * n_hits, 1.8)
