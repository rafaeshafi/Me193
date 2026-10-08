"""Difficulty levels (starting values; tools/sim.py and play-testing tune them).

The AprilTag sets the level and the level sets the ball speed: the player's own
swing can only modulate the incoming speed by +-10%, so adjacent levels never
overlap.  Insane (key/tag 4) is a config row only -- it stacks detection and
camera lag against a 0.32 s flight and is probably unplayable.
"""

from dataclasses import dataclass

INF = float("inf")
MODE_NAMES = {"survival": "RALLY", "match": "MATCH"}       # what the player sees; the stored key stays "survival"


@dataclass(frozen=True)
class Level:
    name: str
    tag: int
    v_tier: float        # m/s, CPU -> player
    early_s: float       # hit window before arrival: the stroke's end may be this early (real swings run early)
    late_s: float        # hit window after arrival
    radius_sw: float     # the hand must be this close to the ball ACROSS the court (shoulder widths; height is free)
    tau_s: float         # CPU reaction delay
    cpu_speed_ms: float  # CPU paddle speed limit
    p0: float            # base miss probability (Match)
    aim_sigma_m: float
    softmax_temp: float
    spin_variety: float
    v_ref: float         # speed above which the CPU starts missing
    spin_ref: float
    fault_th: float      # player fault threshold on s * (1 - q_total); s * (1 - q) is at most 1, so >= 1 means no faults
    reach: float = 1.0   # the share of the reach box the balls arrive in (0.6: the middle 60%), so an easy level
                         # never asks for the corners of a box that was stretched to reach


LEVELS = {
    # (a hit that ended a game as a "fault" looked like a bug: Rookie and Club never fault, only Pro and Insane do)
    # 10/8: every level eased after play (slower balls, a wider paddle, balls nearer the middle, a computer that is slower to reach them,
    # misses more and wrong-foots less), keeping the order of the levels and the gaps between their speeds
    1: Level("Rookie", 1, 2.0, 0.60, 0.28, 0.85, 0.45, 1.0, 0.16, 0.25, INF, 0.0, 4.0, 0.15, 3.0, 0.5),
    2: Level("Club", 2, 3.8, 0.38, 0.21, 0.70, 0.34, 1.8, 0.10, 0.15, 1.4, 0.1, 6.0, 0.28, 2.0, 0.65),
    3: Level("Pro", 3, 5.2, 0.28, 0.15, 0.55, 0.24, 2.8, 0.06, 0.08, 0.7, 0.25, 8.0, 0.40, 0.55, 0.8),
    4: Level("Insane", 4, 7.0, 0.20, 0.10, 0.40, 0.15, 4.0, 0.03, 0.04, 0.3, 0.5, 10.0, 0.55, 0.45),
}


def level_for_tag(tag):
    """Tags 1-3 are the printed LEVEL cards; 0 is START and 4 is not printed."""
    return LEVELS[tag] if tag in (1, 2, 3) else None


def incoming_speed(level, s_prev, ramp=1.0):
    """CPU -> player ball speed: the tag level dominates, the last swing modulates +-10%."""
    return level.v_tier * (0.9 + 0.2 * s_prev) * ramp


def survival_ramp(n_hits):
    return min(1.0 + 0.03 * n_hits, 1.8)
