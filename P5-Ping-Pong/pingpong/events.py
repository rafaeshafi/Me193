"""The data contract between modules (frozen so threads can share instances).

Hub IMU values are RAW COUNTS exactly as the hub sent them; convert with
config.ACCEL_PER_G / config.GYRO_PER_DPS (measured by the bench tools).
Timestamps are integer nanoseconds from a pingpong.clock clock.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ImuSample:
    t_ns: int            # arrival time (the hub sends no timestamps)
    g: tuple             # gyro (x, y, z), raw counts
    a: tuple             # accelerometer (x, y, z), raw counts
    src: str = "hub"


@dataclass(frozen=True)
class GestureEvent:
    t_ns: int
    gesture: int         # le.MOTION_GESTURE_* (SHAKE, TAPPED, FREEFALL, ...)


@dataclass(frozen=True)
class SwingEvent:
    kind: str            # "SWING_START" | "IMPACT"
    t_ns: int            # back-dated peak time for IMPACT
    w_pk: float          # peak |gyro - bias| in dps
    dur_ms: float
    n_reversals: int
    axis_unit: tuple     # unit vector of the gyro peak
    net_rot_unit: tuple
    a_lin_unit: tuple
    clipped: bool
    feat: tuple          # the 12 spin-classifier features
    src: str = "hub"


@dataclass(frozen=True)
class PaddlePose:
    t_scene_ns: int      # capture time minus the measured camera lag
    u: float             # hand position in shoulder-width units
    v: float
    conf: float          # landmark visibility 0..1
    hand: str            # "right" | "left"


@dataclass(frozen=True)
class TagEvent:
    role: str            # "START" | "LEVEL" | "PLAYER" | "MODE"
    value: int
    t_ns: int


@dataclass(frozen=True)
class GateResult:
    name: str            # "J1".."J6"
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class Verdict:
    kind: str            # "HIT" | "MISS" | "IGNORED"
    q_pos: float
    e_s: float           # timing error in seconds (negative = early)
    d_min_sw: float      # closest paddle approach in shoulder widths
    gates: tuple = ()    # tuple of GateResult


@dataclass(frozen=True)
class ShotParams:
    v_out: float         # m/s
    T: float             # topspin (-1 back .. +1 top)
    S: float             # sidespin
    A: float             # spin amplitude
    aim_a: float         # lateral aim -1..1
    q_total: float
    fault: Optional[str] = None   # None | "net" | "out"
    label: str = ""


@dataclass(frozen=True)
class HapticCmd:
    name: str
    strength: float
    fire_at_ns: int


@dataclass(frozen=True)
class GameEvent:
    kind: str
    t_ns: int
    data: dict = field(default_factory=dict)
