"""Haptic patterns and the actuator that plays them on the Double Motor.

The hub has no vibration API: a "tick" is a short timed motor pulse (the firmware
times it, so BLE jitter only moves the start), both motors start in ONE batched
write (antiphase by default, so the reaction torques cancel), and beep + light give
the cues that must always work even if the motors feel weak.

Limits enforced here (each unit-tested):
  * <= config.MAX_WRITES_PER_S device writes in any rolling second (bleak drops writes silently)
  * <= 1 pattern per 100 ms; a higher priority replaces pending lower ones (record > point > hit > tick)
  * motors on <= config.DUTY_CAP of any rolling config.DUTY_WINDOW_S (over the cap the motors stay
    quiet but beep + light still fire)
  * every motor pulse reports a blank window from its ACTUAL write time, so the swing detector
    can ignore the vibration it causes
  * no_motor / disarm() mute the motors and keep beep + light

ActuatorCore is synchronous and deterministic (fake clock friendly); Actuator wraps it in
the one thread that is allowed to send hub commands.
"""

import heapq
import itertools
import threading
from collections import deque
from dataclasses import dataclass

import legoeducation as le

import config
from pingpong.clock import Clock

MS = 1_000_000
S = 1_000_000_000
MAX_WRITES_PER_S = config.MAX_WRITES_PER_S
MIN_PATTERN_GAP_NS = 100 * MS
BLANK_LEAD_NS = 10 * MS


@dataclass(frozen=True)
class Step:
    at_ms: int
    kind: str                    # "motors" | "beep" | "light"
    ms: int = 0                  # motors: pulse length
    strength: int = 0            # motors: speed percent
    same_direction: bool = False
    hz: int = 0                  # beep
    count: int = 1
    beep_pattern: int = le.SOUND_PATTERN_BEEP_SINGLE
    color: int = le.LEGO_COLOR_NOCOLOR   # light
    light_pattern: int = le.LIGHT_PATTERN_SOLID


@dataclass(frozen=True)
class Pattern:
    name: str
    priority: int                # 4 record, 3 point/fault, 2 hit, 1 tick/ready
    steps: tuple


def _motors(at, ms, strength, same=False):
    return Step(at, "motors", ms=ms, strength=strength, same_direction=same)


def _beep(hz, count=1, pattern=le.SOUND_PATTERN_BEEP_SINGLE, at=0):
    return Step(at, "beep", hz=hz, count=count, beep_pattern=pattern)


def _light(color, pattern=le.LIGHT_PATTERN_SOLID, at=0):
    return Step(at, "light", color=color, light_pattern=pattern)


PATTERNS = {p.name: p for p in (
    Pattern("ready", 1, (_motors(0, 150, 40), _beep(660), _light(le.LEGO_COLOR_BLUE, le.LIGHT_PATTERN_BREATHE))),
    Pattern("countdown_tick", 1, (_motors(0, 25, 70),)),
    Pattern("countdown_go", 1, (_motors(0, 60, 100), _beep(880))),
    Pattern("hit_perfect", 2, (_motors(0, 60, 100), _beep(1760), _light(le.LEGO_COLOR_WHITE))),
    Pattern("hit_good", 2, (_beep(1320), _light(le.LEGO_COLOR_GREEN))),           # no motor cue by design
    Pattern("hit_early", 2, (_motors(0, 25, 50), _beep(700), _light(le.LEGO_COLOR_ORANGE))),
    Pattern("hit_late", 2, (_motors(0, 25, 50), _motors(105, 25, 50), _beep(1000), _light(le.LEGO_COLOR_ORANGE))),
    Pattern("fault", 3, (_motors(0, 400, 60, same=True), _beep(220),
                         _light(le.LEGO_COLOR_RED, le.LIGHT_PATTERN_LONG_BLINK))),
    Pattern("point_won", 3, (_motors(0, 25, 50), _motors(125, 25, 50), _motors(250, 200, 50),
                             _beep(880, 2, le.SOUND_PATTERN_BEEP_DOUBLE), _light(le.LEGO_COLOR_GREEN))),
    Pattern("menu_tick", 1, (_motors(0, 20, 35),)),                                  # the hand comes onto a button in a menu
    Pattern("menu_select", 1, (_motors(0, 45, 70), _beep(1320))),                    # ... and holds on it until it is pressed
    Pattern("record", 4, tuple(_motors(160 * i, 40, 80) for i in range(5)) + (
        _beep(2000, 3, le.SOUND_PATTERN_BEEP_TRIPLE), _light(le.LEGO_COLOR_PURPLE))),
)}


class ActuatorCore:
    def __init__(self, device, *, clock=None, no_motor=False, antiphase=True, on_blank=None):
        self.dev = device
        self.clock = clock or Clock()
        self.no_motor, self.antiphase, self.on_blank = no_motor, antiphase, on_blank
        self._armed = True
        self._heap = []
        self._seq = itertools.count()
        self._last_start_ns = None
        self._last_priority = 0
        self._write_times = deque()
        self._motor_log = deque()          # (t_ns, ms) of executed motor pulses
        self._batch_open = False
        self._closed = False
        # Two locks, so the 60 Hz game thread never waits behind a BLE write: _lock guards the
        # schedule and is only ever held for a few microseconds; _exec_lock serialises the
        # device writes (and the rate/duty bookkeeping they update) on the actuator thread.
        self._lock = threading.RLock()
        self._exec_lock = threading.RLock()

    # --- control --------------------------------------------------------------------
    def arm(self):
        with self._lock:
            self._armed = True

    def disarm(self):
        """Stop the motors NOW and keep them quiet (key D). Beep and light still work."""
        with self._lock:
            self._armed = False
        self.dev.motor_stop(motor=le.MOTOR_BOTH, blocking=False)

    def submit(self, name, fire_at_ns=None):
        """Schedule a pattern; False if it was dropped (<= 1 pattern per 100 ms at equal/lower priority)."""
        pattern = PATTERNS[name]
        with self._lock:
            if self._closed:
                return False
            fire = self.clock.now_ns() if fire_at_ns is None else fire_at_ns
            if self._last_start_ns is not None and abs(fire - self._last_start_ns) < MIN_PATTERN_GAP_NS:
                if pattern.priority <= self._last_priority:
                    return False
                self._heap = [h for h in self._heap if h[2] >= pattern.priority]
                heapq.heapify(self._heap)
            for step in pattern.steps:
                heapq.heappush(self._heap, (fire + step.at_ms * MS, next(self._seq), pattern.priority, step))
            self._last_start_ns, self._last_priority = fire, pattern.priority
            return True

    def process(self, now_ns):
        """Execute everything due; return the next wake-up time in ns, or None when idle."""
        with self._exec_lock:
            while True:
                with self._lock:
                    if not self._heap or self._closed:
                        return None
                    due = self._heap[0][0]
                    if due > now_ns:
                        return due
                    wake = self._write_allowed_at(now_ns)
                    if wake > now_ns:
                        return wake
                    _, _, _, step = heapq.heappop(self._heap)
                self._execute(step, now_ns)                 # the slow BLE write runs outside _lock

    def close(self):
        with self._lock:
            self._closed = True
            self._heap.clear()
        if self._batch_open:                                 # best effort: never wait on a hung write
            try:
                self.dev.cancel_batch()
            except Exception:
                pass
            self._batch_open = False

    # --- internals ------------------------------------------------------------------------
    def _write_allowed_at(self, now_ns):
        while self._write_times and now_ns - self._write_times[0] >= S:
            self._write_times.popleft()
        if len(self._write_times) < MAX_WRITES_PER_S:
            return now_ns
        return self._write_times[0] + S

    def _execute(self, step, now_ns):
        if step.kind == "motors":
            if self.no_motor or not self._armed or not self._duty_ok(step, now_ns):
                return                                   # no write, no blank
            self._motor_pulse(step)
            self._motor_log.append((now_ns, step.ms))
            self._write_times.append(now_ns)
            if self.on_blank:
                end = now_ns + step.ms * MS + round(config.BLANK_AFTER_PULSE_S * S)
                self.on_blank(now_ns - BLANK_LEAD_NS, end)
        elif step.kind == "beep":
            self.dev.beep(pattern=step.beep_pattern, frequency=step.hz, count=step.count, blocking=False)
            self._write_times.append(now_ns)
        elif step.kind == "light":
            self.dev.light_color(step.color, pattern=step.light_pattern, blocking=False)
            self._write_times.append(now_ns)

    def _duty_ok(self, step, now_ns):
        window_ns = round(config.DUTY_WINDOW_S * S)
        while self._motor_log and now_ns - self._motor_log[0][0] >= window_ns:
            self._motor_log.popleft()
        on_ms = sum(ms for _, ms in self._motor_log)
        return on_ms + step.ms <= config.DUTY_CAP * config.DUTY_WINDOW_S * 1000

    def _motor_pulse(self, step):
        cw, ccw = le.MOTOR_MOVE_DIRECTION_CLOCKWISE, le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
        right_dir = cw if (step.same_direction or not self.antiphase) else ccw
        self.dev.begin_batch()
        self._batch_open = True
        self.dev.motor_run_for_time(step.ms, direction=cw, motor=le.MOTOR_LEFT, speed=step.strength, blocking=False)
        self.dev.motor_run_for_time(step.ms, direction=right_dir, motor=le.MOTOR_RIGHT, speed=step.strength,
                                    blocking=False)
        self.dev.end_batch(blocking=False)
        self._batch_open = False


class Actuator:
    """The ONE thread that sends hub commands (BLE writes block their caller)."""

    def __init__(self, core, log=print):
        self.core, self.log = core, log
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="actuator", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def submit(self, name, fire_at_ns=None):
        ok = self.core.submit(name, fire_at_ns)
        self._wake.set()
        return ok

    def _run(self):
        while not self._stop.is_set():
            try:
                nxt = self.core.process(self.core.clock.now_ns())
            except Exception as exc:                      # a failed write must not kill the game
                self.log(f"actuator: {exc}")
                nxt = None
            wait = 0.5 if nxt is None else max(0.001, (nxt - self.core.clock.now_ns()) / 1e9)
            self._wake.wait(wait)
            self._wake.clear()

    def stop(self):
        """Teardown step 1: stop and join the thread, then cancel anything left open."""
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.core.close()
