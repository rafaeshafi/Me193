"""Keyboard control for the game window (pure: no cv2 here, so it is testable).

SPACE start / (fake mode) swing   1-3 level   M mode   X x-ray   D disarm motors   Q/ESC quit
Fake mode only:  J soft swing   SPACE normal swing   K hard swing
"""

from pingpong import levels
from pingpong.events import SwingEvent

SPACE, ESC = 32, 27
SWING_KEYS = {ord("j"): 320.0, SPACE: 600.0, ord("k"): 1200.0}     # peak gyro in dps


def fake_swing(now_ns, w_pk):
    return SwingEvent(kind="IMPACT", t_ns=now_ns, w_pk=w_pk, dur_ms=150.0, n_reversals=0, axis_unit=(1, 0, 0),
                      net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1), clipped=False, feat=(0.0,) * 12)


def handle_key(session, key, *, fake):
    """-> (action, events) with action in {None, "quit"}."""
    game = session.game
    if key in (ESC, ord("q")):
        return "quit", []
    if key == SPACE and game.phase in ("LOBBY", "MATCH_OVER"):
        session.on_start()
        return None, []
    if fake and key in SWING_KEYS and game.phase == "RALLY":
        return None, session.on_swing(fake_swing(session.clock.now_ns(), SWING_KEYS[key]))
    if key in (ord("1"), ord("2"), ord("3")):
        game.set_level(levels.LEVELS[int(chr(key))])
    elif key == ord("m"):
        game.set_mode("match" if game.mode == "survival" else "survival")
    elif key == ord("x"):
        session.xray = not session.xray
    elif key == ord("d") and session.actuator is not None:
        core = getattr(session.actuator, "core", session.actuator)
        if getattr(session, "motors_armed", True):
            core.disarm()
            session.motors_armed = False
        else:
            core.arm()
            session.motors_armed = True
    return None, []
