"""Game event -> haptic pattern.

Motor cues for early/late need free time inside the flight; at fast levels (flight
under 0.5 s) they degrade to the beep + light-only cue so the pulse cannot land on
top of the next swing window.  The perfect-hit thump and the fault/record cues
always play.
"""

from pingpong import physics

MIN_FLIGHT_FOR_TIMING_CUES_S = 0.5

_HIT = {"perfect": "hit_perfect", "good": "hit_good", "early": "hit_early", "late": "hit_late"}


def pattern_for(event, level):
    kind, data = event.kind, event.data
    if kind == "hit":
        name = _HIT.get(data.get("label"), "hit_good")
        if name in ("hit_early", "hit_late") and physics.D_M / level.v_tier < MIN_FLIGHT_FOR_TIMING_CUES_S:
            return "hit_good"
        return name
    if kind in ("miss", "fault"):
        return "fault"
    if kind == "record":
        return "record"
    if kind == "point":
        return "point_won" if data.get("scorer") == "player" else "fault"
    return None


def play(events, level, actuator, *, latency=None, now_ns=None):
    """Submit the cue for each event, in order; a dropped pattern is fine (priority rules decide).

    A hit is only DECIDED up to a tenth of a second before the paddle meets the ball on screen, and the motors need
    haptic_s to move: its thump is scheduled for contact - haptic_s, so it arrives with the picture."""
    for event in events:
        name = pattern_for(event, level)
        if not name:
            continue
        contact = event.data.get("contact_ns") if event.kind == "hit" else None
        if contact is not None and latency is not None and now_ns is not None:
            actuator.submit(name, max(now_ns, contact - round(latency.haptic_s * 1e9)))
        else:
            actuator.submit(name)
