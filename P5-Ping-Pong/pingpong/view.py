"""What the screen shows: the world as it will be when the frame reaches the eye.

The game runs on one clock; a frame drawn now is seen display_s later (and up to a loop's period after what it shows),
the hand's newest reading is already old, and the hub's stamps are late.  View puts every moving thing where the
PLAYER will see it: the ball and the computer's paddle (deterministic) are drawn ahead by the display's delay, the hand
is extrapolated across its whole pipeline, and a swing is a lunge that reaches the ball exactly when the picture shows
the contact.  Nothing here changes the game: it only decides what to draw.
"""

import dataclasses

from pingpong import latency as latency_mod
from pingpong import pd, physics, stage

S = 1_000_000_000
CPU_SWING_S = 0.3          # how long the computer's paddle takes to hit the ball (the HUD animates it)
CPU_RECOVER_S = 0.6        # ... and to drift back to the middle after a return


class View:
    def __init__(self, game, lat, poses):
        self.game, self.latency, self.poses = game, lat, poses
        self.paddle_angle = 0.0                          # degrees: how far the hub is turned in the hand (set each frame)
        self._stroke = None
        self.cpu_swing_ns = None                         # when the computer last hit the ball

    # --- time -----------------------------------------------------------------------------------------------------
    def view_ns(self, now):
        """The moment the frame drawn at `now` will be SEEN (a frozen game stays frozen)."""
        g = self.game
        if g.paused and g.paused_at_ns is not None:
            return g.paused_at_ns
        return now + round(self.latency.view_ahead_s * S)

    # --- your paddle ---------------------------------------------------------------------------------------------------
    def hand(self, now):
        return latency_mod.predict_hand(self.poses, now, self.latency)

    def rest(self, now):
        """Where the hand puts the paddle (None until a hand has been seen)."""
        hand = self.hand(now)
        return None if hand is None else stage.rest_position(self.game.judge.box, *hand)

    def paddle(self, now, view):
        rest = self.rest(now)
        if rest is None or self._stroke is None:
            return rest, rest
        if self._stroke.done(view):
            self._stroke = None
            return rest, rest
        return self._stroke.pose(rest, view), rest

    def start_stroke(self, events, now):
        """A swing was seen: the paddle lunges to the ball it hit, or forward at nothing."""
        hit = next((e for e in events if e.kind in ("hit", "fault") and "contact" in e.data), None)
        if hit is not None:
            self._stroke = stage.hit_stroke(now, hit.data["contact_ns"], hit.data["contact"])
            return
        rest = self.rest(now)
        if rest is not None:
            self._stroke = stage.miss_stroke(rest, now)

    def reach_m(self):
        """How far from the paddle across the table a ball can still be hit: the level's radius, in metres."""
        box, level = self.game.judge.box, self.game.level
        return level.radius_sw * 2 * 0.8 * physics.HALF_WIDTH_M / (box.u_max - box.u_min)

    def zone(self, now):
        """The stretch of table (far z, near z) over which the incoming ball can be hit, for the paddle where it stands."""
        g = self.game
        if g.incoming is None or g.incoming_leg is None:
            return None
        hand = self.hand(now)
        sweet = g.judge.sweet_ns(g.incoming, None if hand is None else hand[1])
        return stage.zone(g.incoming_leg, sweet, g.level)

    # --- the ball -------------------------------------------------------------------------------------------------------
    def ball(self, view):
        """(x, y, z): your ball once it has left your paddle, otherwise the one coming at you, flying on past you if it
        is not hit (it is seen to go by), and None while there is no ball."""
        g = self.game
        if g.phase in ("LOBBY", "COUNTDOWN"):
            return None
        out, inc = g.outgoing_leg, g.incoming_leg
        if out is not None and view >= out.t0_ns:
            return out.position(view) if view <= out.end_ns else None
        if inc is not None and view <= inc.arrival_ns + round(physics.FLY_ON_S * S):
            return inc.position(view)
        return None

    # --- the computer's paddle ----------------------------------------------------------------------------------------------
    def cpu_swing(self, view):
        """How far through its stroke the computer's paddle is (0..1), None when it is not hitting."""
        if self.cpu_swing_ns is None:
            return None
        progress = (view - self.cpu_swing_ns) / (CPU_SWING_S * S)
        return progress if 0.0 <= progress < 1.0 else None

    def cpu_x(self, view):
        """The computer's paddle: after its return it stands where it hit and drifts back to the middle; while it chases
        your shot it moves to where the ball will land (always arrives in Rally, can fall short in Match)."""
        g, leg = self.game, self.game.outgoing_leg
        if g.phase == "RALLY" and g.incoming is not None and g.incoming_leg is not None:
            back = g.incoming_leg
            return back.x_start * max(0.0, 1.0 - max(0.0, (view - back.t0_ns) / S) / CPU_RECOVER_S)
        if g.phase != "RALLY" or leg is None:
            return 0.0
        level = g.level
        if g.mode != "match":                          # Survival never misses: the drawn paddle always gets there
            level = dataclasses.replace(level, cpu_speed_ms=50.0, tau_s=min(level.tau_s, 0.25 * leg.flight_s))
        return pd.paddle_x(level, leg.x_end, 0.0, max(0.0, (view - leg.t0_ns) / S))
