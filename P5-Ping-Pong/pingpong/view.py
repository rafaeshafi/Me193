"""What the screen shows: the world as it will be when the frame reaches the eye.

The game runs on one clock; a frame drawn now is seen display_s later (and up to a loop's period after what it shows),
the hand's newest reading is already old, and the hub's stamps are late.  View puts every moving thing where the
PLAYER will see it: the ball and the computer's paddle (deterministic) are drawn ahead by the display's delay, the hand
is drawn where it was read (or where a trained predictor says it is going), and a swing is a lunge that reaches the ball
exactly when the picture shows the contact.  Nothing here changes the game: it only decides what to draw.
"""

import dataclasses

from pingpong import ballclock, contact, glide
from pingpong import latency as latency_mod
from pingpong import pd, physics, stage

S = 1_000_000_000
CPU_SWING_S = 0.3          # how long the computer's paddle takes to hit the ball (the HUD animates it)
CPU_RECOVER_S = 0.6        # ... and to drift back to the middle after a return
WAIT_REACH = 1.5           # a hand within this many radii of the ball is taken to be going to hit it, until the game says it did not


class View:
    def __init__(self, game, lat, poses, hand_model=None):
        self.game, self.latency, self.poses = game, lat, poses
        self.hand_model = hand_model                     # a trained posemodel.HandPredictor, or None: the hand is drawn as read
        self.paddle_angle = 0.0                          # degrees: how far the hub is turned in the hand (set each frame)
        self._stroke = None
        self.cpu_swing_ns = None                         # when the computer last hit the ball
        self._glide = glide.Glide(lat.glide_s)           # the hand as drawn: smoothed between the camera's readings
        self._ball_clock = ballclock.BallClock()         # when, on the game's timeline, the picture shows the ball (see ballclock.py)
        self._was_held = False

    # --- time -----------------------------------------------------------------------------------------------------
    def view_ns(self, now):
        """The moment the frame drawn at `now` will be SEEN (a frozen game stays frozen)."""
        g = self.game
        if g.paused and g.paused_at_ns is not None:
            return g.paused_at_ns
        return now + round(self.latency.view_ahead_s * S)

    # --- your paddle ---------------------------------------------------------------------------------------------------
    def hand(self, now):
        """The hand as it is drawn at `now`: where it is read (or predicted), smoothed between the camera's readings."""
        self._glide.tau_s = self.latency.glide_s
        return self._glide.update(now, latency_mod.predict_hand(self.poses, now, self.latency, model=self.hand_model))

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

    def start_stroke(self, events, now, swing=True):
        """The paddle lunges to the ball it hit; a swing that hit nothing (swing=True) lunges forward at nothing."""
        hit = next((e for e in events if e.kind in ("hit", "fault") and "contact" in e.data), None)
        if hit is not None:
            self._stroke = stage.hit_stroke(now, hit.data["contact_ns"], hit.data["contact"])
            return
        rest = self.rest(now) if swing else None
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
    def ball(self, view, now=None):
        """(x, y, z): your ball once it has left your paddle, otherwise the one coming at you, flying on past you if it
        is not hit (it is seen to go by), and None while there is no ball.  `now`: the game's clock for this picture (the hand
        is looked up at it); by default view less the display's lead.

        The ball is shown at the moment `ballclock` says, which is `view` except at a junction (the hand meeting the ball, the
        ball being let go, the computer meeting it): there it waits where the paddle is until the game has said what happens."""
        g = self.game
        if g.phase in ("LOBBY", "COUNTDOWN"):
            self._ball_clock.reset()
            self._was_held = False
            return None
        out, inc, held = g.outgoing_leg, g.incoming_leg, g.held
        floor = None
        if held is not None:
            pin = held[0]                                              # on the paddle until it is let go
        elif out is not None:
            pin = out.arrival_ns if self._awaiting_return(out) else None
            floor = out.t0_ns if self._was_held else None             # let go: it starts from the paddle, however long it waited
        else:
            pin = None if inc is None else self._paddle_plane_ns(view - round(self.latency.view_ahead_s * S) if now is None else now)
        self._was_held = held is not None
        at = self._ball_clock.tick(view, pin, floor)
        if held is not None and at >= held[0] and (out is None or at < out.t0_ns):
            return held[1]                                             # the ball sits on your paddle until it is let go
        if out is not None and at >= out.t0_ns:
            limit = out.arrival_ns + round(physics.FLY_ON_S * S) if out.terminal == "arrive" else out.end_ns
            return out.position(at) if at <= limit else None           # a ball nobody returns is seen to go on by
        if inc is not None and at <= inc.arrival_ns + round(physics.FLY_ON_S * S):
            return inc.position(at)
        return None

    def _awaiting_return(self, out):
        """Your ball has reached the other end and the answer (the computer's, or the friend's) has not come yet."""
        g = self.game
        if out.terminal != "arrive":
            return False
        if g.remote is not None:
            return g.remote.awaiting
        return g.phase == "RALLY" and g.next_cpu_contact_ns is not None

    def _paddle_plane_ns(self, now):
        """When the incoming ball's depth reaches the paddle's, while its hit is still undecided and the hand is level with it
        (contact mode, where the game decides it); None when there is nothing to wait for."""
        g = self.game
        if g.hit_mode != "contact" or g.phase != "RALLY" or g.incoming is None or g.paused or g.ball_passed:
            return None
        hand = self.hand(now)
        if hand is None:
            return None
        box, leg = g.judge.box, g.incoming_leg
        t_plane = leg.time_at_z(stage.rest_z(box, hand[1]))
        if abs(hand[0] - contact.ball_u(box, leg.position(t_plane)[0])) > WAIT_REACH * g.incoming.level.radius_sw:
            return None
        return t_plane

    # --- the computer's paddle ----------------------------------------------------------------------------------------------
    def cpu_swing(self, view):
        """How far through its stroke the computer's paddle is (0..1), None when it is not hitting.  The stroke is timed
        to meet the ball exactly as it leaves (the middle of the swing), so it starts before the serve or the return."""
        g, half = self.game, CPU_SWING_S / 2 * S
        waited_for = g.outgoing_leg.arrival_ns if g.remote is not None and g.remote.awaiting and g.outgoing_leg is not None else None
        contacts = [t for t in (self.cpu_swing_ns, waited_for if g.remote is not None else g.next_cpu_contact_ns) if t is not None]
        near = [t for t in contacts if abs(view - t) < half]
        if not near:
            return None
        return 0.5 + (view - min(near, key=lambda t: abs(view - t))) / (CPU_SWING_S * S)

    def cpu_x(self, view):
        """The computer's paddle: after its return it stands where it hit and drifts back to the middle; while it chases
        your shot it moves to where the ball will land (always arrives in Rally, can fall short in Match)."""
        g, leg = self.game, self.game.outgoing_leg
        if g.remote is not None:
            return g.remote.paddle_x                                 # the other person's paddle, where they hold it
        if g.phase == "RALLY" and g.incoming is not None and g.incoming_leg is not None:
            back = g.incoming_leg
            return back.x_start * max(0.0, 1.0 - max(0.0, (view - back.t0_ns) / S) / CPU_RECOVER_S)
        if g.phase != "RALLY" or leg is None:
            return 0.0
        level = g.level
        if g.mode != "match":                          # Survival never misses: the drawn paddle always gets there
            level = dataclasses.replace(level, cpu_speed_ms=50.0, tau_s=min(level.tau_s, 0.25 * leg.flight_s))
        return pd.paddle_x(level, leg.x_end, 0.0, max(0.0, (view - leg.t0_ns) / S))
