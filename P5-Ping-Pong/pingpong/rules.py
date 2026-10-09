"""GameCore: the rally lifecycle (LOBBY > COUNTDOWN > RALLY > POINT_OVER / MATCH_OVER).

Pure game logic driven by four inputs -- start(), on_swing(), on_pose(), tick() -- all with an
explicit `now_ns`, so it runs identically on the real sensors, in --fake mode and
in tests.  It owns no threads, no hardware and no I/O except calling the injected
publisher when the score changes.

  hit_mode "swing"    a swing the IMU detects meets the ball (the judge's gates J1-J6 decide)
  hit_mode "contact"  the hand moving into the ball is the hit (contact.py); the ball stays on the paddle for a moment so the
                      wrist's flick can be read, and leaves with the spin the flick gave it (flick.py)

  SURVIVAL  the CPU never misses; the first player miss/fault ends the game; the
            score is the streak (the record is what goes to MQTT)
  MATCH     rally scoring to `target_points` (win by 2 from 11), the CPU serves every
            point (receive-only) and misses with a probability from policy.py
"""

import dataclasses
import math

from pingpong import contact, flick, levels, physics, stage, strokepath
from pingpong import shot as shotmod
from pingpong.events import GameEvent, GateResult, Verdict
from pingpong.judge import BallWindow, pose_at
from pingpong.policy import reach_deficit_m
from pingpong.scoring import ScoreTracker

S = 1_000_000_000
BETWEEN_RALLIES = ("LOBBY", "POINT_OVER", "MATCH_OVER")
CONTACT_GRACE_S = 0.15      # a ball is gone when the lowest paddle could not have met it this long ago and the hand was seen since


class GameCore:
    def __init__(self, *, judge, tracker, policy, publisher=None, level=None, mode="survival",
                 target_points=7, omega_lo=300.0, omega_hi=1200.0, spin_probs_fn=None,
                 countdown_s=3.0, point_pause_s=2.0, shot_model=None, hit_mode="swing", wrist_frame=None,
                 gyro_window=None, imu_delay_s=0.04):
        self.judge, self.tracker, self.policy, self.publisher = judge, tracker, policy, publisher
        self.hit_mode = hit_mode
        self.wrist_frame = wrist_frame        # the hub's up / forward / right (flick.WristFrame); None: no spin from the wrist
        self.gyro_window = gyro_window        # (lo_ns, hi_ns) -> [(arrival ns, gyro dps)]: what the hub felt around a contact
        self.imu_delay_s = imu_delay_s        # the hub's samples are stamped when they arrive, this long after they happened
        self.shot_model = shot_model or strokepath.ShotModel()
        self.level_overrides = {}                      # --set level.X=v: applied to whichever level is chosen
        if publisher is not None:
            publisher.on_resume = tracker.seed_record      # --resume: the retained best is where this run starts
        self.level = level or levels.LEVELS[1]
        self.mode, self.target_points = mode, target_points
        self.omega_lo, self.omega_hi, self.spin_probs_fn = omega_lo, omega_hi, spin_probs_fn
        self.countdown_s, self.point_pause_s = countdown_s, point_pause_s
        self.phase = "LOBBY"
        self.player_points = self.cpu_points = 0
        self.incoming = self.incoming_leg = self.outgoing_leg = None
        self.point_over_until_ns = None
        self.s_prev, self.player_a = 0.5, 0.5
        self._ball_id = 0
        self._serve_at = self._cpu_at = self._out_shot = None
        self._record_before = tracker.record
        self._pause_reasons, self._paused_at = set(), None
        self.started_at_ns = None            # exactly when start() was last called (a replay needs it) ...
        self.started_countdown_s = countdown_s       # ... and how long its countdown was
        self.held = None                     # (when, (x, y, z)): the ball sitting on your paddle until it is let go
        self.ball_passed = False             # contact mode: the ball's depth met the paddle's, the hand was not level, it goes by
        self.remote = None                   # versus.Remote: the opponent is a person at another laptop, not the computer
        self._hand = self._pending = self._pose_horizon_ns = None

    @property
    def shot_model(self):
        return self._shot_model

    @shot_model.setter
    def shot_model(self, model):
        """How the path of the hand and the flick of the wrist shape a return (`--set shot.k_aim=...`); what the player's
        strokes usually turn the hub by is learnt afresh with it."""
        self._shot_model = model
        self._flick = flick.FlickBaseline(model.flick_warmup)

    # --- inputs ------------------------------------------------------------------------
    def set_level(self, level):
        if self.phase not in BETWEEN_RALLIES:
            return False
        self.level = dataclasses.replace(level, **self.level_overrides) if self.level_overrides else level
        return True

    def set_mode(self, mode):
        if self.phase not in ("LOBBY", "MATCH_OVER"):
            return False
        self.mode = mode
        return True

    def start(self, now_ns, countdown_s=None):
        """Begin a game: the countdown (countdown_s seconds, else the game's usual) and the first serve."""
        if self.phase not in ("LOBBY", "MATCH_OVER") or self.paused:
            return False
        if self.phase == "MATCH_OVER":
            self.player_points = self.cpu_points = 0
        self.phase = "COUNTDOWN"
        if self.remote is not None:
            self.remote.on_start(now_ns)
        self.started_at_ns = now_ns
        self._hand = self._pending = self.held = None
        self.started_countdown_s = self.countdown_s if countdown_s is None else countdown_s
        self._serve_at = now_ns + round(self.started_countdown_s * S)
        return True

    def tick(self, now_ns, data_ns=None):
        """Settle everything that is due by now_ns (a long frame stall may owe several transitions).

        data_ns = arrival time of the newest IMU sample.  A miss means "no swing", which is only
        a fact once the IMU stream has reached the deadline: if the hub goes quiet around the
        swing window the wall clock alone must not call the ball a miss.
        """
        events = [] if self.remote is None else self.remote.step(self, now_ns)          # heard even while the game waits for them
        if self.paused:
            return events
        data_ns = now_ns if data_ns is None else min(now_ns, data_ns)
        for _ in range(16):
            step = self._advance(now_ns, data_ns)
            if not step:
                break
            events += step
        return events

    def _advance(self, now_ns, data_ns):
        if self.phase == "COUNTDOWN" and now_ns >= self._serve_at:
            return self._serve(self._serve_at)
        if self.phase == "POINT_OVER" and now_ns >= self.point_over_until_ns:
            return self._serve(self.point_over_until_ns)
        if self.phase == "RALLY":
            if self._pending is not None and now_ns >= self._pending["launch_ns"]:
                return self._finish_contact(now_ns)
            if self._cpu_at is not None and now_ns >= self._cpu_at:
                return self._cpu_response(self._cpu_at)
            if self.incoming is not None and self._gone(now_ns, data_ns):
                self._observe(1.0, True)                                  # the player failed to return the ball
                events = [GameEvent("miss", now_ns, {"ball_id": self.incoming.ball_id})] + self._end_rally("miss", now_ns)
                if self.remote is not None:
                    self.remote.sent_miss(self, "miss", now_ns)
                return events
        return []

    def _gone(self, now_ns, data_ns):
        """No paddle position could still meet the ball.  A swing is only a fact once the IMU stream has reached the
        deadline; a hand into the ball only once the poses have gone past the lowest paddle's depth."""
        if self.hit_mode == "contact":
            by = self.incoming_leg.time_at_z(stage.Z_REST_MIN)
            return (self._pose_horizon_ns or 0) >= by and now_ns > by + round(CONTACT_GRACE_S * S)
        return data_ns > self.judge.miss_deadline_ns(self.incoming)

    @property
    def paused(self):
        return bool(self._pause_reasons)

    @property
    def pause_reasons(self):
        return set(self._pause_reasons)

    @property
    def next_cpu_contact_ns(self):
        """When the computer next hits the ball: the serve after the countdown or a point, or its return of your shot."""
        return {"COUNTDOWN": self._serve_at, "POINT_OVER": self.point_over_until_ns, "RALLY": self._cpu_at}.get(self.phase)

    @property
    def paused_at_ns(self):
        """When the freeze began (the picture stays as it was then); None while the game is running."""
        return self._paused_at

    def set_pause(self, reason, active, now_ns):
        """Freeze the game while a sensor is silent (a pause is never a hit or a fault).

        Reasons combine; when the last one clears, every pending deadline moves by the time
        spent paused, so the ball is exactly where it was.
        """
        was_paused = self.paused
        (self._pause_reasons.add if active else self._pause_reasons.discard)(reason)
        if not was_paused and self.paused:
            self._paused_at = now_ns
        elif was_paused and not self.paused:
            self._shift(now_ns - self._paused_at)
            self._paused_at = None

    def _shift(self, delta_ns):
        for name in ("incoming_leg", "outgoing_leg"):
            leg = getattr(self, name)
            if leg is not None:
                setattr(self, name, dataclasses.replace(leg, t0_ns=leg.t0_ns + delta_ns))
        if self._pending is not None:                     # the ball waiting on the paddle waits as long again
            self._pending = dict(self._pending, launch_ns=self._pending["launch_ns"] + delta_ns,
                                 met_ns=self._pending["met_ns"] + delta_ns)
        if self.held is not None:
            self.held = (self.held[0] + delta_ns, self.held[1])
        self._hand = None
        if self.incoming is not None:                     # the ball the judge holds follows its own (shifted) flight
            self.incoming = dataclasses.replace(self.incoming, t_c_ns=self.incoming.t_c_ns + delta_ns,
                                                leg=self.incoming_leg)
        for name in ("_serve_at", "_cpu_at", "point_over_until_ns"):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name) + delta_ns)

    def seconds_to_serve(self, now_ns):
        """Countdown remaining (None outside the countdown)."""
        if self.phase != "COUNTDOWN":
            return None
        return max(0.0, (self._serve_at - now_ns) / S)

    def on_swing(self, swing, pose_samples, now_ns):
        if self.hit_mode == "contact" or self.phase != "RALLY" or self.incoming is None or self.paused:
            return []
        verdict = self.judge.judge(swing, self.incoming, pose_samples, now_ns)
        events = [GameEvent("verdict", now_ns, {"verdict": verdict})]
        if verdict.kind == "REJECTED":
            events.append(GameEvent("rejected", now_ns, {"gates": verdict.gates}))
        elif verdict.kind == "HIT":
            events += self._on_hit(swing, verdict, pose_samples, now_ns)
        return events

    # --- the CPU serves / returns ------------------------------------------------------------
    def _serve(self, t0_ns, x_start=0.0):
        """The computer hits a ball at the player: a serve from the middle, or (x_start) a return from where it met yours."""
        if self.remote is not None:
            return self.remote.serve(self, t0_ns)
        survival = self.mode == "survival"
        plan = self.policy.serve(self.level, self.s_prev, self.tracker.streak, self.player_a, survival)
        leg = physics.plan_leg(t0_ns, plan.v, x_start, plan.aim_ab, plan.topspin, plan.sidespin)
        self._ball_id += 1
        self.incoming = BallWindow(self._ball_id, leg.arrival_ns, plan.aim_ab, self.level, leg)
        self.incoming_leg, self.outgoing_leg = leg, None
        self.phase, self._cpu_at, self._hand, self.ball_passed = "RALLY", None, None, False
        return [GameEvent("serve", t0_ns, {"ball_id": self._ball_id, "v": plan.v, "aim_ab": plan.aim_ab,
                                           "arrival_ns": leg.arrival_ns, "special": plan.special})]

    def _cpu_response(self, t_ns):
        shot, leg = self._out_shot, self.outgoing_leg
        deficit = reach_deficit_m(self.level, leg.x_end, 0.0, leg.flight_s)
        self._cpu_at = None
        if self.policy.returns(self.level, shot.v_out, shot.A, deficit, self.mode == "survival"):
            return self._serve(t_ns, x_start=leg.x_end)
        return self._end_rally("cpu_miss", t_ns)

    # --- the player's swing ----------------------------------------------------------------------
    def _on_hit(self, swing, verdict, pose_samples, now_ns):
        """A swing that met the ball (hit_mode "swing"): the ball leaves where it was at the contact, at the gyro's strength."""
        ball, leg_in = self.incoming, self.incoming_leg
        contact_ns = verdict.contact_ns or max(now_ns, ball.t_c_ns)
        # the paddle meets the ball where the ball IS at the contact: the return leaves from that point
        contact = leg_in.position(contact_ns) if leg_in is not None else (
            physics.x_of_a(ball.aim_ab[0]), physics.STRIKE_Y_M, physics.HIT_Z_M)
        at = pose_at(pose_samples, swing.t_ns, min_conf=self.judge.min_conf) or (pose_samples[-1] if pose_samples else None)
        paddle_a = self.judge.box.to_ab(at.u, at.v)[0] if at else 0.5          # where the hand was AT the impact
        probs = self.spin_probs_fn(swing.feat) if self.spin_probs_fn else None
        return self._launch(
            ball=ball, contact_ns=contact_ns, contact=contact, verdict=verdict, paddle_a=paddle_a, probs=probs, now_ns=now_ns,
            strength=shotmod.swing_strength(swing.w_pk, self.omega_lo, self.omega_hi),
            path=strokepath.hand_path(pose_samples, swing.t_ns), flick_at=contact_ns)

    # --- a hand into the ball (hit_mode "contact") ------------------------------------------------------------------------------
    def on_pose(self, pose, pose_samples, now_ns):
        """A hand reading.  In contact mode this is how a ball is hit: the readings before and after the moment the ball's
        depth met the paddle's say whether the hand was level with it."""
        self._pose_horizon_ns = max(self._pose_horizon_ns or 0, pose.t_scene_ns)
        if self.hit_mode != "contact" or self.phase != "RALLY" or self.incoming is None or self.paused:
            self._hand = None
            return []
        cur = (pose.t_scene_ns, pose.u, pose.v) if pose.conf >= self.judge.min_conf else None
        prev, self._hand = self._hand, cur
        if prev is None or cur is None or cur[0] <= prev[0]:
            return []
        met = contact.crossing(prev, cur, self.incoming_leg, self.judge.box, self.incoming.level.radius_sw)
        return [] if met is None else self._met(met, pose_samples, now_ns)

    def _met(self, met, pose_samples, now_ns):
        """The ball's depth met the paddle's: a hit if the hand was level with it across the table, else it goes by."""
        ball, model = self.incoming, self.shot_model
        radius = ball.level.radius_sw
        reached = GateResult("J1", True, "the ball reached your paddle")
        if not met.hit:
            self.ball_passed = True
            gate = GateResult("J2", False, f"hand {met.d_sw:.2f} SW from the ball when it passed (limit {radius:.2f})")
            verdict = Verdict("REJECTED", 0.0, 0.0, met.d_sw, (reached, gate), met.t_ns)
            return [GameEvent("verdict", now_ns, {"verdict": verdict}), GameEvent("rejected", now_ns, {"gates": verdict.gates})]
        path = strokepath.hand_path(pose_samples, met.t_ns, window_s=(-0.25, 0.0))
        strength = strokepath.hand_strength(path, model)
        speed = 0.0 if path is None else math.hypot(path.vu, path.vv)
        gates = (reached, GateResult("J2", True, f"hand {met.d_sw:.2f} SW from the ball (limit {radius:.2f})"),
                 GateResult("J3", True, f"hand {speed:.1f} SW/s: strength {strength:.2f}"))
        verdict = Verdict("HIT", shotmod.quality(met.d_sw, 0.0, ball.level)[0], 0.0, met.d_sw, gates, met.t_ns)
        paddle_a = self.judge.box.to_ab(met.u, met.v)[0]
        aim_a = min(1.0, max(0.0, 0.5 + (paddle_a - 0.5) / self.level.reach))
        hold_s = max(model.hold_s, model.flick_after_s + self.imu_delay_s + 0.01)     # never shorter than the flick's window needs
        launch_ns = max(now_ns, met.t_ns + round(hold_s * S))
        # the ball sits on the paddle until it is let go; this return only holds it there (the real one is planned then)
        self.outgoing_leg = physics.plan_return(launch_ns, shotmod.out_speed(strength), met.ball, (aim_a, 0.5))
        self.held = (met.t_ns, met.ball)
        self._pending = {"ball": ball, "met_ns": met.t_ns, "launch_ns": launch_ns, "contact": met.ball, "verdict": verdict,
                         "path": path, "strength": strength, "paddle_a": paddle_a}
        self.incoming = None
        return [GameEvent("verdict", now_ns, {"verdict": verdict})]

    def _finish_contact(self, now_ns):
        """The hold is over: read the wrist's flick and let the ball go."""
        p, self._pending, self.held = self._pending, None, None
        return self._launch(ball=p["ball"], contact_ns=p["launch_ns"], contact=p["contact"], verdict=p["verdict"],
                            strength=p["strength"], path=p["path"], paddle_a=p["paddle_a"], probs=None, now_ns=now_ns,
                            flick_at=p["met_ns"], met_ns=p["met_ns"])

    def _read_flick(self, met_ns):
        """How the hub turned around the moment the paddle met the ball: flick.read_flick_detail's dict."""
        if self.gyro_window is None or self.wrist_frame is None:
            return {"rate": None, "dev": None, "topspin": 0.0, "sidespin": 0.0}
        delay = round(self.imu_delay_s * S)
        lo, hi = flick.window_ns(met_ns, delay, self.shot_model)
        return flick.read_flick_detail(self.gyro_window(lo, hi), met_ns, delay, self.shot_model, self.wrist_frame, self._flick)

    # --- the ball leaves the paddle --------------------------------------------------------------------------------------------------
    def _launch(self, *, ball, contact_ns, contact, verdict, strength, path, paddle_a, probs, now_ns, flick_at=None, met_ns=None):
        """Shape the return (speed from the strength, aim and loft from the hand's path, spin from the flick), plan its
        flight from `contact_ns`, and settle the rally: a fault, or a hit that counts."""
        # the balls come in a level's share of the box, so the hand's lateral range is that share too: the aim spreads it
        # back over the whole table (wide returns are how a point is won)
        aim_a = min(1.0, max(0.0, 0.5 + (paddle_a - 0.5) / self.level.reach))
        wrist = self._read_flick(flick_at) if flick_at is not None else {"rate": None, "dev": None, "topspin": 0.0,
                                                                         "sidespin": 0.0}
        stroke = strokepath.shape_return(path, wrist["topspin"], wrist["sidespin"], self.shot_model)
        sp = shotmod.make(w_pk=0.0, omega_lo=self.omega_lo, omega_hi=self.omega_hi, d_min_sw=verdict.d_min_sw,
                          e_s=verdict.e_s, level=self.level, paddle_a=aim_a, spin_probs=probs, stroke=stroke,
                          strength=strength)
        self.s_prev = strength
        self.player_a, self.incoming = paddle_a, None
        if self.remote is None:
            self.outgoing_leg = physics.plan_return(
                contact_ns, sp.v_out, contact, (0.5 + sp.aim_a / 1.6, 0.5), topspin=sp.T, sidespin=sp.S, fault=sp.fault,
                loft_m=stroke.loft_m)
        else:
            sp = self.remote.shape(self, sp, strength)                  # at the match's pace, not the computer's
            self.outgoing_leg = self.remote.plan(self, contact_ns, sp, contact, stroke)
        data = {"label": sp.label, "v_out": sp.v_out, "kmh": shotmod.kmh(sp.v_out), "topspin": sp.T,
                "sidespin": sp.S, "q_total": sp.q_total, "gates": verdict.gates, "e_s": verdict.e_s,
                "contact_ns": contact_ns, "contact": contact, "mode": self.hit_mode,
                "stroke": {"vu": 0.0 if path is None else path.vu, "vv": 0.0 if path is None else path.vv,
                           "aim_shift": stroke.aim_shift, "loft_m": stroke.loft_m, "topspin": stroke.topspin,
                           "sidespin": stroke.sidespin},
                "flick": {"rate": wrist["rate"], "dev": wrist["dev"]}}
        if met_ns is not None:
            data["met_ns"] = met_ns                                   # when the ball met the paddle; contact_ns is when it left it
        self._observe(1.0 if sp.fault else 0.5 * (1.0 - sp.q_total), bool(sp.fault))
        if sp.fault:
            events = [GameEvent("fault", now_ns, dict(data, fault=sp.fault))] + self._end_rally("fault", now_ns)
            if self.remote is not None:
                self.remote.sent_miss(self, "fault", now_ns)
            return events
        counted = self.tracker.on_valid_hit(ball.ball_id)
        if counted and self.publisher:
            self.publisher.update(self.tracker.value())
        events = [GameEvent("hit", now_ns, dict(data, streak=self.tracker.streak))]
        # A record is announced once per rally, when the streak passes the best that stood when the
        # rally began -- never in a first rally with nothing to beat (that would buzz every hit).
        if counted and self.remote is None and self._record_before > 0 and self.tracker.streak == self._record_before + 1:
            events.append(GameEvent("record", now_ns, {"value": self.tracker.record}))
        self._out_shot = sp
        if self.remote is None:
            self._cpu_at = self.outgoing_leg.arrival_ns
        else:
            self.remote.send_hit(self, now_ns)                              # the other person answers when they answer
        return events

    def _observe(self, reward, terminal):
        observe = getattr(self.policy, "observe", None)
        if observe is not None:
            observe(reward, terminal)

    # --- rally / match bookkeeping ----------------------------------------------------------------
    def _end_rally(self, reason, now_ns):
        final = self.tracker.end_rally()
        self._record_before = self.tracker.record
        if self.publisher:
            self.publisher.update(self.tracker.value())      # live_streak publishes the reset
        self.incoming = self._cpu_at = self._pending = self.held = None
        events = [GameEvent("rally_end", now_ns, {"reason": reason, "streak": final,
                                                  "record": self.tracker.record})]
        if self.mode == "survival":
            self.phase = "MATCH_OVER"
            self._end_game()
            events.append(GameEvent("game_over", now_ns, {"streak": final, "record": self.tracker.record,
                                                          "reason": reason}))
            return events
        scorer = "player" if reason == "cpu_miss" else "cpu"
        if scorer == "player":
            self.player_points += 1
        else:
            self.cpu_points += 1
        events.append(GameEvent("point", now_ns, {"scorer": scorer, "player_points": self.player_points,
                                                  "cpu_points": self.cpu_points}))
        winner = self._winner()
        if winner:
            self.phase = "MATCH_OVER"
            self._end_game()
            events.append(GameEvent("match_over", now_ns, {"winner": winner, "player_points": self.player_points,
                                                           "cpu_points": self.cpu_points}))
        else:
            self.phase = "POINT_OVER"
            self.point_over_until_ns = now_ns + round(self.point_pause_s * S)
        return events

    def play_friend(self, remote, target_points):
        """Turn this game into one with another person: their Remote, a tracker of its own and no publisher (the score on the broker
        is for games against the computer, and a friend's rallies must never reach it).  -> what end_friend needs to put it back."""
        saved = (self.tracker, self.publisher, self.target_points)
        self.tracker, self.publisher, self.target_points, self.remote = ScoreTracker(scope=self.tracker.scope), None, target_points, remote
        self._record_before = 0
        return saved

    def end_friend(self, saved):
        """The game against the computer again, with the tracker and the publisher it had."""
        self.tracker, self.publisher, self.target_points = saved
        self.remote, self._record_before = None, self.tracker.record

    def walkover(self, now_ns):
        """The other person is gone: the game ends where it stands, with no winner named and the score as it was."""
        if self.phase in ("LOBBY", "MATCH_OVER"):
            return []
        self.phase = "MATCH_OVER"
        self.incoming = self.incoming_leg = self.outgoing_leg = self._cpu_at = self._pending = self.held = None
        self.tracker.end_rally()
        self._end_game()
        return [GameEvent("match_over", now_ns, {"winner": None, "walkover": True, "player_points": self.player_points,
                                                 "cpu_points": self.cpu_points})]

    def _end_game(self):
        end_game = getattr(self.policy, "end_game", None)
        if end_game is not None:
            end_game()

    def _winner(self):
        p, c, target = self.player_points, self.cpu_points, self.target_points
        lead = 2 if target >= 11 else 1
        if p >= target and p - c >= lead:
            return "player"
        if c >= target and c - p >= lead:
            return "cpu"
        return None
