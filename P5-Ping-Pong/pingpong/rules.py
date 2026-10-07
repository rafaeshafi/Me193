"""GameCore: the rally lifecycle (LOBBY > COUNTDOWN > RALLY > POINT_OVER / MATCH_OVER).

Pure game logic driven by three inputs -- start(), on_swing(), tick() -- all with an
explicit `now_ns`, so it runs identically on the real sensors, in --fake mode and
in tests.  It owns no threads, no hardware and no I/O except calling the injected
publisher when the score changes.

  SURVIVAL  the CPU never misses; the first player miss/fault ends the game; the
            score is the streak (the record is what goes to MQTT)
  MATCH     rally scoring to `target_points` (win by 2 from 11), the CPU serves every
            point (receive-only) and misses with a probability from policy.py
"""

import dataclasses

from pingpong import levels, physics
from pingpong import shot as shotmod
from pingpong.events import GameEvent
from pingpong.judge import BallWindow, pose_at
from pingpong.policy import reach_deficit_m

S = 1_000_000_000
BETWEEN_RALLIES = ("LOBBY", "POINT_OVER", "MATCH_OVER")


class GameCore:
    def __init__(self, *, judge, tracker, policy, publisher=None, level=None, mode="survival",
                 target_points=7, omega_lo=300.0, omega_hi=1200.0, spin_probs_fn=None,
                 countdown_s=3.0, point_pause_s=2.0):
        self.judge, self.tracker, self.policy, self.publisher = judge, tracker, policy, publisher
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
        self.started_at_ns = None            # exactly when start() was last called (a replay needs it)

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

    def start(self, now_ns):
        if self.phase not in ("LOBBY", "MATCH_OVER") or self.paused:
            return False
        if self.phase == "MATCH_OVER":
            self.player_points = self.cpu_points = 0
        self.phase = "COUNTDOWN"
        self.started_at_ns = now_ns
        self._serve_at = now_ns + round(self.countdown_s * S)
        return True

    def tick(self, now_ns, data_ns=None):
        """Settle everything that is due by now_ns (a long frame stall may owe several transitions).

        data_ns = arrival time of the newest IMU sample.  A miss means "no swing", which is only
        a fact once the IMU stream has reached the deadline: if the hub goes quiet around the
        swing window the wall clock alone must not call the ball a miss.
        """
        if self.paused:
            return []
        data_ns = now_ns if data_ns is None else min(now_ns, data_ns)
        events = []
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
            if self._cpu_at is not None and now_ns >= self._cpu_at:
                return self._cpu_response(self._cpu_at)
            if self.incoming is not None and data_ns > self.judge.miss_deadline_ns(self.incoming):
                self._observe(1.0, True)                                  # the player failed to return the ball
                return [GameEvent("miss", now_ns, {"ball_id": self.incoming.ball_id})] + self._end_rally("miss", now_ns)
        return []

    @property
    def paused(self):
        return bool(self._pause_reasons)

    @property
    def pause_reasons(self):
        return set(self._pause_reasons)

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
        if self.phase != "RALLY" or self.incoming is None or self.paused:
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
        survival = self.mode == "survival"
        plan = self.policy.serve(self.level, self.s_prev, self.tracker.streak, self.player_a, survival)
        leg = physics.plan_leg(t0_ns, plan.v, x_start, plan.aim_ab, plan.topspin, plan.sidespin)
        self._ball_id += 1
        self.incoming = BallWindow(self._ball_id, leg.arrival_ns, plan.aim_ab, self.level, leg)
        self.incoming_leg, self.outgoing_leg = leg, None
        self.phase, self._cpu_at = "RALLY", None
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
        ball, leg_in = self.incoming, self.incoming_leg
        contact_ns = verdict.contact_ns or max(now_ns, ball.t_c_ns)
        # the paddle meets the ball where the ball IS at the contact: the return leaves from that point
        contact = leg_in.position(contact_ns) if leg_in is not None else (
            physics.x_of_a(ball.aim_ab[0]), physics.STRIKE_Y_M, physics.HIT_Z_M)
        at = pose_at(pose_samples, swing.t_ns, min_conf=self.judge.min_conf) or (pose_samples[-1] if pose_samples else None)
        paddle_a = self.judge.box.to_ab(at.u, at.v)[0] if at else 0.5          # where the hand was AT the impact
        # the balls come in a level's share of the box, so the hand's lateral range is that share too: the aim
        # spreads it back over the whole table (wide returns are how a point is won)
        aim_a = min(1.0, max(0.0, 0.5 + (paddle_a - 0.5) / self.level.reach))
        probs = self.spin_probs_fn(swing.feat) if self.spin_probs_fn else None
        sp = shotmod.make(w_pk=swing.w_pk, omega_lo=self.omega_lo, omega_hi=self.omega_hi,
                          d_min_sw=verdict.d_min_sw, e_s=verdict.e_s, level=self.level,
                          paddle_a=aim_a, spin_probs=probs)
        self.s_prev = shotmod.swing_strength(swing.w_pk, self.omega_lo, self.omega_hi)
        self.player_a, self.incoming = paddle_a, None
        self.outgoing_leg = physics.plan_return(
            contact_ns, sp.v_out, contact, (0.5 + sp.aim_a / 1.6, 0.5), topspin=sp.T, sidespin=sp.S, fault=sp.fault)
        data = {"label": sp.label, "v_out": sp.v_out, "kmh": shotmod.kmh(sp.v_out), "topspin": sp.T,
                "sidespin": sp.S, "q_total": sp.q_total, "gates": verdict.gates, "e_s": verdict.e_s,
                "contact_ns": contact_ns, "contact": contact}
        self._observe(1.0 if sp.fault else 0.5 * (1.0 - sp.q_total), bool(sp.fault))
        if sp.fault:
            events = [GameEvent("fault", now_ns, dict(data, fault=sp.fault))]
            return events + self._end_rally("fault", now_ns)
        counted = self.tracker.on_valid_hit(ball.ball_id)
        if counted and self.publisher:
            self.publisher.update(self.tracker.value())
        events = [GameEvent("hit", now_ns, dict(data, streak=self.tracker.streak))]
        # A record is announced once per rally, when the streak passes the best that stood when the
        # rally began -- never in a first rally with nothing to beat (that would buzz every hit).
        if counted and self._record_before > 0 and self.tracker.streak == self._record_before + 1:
            events.append(GameEvent("record", now_ns, {"value": self.tracker.record}))
        self._out_shot, self._cpu_at = sp, self.outgoing_leg.arrival_ns
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
        self.incoming = self._cpu_at = None
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
