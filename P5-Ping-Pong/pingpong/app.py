"""Session: wires GameCore to feedback (haptics), HUD state, tags and the score publisher.

Pure orchestration with injected I/O, so the same code runs on the real sensors,
in --fake mode and in tests.  run_scripted() is a scripted fake player on a fake
clock -- the automated version of "play.py --fake reaches 10 hits".
"""

import math
import random
from collections import deque

from pingpong import feedback, holdstart, levels, uistate
from pingpong import latency as latency_mod
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose, SwingEvent
from pingpong.flow import Flow
from pingpong.hud import HudState
from pingpong.judge import HitJudge
from pingpong.mqtt_pub import ScorePublisher
from pingpong.paddle import ReachBox
from pingpong.policy import CpuPolicy
from pingpong.rules import GameCore
from pingpong.scoring import ScoreTracker
from pingpong.view import View

S = 1_000_000_000
DEFAULT_BOX = ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5)
FLASH = {"perfect": ((255, 255, 255), 0.25), "good": ((0, 200, 0), 0.18), "early": ((0, 140, 255), 0.2),
         "late": ((0, 140, 255), 0.2), "fault": ((0, 0, 255), 0.35)}


class Session:
    def __init__(self, game, clock, actuator=None, mqtt_status=None, hub_status=None, latency=None, hand_model=None,
                 hold_start=None, flow=None):
        self.game, self.clock, self.actuator = game, clock, actuator
        self.hold_start = hold_start             # a holdstart.HoldStart: the game also starts when the hub is held on the START button
        self.flow = flow                         # a flow.Flow: the intro, the title and the choices of game and opponent, and the results
        self._hand_ab = None                     # where the hand points in the reach box, for that button
        self._t0_ns = clock.now_ns()             # the faces' animation clock starts here
        self._last_summary, self._record_game = None, False       # how the last game went, for the results screen
        self._cpu_mood, self._mood_until, self._point_for = "happy", 0, ""
        self._mqtt_status = mqtt_status or (lambda: "off")
        self._hub_status = hub_status or (lambda: "ok")
        self.latency = latency or latency_mod.Latency.from_config()
        self.poses = deque(maxlen=90)
        self.view = View(game, self.latency, self.poses, hand_model)
        self.xray = False
        self._gates, self._last_kmh, self._last_label, self._spin = (), None, "", ""
        self._message, self._message_until = "", 0
        self._notice = self._soft_notice = ""
        self.audio = None                        # pingpong.audio.Audio (optional)
        self._last_digit = None
        self.player = ""                         # the player's name (highlighted on the leaderboard)
        self.on_game_over = None                 # callback(summary dict) when a game or match ends
        self.leaderboard_fn = None               # () -> ((name, score), ...) for the end screen
        self._stats_key, self._stats = object(), {}
        self._flash, self._flash_until = None, 0
        self._sounds, self._last_tick_ns = [], None      # (due ns, name) waiting for the moment the picture shows them

    @property
    def paddle_angle(self):
        return self.view.paddle_angle

    @paddle_angle.setter
    def paddle_angle(self, degrees):
        self.view.paddle_angle = degrees

    def set_notice(self, text, soft=False):
        """A standing message for the lobby (e.g. "UNCALIBRATED"); pauses and event banners win over it.

        A soft one is only a hint: any real notice (what the broker holds, UNCALIBRATED) takes its place."""
        if soft:
            self._soft_notice = text
        else:
            self._notice = text

    def has_notice(self):
        return bool(self._notice)

    def bind_status(self, hub=None, mqtt=None):
        """Point the HUD's HUB / MQTT indicators at live sources (the rig exists after the session)."""
        if hub is not None:
            self._hub_status = hub
        if mqtt is not None:
            self._mqtt_status = mqtt

    # --- inputs ------------------------------------------------------------------------------
    def on_pose(self, pose):
        """A hand reading; in contact mode a hand moving into the ball is a hit (the events say so)."""
        self.poses.append(pose)
        if self.game.hit_mode != "contact":
            return []
        events = self.game.on_pose(pose, list(self.poses), self.clock.now_ns())
        self._absorb(events)
        return events

    def on_start(self):
        return self.game.start(self.clock.now_ns())

    def on_tag(self, tag):
        if self.flow is not None:
            actions = self.flow.on_tag(tag.role, tag.value, self.clock.now_ns())
            if actions is not None:
                self.apply(actions)
                return
        if tag.role == "START":
            self.on_start()
        elif tag.role == "LEVEL":
            level = levels.level_for_tag(tag.value)
            if level is not None:
                self.game.set_level(level)

    def on_swing(self, swing):
        if self.game.hit_mode == "contact":
            return []                          # a swing is not what hits the ball here; the wrist's flick is what spins it
        now = self.clock.now_ns()
        events = self.game.on_swing(swing, list(self.poses), now)
        self._absorb(events)
        self.view.start_stroke(events, now)                 # the paddle lunges (a hit, or a swing at nothing)
        return events

    def tick(self, data_ns=None):
        now = self.clock.now_ns()
        events = self.game.tick(now, data_ns)
        self._absorb(events)
        if self.game.hit_mode == "contact":
            self.view.start_stroke(events, now, swing=False)       # the ball is let go: the paddle meets it
        self._countdown_sounds(events)
        self._bounce_sound(now)
        self._flush_sounds(now)
        self._last_tick_ns = now
        self._hold_to_start(now)
        self._flow_tick(now)
        return events

    def _flow_tick(self, now):
        """The screens round the game: the hand is their pointer, and what they decide is carried out here."""
        if self.flow is None:
            return
        self._hand_ab = holdstart.hand_ab(self.game.judge.box, self.poses, now)
        self.apply(self.flow.update(now, self._hand_ab, self.game.phase))

    def apply(self, actions):
        """Carry out what the flow asked for: start a game, make a sound, buzz the hub, change the music."""
        for kind, value in actions or ():
            if kind == "start":
                self._start_game(*value)
            elif kind == "sound" and self.audio is not None:
                self.audio.play(value)
            elif kind == "haptic" and self.actuator is not None:
                self.actuator.submit(value)
            elif kind == "music" and self.audio is not None:
                self.audio.play_music(value)

    def _start_game(self, level_tag, mode):
        self.game.set_level(levels.LEVELS[level_tag])
        self.game.set_mode(mode)
        self.apply(self.flow.started(self.on_start()))

    def _hold_to_start(self, now):
        """The hub held on the START button for long enough starts the game, like the key and the card."""
        if self.hold_start is None or self.flow is not None:
            return
        self._hand_ab = holdstart.hand_ab(self.game.judge.box, self.poses, now)
        if self.hold_start.update(now, self._hand_ab, self.game.phase in holdstart.PHASES):
            self.on_start()

    def _countdown_sounds(self, events):
        """A tick per countdown digit (3-2-1), and a "go" (heard, and flashed on the screen) when the first ball is served."""
        remaining = self.game.seconds_to_serve(self.clock.now_ns())
        digit = None if remaining is None else max(1, math.ceil(remaining))
        went = digit is None and self._last_digit is not None and any(e.kind == "serve" for e in events)
        if went:
            self._set_message("GO!", self.clock.now_ns(), 0.8)
        if self.audio is not None:
            if digit is not None and digit != self._last_digit:
                self.audio.play("tick")
            elif went:
                self.audio.play("go")
        self._last_digit = digit

    # --- events -> feedback + HUD memory ---------------------------------------------------------
    def _game_stats(self):
        """Per-game counters, reset whenever a new game is started."""
        if self._stats_key != self.game.started_at_ns:
            self._stats_key = self.game.started_at_ns
            self._stats = {"hits": 0, "misses": 0, "faults": 0, "max_kmh": 0.0, "best_streak": 0}
            self._last_summary, self._record_game, self._cpu_mood, self._mood_until = None, False, "happy", 0
        return self._stats

    def game_stats(self):
        """Counters of the current game: hits, misses, faults, max_kmh, best_streak (a copy)."""
        return dict(self._game_stats())

    def _summary(self, e):
        g, st = self.game, self._game_stats()
        return {"mode": g.mode, "level": g.level.name, "target": g.target_points, "streak": st["best_streak"],
                "record": g.tracker.record, "player_points": g.player_points, "cpu_points": g.cpu_points,
                "winner": e.data.get("winner"), "hits": st["hits"], "misses": st["misses"], "faults": st["faults"],
                "max_kmh": st["max_kmh"], "duration_s": (e.t_ns - (g.started_at_ns or e.t_ns)) / S,
                "started_at_ns": g.started_at_ns}

    def _queue_sounds(self, events, now):
        """A hit's sound is due when the picture shows the contact (less the time the speakers take); the rest are due now."""
        for e in events:
            name = self.audio.sound_for(e, self.game.level)
            if name:
                due = now
                if e.kind == "hit" and "contact_ns" in e.data:
                    due = max(now, e.data["contact_ns"] - round(self.latency.audio_s * S))
                self._sounds.append((due, name))
        self._flush_sounds(now)

    def _flush_sounds(self, now):
        if self.audio is None or not self._sounds:
            return
        ready = [name for due, name in self._sounds if due <= now]
        self._sounds = [(due, name) for due, name in self._sounds if due > now]
        for name in ready:
            self.audio.play(name)

    def _bounce_sound(self, now):
        """The bounce is heard as it is seen: sent early by the speakers' delay."""
        if self.audio is None or self._last_tick_ns is None or self.game.paused:
            return
        lead = round(self.latency.audio_s * S)
        for leg in (self.game.incoming_leg, self.game.outgoing_leg):
            if leg is not None and leg.p_land <= 1.0 and self._last_tick_ns < leg.bounce_ns - lead <= now:
                self.audio.play("bounce")

    def _absorb(self, events):
        now = self.clock.now_ns()
        if self.actuator is not None:
            feedback.play(events, self.game.level, self.actuator, latency=self.latency, now_ns=now)
        if self.audio is not None:
            self._queue_sounds(events, now)
        st = self._game_stats()
        for e in events:
            if e.kind == "hit":
                st["hits"] += 1
                st["max_kmh"] = max(st["max_kmh"], e.data["kmh"])
                st["best_streak"] = max(st["best_streak"], e.data["streak"])
            elif e.kind == "miss":
                st["misses"] += 1
            elif e.kind == "fault":
                st["faults"] += 1
            elif e.kind in ("game_over", "match_over"):
                self._last_summary = self._summary(e)
                self._set_mood("sad" if e.data.get("winner") == "player" else "cheer", now, 600)
                if self.on_game_over is not None:
                    self.on_game_over(self._last_summary)
            if e.kind == "serve":
                self.view.cpu_swing_ns = e.t_ns              # the computer hits the ball: its paddle swings
            if e.kind == "verdict":
                self._gates = e.data["verdict"].gates
            elif e.kind == "hit":
                self._last_kmh, self._last_label = e.data["kmh"], e.data["label"]
                self._spin = _spin_text(e.data["topspin"], e.data["sidespin"])
                self._set_flash(FLASH.get(e.data["label"]), now)
                if e.data["label"] == "perfect":
                    self._set_mood("surprised", now, 1.2)
            elif e.kind == "point":
                self._point_for = e.data["scorer"]
                self._set_mood("sad" if self._point_for == "player" else "cheer", now, 2.5)
            elif e.kind == "record":
                self._record_game = True
                self._set_message("NEW RECORD", now, 1.5)
            elif e.kind == "fault":
                self._last_kmh, self._last_label = e.data["kmh"], "fault " + e.data["fault"]
                self._set_message(f"FAULT: {e.data['fault'].upper()}", now, 1.5)
                self._set_flash(FLASH["fault"], now)
            elif e.kind == "miss":
                self._set_message("MISSED", now, 1.5)
                self._set_flash(FLASH["fault"], now)

    def _set_mood(self, mood, now, seconds):
        self._cpu_mood, self._mood_until = mood, now + round(seconds * S)

    def _set_message(self, text, now, seconds):
        self._message, self._message_until = text, now + round(seconds * S)

    def _set_flash(self, flash, now):
        if flash:
            self._flash, self._flash_until = flash, now + round(0.15 * S)

    # --- HUD -------------------------------------------------------------------------------------------
    def hud_state(self, leaderboard=()):
        g, now = self.game, self.clock.now_ns()
        if not leaderboard and g.phase == "MATCH_OVER" and self.leaderboard_fn is not None:
            leaderboard = self.leaderboard_fn()
        remaining = g.seconds_to_serve(now)
        v = self.view
        view = v.view_ns(now)
        paddle, rest = v.paddle(now, view)
        button = cursor = None
        if self.hold_start is not None and self.flow is None and g.phase in holdstart.PHASES:
            button, cursor = (self.hold_start.progress(), self.hold_start.inside), self._hand_ab
        screen, ui_state, results = "GAME", None, None
        if self.flow is not None and self.flow.active:
            screen, ui_state = self.flow.screen, self.flow.ui_state(now, self._hand_ab)
            if screen == "RESULTS" and self._last_summary is not None:
                results = uistate.results_from_summary(self._last_summary, self._record_game)
        digit = None if remaining is None else max(1, math.ceil(remaining))
        return HudState(
            phase=g.phase, mode=g.mode, level_name=g.level.name, streak=g.tracker.streak,
            record=g.tracker.record, player_points=g.player_points, cpu_points=g.cpu_points,
            target=g.target_points, countdown=digit,
            countdown_t=0.0 if digit is None else min(1.0, max(0.0, 1.0 - (remaining - (digit - 1)))),
            ball=v.ball(view), paddle=paddle, rest=rest, paddle_angle=v.paddle_angle, reach_m=v.reach_m(),
            zone=v.zone(now), cpu_x_m=v.cpu_x(view), cpu_swing=v.cpu_swing(view),
            last_kmh=self._last_kmh, last_label=self._last_label, spin_text=self._spin,
            mqtt_status=self._mqtt_status(), hub_status=self._hub_status(),
            message=(self._paused_text() or (self._message if now < self._message_until else "")
                     or ((self._notice or self._soft_notice) if g.phase == "LOBBY" else "")),
            gates=self._gates, show_xray=self.xray,
            flash=self._flash if now < self._flash_until else None, leaderboard=tuple(leaderboard),
            player_name=self.player, start_button=button, cursor=cursor,
            cpu_mood=self._cpu_mood if now < self._mood_until else "happy", anim_t=(now - self._t0_ns) / S,
            point_for=self._point_for if g.phase == "POINT_OVER" else "", screen=screen, ui=ui_state, results=results)

    def _paused_text(self):
        if not self.game.paused:
            return ""
        return "PAUSED: " + ", ".join(sorted(self.game.pause_reasons)) + " lost"


def _spin_text(top, side):
    parts = []
    if abs(top) > 0.15:
        parts.append("TOPSPIN" if top > 0 else "BACKSPIN")
    if abs(side) > 0.15:
        parts.append("SIDESPIN R" if side > 0 else "SIDESPIN L")
    return " ".join(parts)


def make_session(*, level=1, mode="survival", target=7, clock=None, actuator=None, client=None,
                 source="live", scope="record_session", no_publish=False, seed=1, box=None,
                 omega_lo=300.0, omega_hi=1200.0, t_pk=250.0, spin_probs_fn=None, learner=None, resume=False,
                 latency=None, hand_model=None, hold_start=False, shot_model=None, hit_mode="swing", wrist_frame=None,
                 gyro_window=None, flow=None):
    clock = clock or FakeClock(start_ns=1_000_000_000)
    latency = latency or latency_mod.Latency.from_config()
    box = box or DEFAULT_BOX
    tracker = ScoreTracker(scope=scope)
    publisher = ScorePublisher(client, scope=scope, source=source, no_publish=no_publish, resume=resume) \
        if client else None
    game = GameCore(judge=HitJudge(box, t_pk=t_pk, contact_lag_s=latency.contact_lag_s), tracker=tracker,
                    policy=CpuPolicy(random.Random(seed), learner=learner),
                    publisher=publisher, level=levels.LEVELS[level], mode=mode, target_points=target,
                    omega_lo=omega_lo, omega_hi=omega_hi, spin_probs_fn=spin_probs_fn, shot_model=shot_model,
                    hit_mode=hit_mode, wrist_frame=wrist_frame, gyro_window=gyro_window, imu_delay_s=latency.imu_s)
    if flow is True:
        flow = Flow(intro=True, level_tag=level, mode=mode)
    return Session(game, clock, actuator=actuator, latency=latency, hand_model=hand_model,
                   hold_start=holdstart.HoldStart() if hold_start else None, flow=flow or None,
                   mqtt_status=(lambda: "ok") if client is not None and not no_publish else None)


def play_until(session, stop, w_pk=600.0, dt=0.01, max_sim_s=300.0, feat=None, du=0.0, timing_s=0.0):
    """A scripted player on the session's (fake) clock until stop(session); returns simulated seconds used.

    Perfect by default; du moves the hand off the ball by that many shoulder widths and timing_s swings
    that many seconds off the ball's arrival (both lower the hit quality, so hard swings can fault).
    """
    game, clock = session.game, session.clock
    if game.phase in ("LOBBY", "MATCH_OVER"):
        session.on_start()
    t_begin, swung, step = clock.now_ns(), None, 0
    while not stop(session):
        if (clock.now_ns() - t_begin) / S > max_sim_s:
            raise RuntimeError("scripted player ran out of simulated time")
        clock.advance_s(dt)
        step += 1
        now = clock.now_ns()
        session.tick()
        ball = game.incoming
        if ball is None:
            continue
        if ball.t_c_ns - int(0.5 * S) <= now <= ball.t_c_ns + int(0.05 * S) and step % 3 == 0:
            u, v = game.judge.box.to_uv(ball.aim_ab[0], 0.5)         # across the court, at the nominal depth
            session.on_pose(PaddlePose(t_scene_ns=now, u=u + du, v=v, conf=0.9, hand="right"))
        if now >= ball.t_c_ns and swung != ball.ball_id:
            swung = ball.ball_id
            peak = ball.t_c_ns - round(game.judge.contact_lag_s * S) + round(timing_s * S)   # the stroke ends at t_c
            session.on_swing(SwingEvent(
                kind="IMPACT", t_ns=peak, w_pk=w_pk, dur_ms=150.0, n_reversals=0,
                axis_unit=(1, 0, 0), net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1), clipped=False,
                feat=feat or (0.0,) * 12))
    return (clock.now_ns() - t_begin) / S


def play_until_hits(session, n_hits, w_pk=600.0, dt=0.01, max_sim_s=300.0, feat=None):
    """A scripted perfect player returns balls until n_hits in a row (or the game ends)."""
    return play_until(session, lambda s: s.game.tracker.streak >= n_hits or s.game.phase == "MATCH_OVER",
                      w_pk=w_pk, dt=dt, max_sim_s=max_sim_s, feat=feat)


def run_scripted(n_hits, *, level=1, client=None, source="live", seed=1, mode="survival"):
    """Headless fake run: a perfect scripted player returns n_hits balls."""
    from pingpong.sources_fake import FakeMqttClient

    client = client if client is not None else FakeMqttClient()
    session = make_session(level=level, client=client, source=source, seed=seed, mode=mode)
    seconds = play_until_hits(session, n_hits)
    return {"streak": session.game.tracker.streak, "record": session.game.tracker.record,
            "sim_seconds": seconds, "client": client, "session": session}
