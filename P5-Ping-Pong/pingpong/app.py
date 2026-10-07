"""Session: wires GameCore to feedback (haptics), HUD state, tags and the score publisher.

Pure orchestration with injected I/O, so the same code runs on the real sensors,
in --fake mode and in tests.  run_scripted() is a scripted fake player on a fake
clock -- the automated version of "play.py --fake reaches 10 hits".
"""

import dataclasses
import math
import random
from collections import deque

from pingpong import feedback, levels, pd, physics
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose, SwingEvent
from pingpong.hud import HudState
from pingpong.judge import HitJudge
from pingpong.mqtt_pub import ScorePublisher
from pingpong.paddle import ReachBox
from pingpong.policy import CpuPolicy
from pingpong.rules import GameCore
from pingpong.scoring import ScoreTracker

S = 1_000_000_000
DEFAULT_BOX = ReachBox(u_min=-1.0, u_max=1.0, v_min=-0.5, v_max=0.5)
FLASH = {"perfect": ((255, 255, 255), 0.25), "good": ((0, 200, 0), 0.18), "early": ((0, 140, 255), 0.2),
         "late": ((0, 140, 255), 0.2), "fault": ((0, 0, 255), 0.35)}


class Session:
    def __init__(self, game, clock, actuator=None, mqtt_status=None, hub_status=None):
        self.game, self.clock, self.actuator = game, clock, actuator
        self._mqtt_status = mqtt_status or (lambda: "off")
        self._hub_status = hub_status or (lambda: "ok")
        self.poses = deque(maxlen=90)
        self.paddle_ab = None
        self.xray = False
        self._gates, self._last_kmh, self._last_label, self._spin = (), None, "", ""
        self._message, self._message_until = "", 0
        self._notice = ""
        self.audio = None                        # pingpong.audio.Audio (optional)
        self._last_digit = None
        self.player = ""                         # the player's name (highlighted on the leaderboard)
        self.on_game_over = None                 # callback(summary dict) when a game or match ends
        self.leaderboard_fn = None               # () -> ((name, score), ...) for the end screen
        self._stats_key, self._stats = object(), {}
        self._flash, self._flash_until = None, 0

    def set_notice(self, text):
        """A standing message for the lobby (e.g. "UNCALIBRATED"); pauses and event banners win over it."""
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
        self.poses.append(pose)
        self.paddle_ab = self.game.judge.box.to_ab(pose.u, pose.v)

    def on_start(self):
        return self.game.start(self.clock.now_ns())

    def on_tag(self, tag):
        if tag.role == "START":
            self.on_start()
        elif tag.role == "LEVEL":
            level = levels.level_for_tag(tag.value)
            if level is not None:
                self.game.set_level(level)

    def on_swing(self, swing):
        events = self.game.on_swing(swing, list(self.poses), self.clock.now_ns())
        self._absorb(events)
        return events

    def tick(self, data_ns=None):
        events = self.game.tick(self.clock.now_ns(), data_ns)
        self._absorb(events)
        self._countdown_sounds(events)
        return events

    def _countdown_sounds(self, events):
        """A tick per countdown digit (3-2-1) and a "go" when the first ball is served."""
        if self.audio is None:
            return
        remaining = self.game.seconds_to_serve(self.clock.now_ns())
        digit = None if remaining is None else max(1, math.ceil(remaining))
        if digit is not None and digit != self._last_digit:
            self.audio.play("tick")
        elif digit is None and self._last_digit is not None and any(e.kind == "serve" for e in events):
            self.audio.play("go")
        self._last_digit = digit

    # --- events -> feedback + HUD memory ---------------------------------------------------------
    def _game_stats(self):
        """Per-game counters, reset whenever a new game is started."""
        if self._stats_key != self.game.started_at_ns:
            self._stats_key = self.game.started_at_ns
            self._stats = {"hits": 0, "misses": 0, "faults": 0, "max_kmh": 0.0, "best_streak": 0}
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

    def _absorb(self, events):
        now = self.clock.now_ns()
        if self.actuator is not None:
            feedback.play(events, self.game.level, self.actuator)
        if self.audio is not None:
            self.audio.play_events(events, self.game.level)
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
            elif e.kind in ("game_over", "match_over") and self.on_game_over is not None:
                self.on_game_over(self._summary(e))
            if e.kind == "verdict":
                self._gates = e.data["verdict"].gates
            elif e.kind == "hit":
                self._last_kmh, self._last_label = e.data["kmh"], e.data["label"]
                self._spin = _spin_text(e.data["topspin"], e.data["sidespin"])
                self._set_flash(FLASH.get(e.data["label"]), now)
            elif e.kind == "record":
                self._set_message("NEW RECORD", now, 1.5)
            elif e.kind == "fault":
                self._last_kmh, self._last_label = e.data["kmh"], "fault " + e.data["fault"]
                self._set_message(f"FAULT: {e.data['fault'].upper()}", now, 1.5)
                self._set_flash(FLASH["fault"], now)
            elif e.kind == "miss":
                self._set_message("MISSED", now, 1.5)
                self._set_flash(FLASH["fault"], now)

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
        return HudState(
            phase=g.phase, mode=g.mode, level_name=g.level.name, streak=g.tracker.streak,
            record=g.tracker.record, player_points=g.player_points, cpu_points=g.cpu_points,
            target=g.target_points, countdown=None if remaining is None else max(1, math.ceil(remaining)),
            ball=self._ball(now), cpu_x_m=self._cpu_x(now), paddle_ab=self.paddle_ab,
            arrival_ab=g.incoming.aim_ab if g.incoming is not None else None,
            last_kmh=self._last_kmh, last_label=self._last_label, spin_text=self._spin,
            mqtt_status=self._mqtt_status(), hub_status=self._hub_status(),
            message=(self._paused_text() or (self._message if now < self._message_until else "")
                     or (self._notice if g.phase == "LOBBY" else "")),
            gates=self._gates, show_xray=self.xray,
            flash=self._flash if now < self._flash_until else None, leaderboard=tuple(leaderboard),
            player_name=self.player,
            box_sw=(g.judge.box.u_max - g.judge.box.u_min, g.judge.box.v_max - g.judge.box.v_min),
            radius_sw=g.level.radius_sw, reach=g.level.reach)

    def _paused_text(self):
        if not self.game.paused:
            return ""
        return "PAUSED: " + ", ".join(sorted(self.game.pause_reasons)) + " lost"

    def _cpu_x(self, now):
        """The computer's paddle while it chases your shot (always arrives in Survival, can fall short in Match)."""
        g, leg = self.game, self.game.outgoing_leg
        if g.phase != "RALLY" or leg is None or g.incoming is not None:
            return 0.0
        level = g.level
        if g.mode != "match":                          # Survival never misses: the drawn paddle always gets there
            level = dataclasses.replace(level, cpu_speed_ms=50.0, tau_s=min(level.tau_s, 0.25 * leg.flight_s))
        return pd.paddle_x(level, leg.x_end, 0.0, max(0.0, (now - leg.t0_ns) / S))

    def _ball(self, now):
        g = self.game
        if g.phase != "RALLY":
            return None
        if g.incoming is not None and g.incoming_leg is not None:
            x, p, h = g.incoming_leg.position(now)
            return x, max(0.0, min(1.0, p)), h
        leg = g.outgoing_leg
        if leg is not None and now <= leg.end_ns:
            x, p, h = leg.position(now)
            return x, max(0.0, min(1.0, 1.0 - p)), h
        return None


def _spin_text(top, side):
    parts = []
    if abs(top) > 0.15:
        parts.append("TOPSPIN" if top > 0 else "BACKSPIN")
    if abs(side) > 0.15:
        parts.append("SIDESPIN R" if side > 0 else "SIDESPIN L")
    return " ".join(parts)


def make_session(*, level=1, mode="survival", target=7, clock=None, actuator=None, client=None,
                 source="live", scope="record_session", no_publish=False, seed=1, box=None,
                 omega_lo=300.0, omega_hi=1200.0, t_pk=250.0, spin_probs_fn=None, learner=None, resume=False):
    clock = clock or FakeClock(start_ns=1_000_000_000)
    box = box or DEFAULT_BOX
    tracker = ScoreTracker(scope=scope)
    publisher = ScorePublisher(client, scope=scope, source=source, no_publish=no_publish, resume=resume) \
        if client else None
    game = GameCore(judge=HitJudge(box, t_pk=t_pk), tracker=tracker, policy=CpuPolicy(random.Random(seed), learner=learner),
                    publisher=publisher, level=levels.LEVELS[level], mode=mode, target_points=target,
                    omega_lo=omega_lo, omega_hi=omega_hi, spin_probs_fn=spin_probs_fn)
    return Session(game, clock, actuator=actuator,
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
            u, v = game.judge.box.to_uv(*ball.aim_ab)
            session.on_pose(PaddlePose(t_scene_ns=now, u=u + du, v=v, conf=0.9, hand="right"))
        if now >= ball.t_c_ns and swung != ball.ball_id:
            swung = ball.ball_id
            session.on_swing(SwingEvent(
                kind="IMPACT", t_ns=ball.t_c_ns + round(timing_s * S), w_pk=w_pk, dur_ms=150.0, n_reversals=0,
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
