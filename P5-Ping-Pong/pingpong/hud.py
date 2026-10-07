"""HUD composition: one OpenCV frame from a HudState (large glyphs, readable at 1.8 m).

The x-ray panel lists the judge gates J1-J6 with OK/X and a note, so the screen
explains every decision ("hand 0.9 SW from the ball") -- the policy made visible.
"""

from dataclasses import dataclass, field

from pingpong import canvas, physics

GREEN, RED, AMBER, WHITE, GREY = (80, 220, 80), (70, 70, 240), (40, 170, 255), (255, 255, 255), (170, 170, 170)


@dataclass(frozen=True)
class HudState:
    phase: str = "LOBBY"             # LOBBY | COUNTDOWN | RALLY | POINT_OVER | MATCH_OVER
    mode: str = "survival"
    level_name: str = "Rookie"
    streak: int = 0
    record: int = 0
    player_points: int = 0
    cpu_points: int = 0
    target: int = 7
    countdown: int | None = None
    ball: tuple | None = None        # (lateral_m, depth 0 far..1 near, height_m)
    paddle_ab: tuple | None = None   # the player's hand in reach-box coordinates
    arrival_ab: tuple | None = None  # where the incoming ball will arrive
    cpu_x_m: float = 0.0
    last_kmh: float | None = None
    last_label: str = ""
    spin_text: str = ""
    mqtt_status: str = "off"         # ok | offline | off
    hub_status: str = "ok"           # ok | stale | off
    message: str = ""
    gates: tuple = ()
    show_xray: bool = False
    flash: tuple | None = None       # (BGR colour, alpha)
    keys_hint: str = "SPACE start/swing   1-3 level   M mode   X x-ray   D motors   S sound   Q quit"
    leaderboard: tuple = field(default_factory=tuple)   # ((name, score), ...) top rows for the end screen
    swing_trace: tuple = ()          # recent signed forward gyro rate in dps (the IMU, made visible)
    swing_scale: float = 1200.0      # dps that fills the trace panel (the player's hard-swing rate)
    swing_threshold: float = 0.0     # dps below which a swing does not count (T_PK)
    player_name: str = ""            # highlights the player's own row on the leaderboard


def render(state, size=(1280, 720), background=None):
    w, h = size
    frame = canvas.fit_background(background, w, h) if background is not None else canvas.new_frame(w, h)
    canvas.draw_court(frame)
    _draw_actors(frame, state, w, h)
    _draw_top_bar(frame, state, w, h)
    _draw_phase(frame, state, w, h)
    if state.phase == "MATCH_OVER":
        _draw_leaderboard(frame, state)
    if state.show_xray and state.gates:
        _draw_xray(frame, state, w, h)
    _draw_swing(frame, state, w, h)
    _draw_footer(frame, state, w, h)
    if state.flash:
        canvas.tint(frame, state.flash[0], state.flash[1])
    return frame


def _draw_actors(frame, s, w, h):
    canvas.draw_ring(frame, s.cpu_x_m, 0.0, 0.0, 40, GREY, 4)                     # the CPU paddle
    if s.arrival_ab is not None and s.phase == "RALLY":
        canvas.draw_ring(frame, physics.x_of_a(s.arrival_ab[0]), 1.0, 0.55 * s.arrival_ab[1], 70, AMBER, 3)
    if s.ball is not None:
        canvas.draw_ball(frame, *s.ball)
    if s.paddle_ab is not None:
        canvas.draw_ring(frame, physics.x_of_a(s.paddle_ab[0]), 1.0, 0.55 * s.paddle_ab[1], 55, GREEN, 5)


def _draw_top_bar(frame, s, w, h):
    mode = "SURVIVAL" if s.mode == "survival" else "MATCH"
    canvas.draw_text(frame, f"{mode}  {s.level_name.upper()}", (24, 44), 1.0, WHITE, 2)
    canvas.draw_text(frame, f"MQTT {s.mqtt_status.upper()}   HUB {s.hub_status.upper()}", (w - 24, 44), 0.8,
                     GREEN if (s.mqtt_status, s.hub_status) == ("ok", "ok") else AMBER, 2, anchor="right")
    if s.mode == "survival":
        canvas.draw_text(frame, str(s.streak), (w // 2, 150), 4.2, WHITE, 8, anchor="center")
        canvas.draw_text(frame, f"BEST {max(s.record, s.streak)}", (w // 2, 200), 1.1, AMBER, 2, anchor="center")
    else:
        canvas.draw_text(frame, f"YOU {s.player_points}  -  {s.cpu_points} CPU", (w // 2, 120), 2.2, WHITE, 5,
                         anchor="center")
        canvas.draw_text(frame, f"first to {s.target}   streak {s.streak}   best {max(s.record, s.streak)}",
                         (w // 2, 175), 0.9, GREY, 2, anchor="center")


def _draw_phase(frame, s, w, h):
    if s.phase == "LOBBY":
        canvas.draw_text(frame, "SHOW THE START CARD", (w // 2, h // 2 - 20), 2.0, WHITE, 4, anchor="center")
        canvas.draw_text(frame, "or press SPACE", (w // 2, h // 2 + 40), 1.2, GREY, 2, anchor="center")
    elif s.phase == "COUNTDOWN" and s.countdown is not None:
        canvas.draw_text(frame, str(s.countdown), (w // 2, h // 2 + 80), 9.0, AMBER, 16, anchor="center")
    elif s.phase == "POINT_OVER":
        canvas.draw_text(frame, "NEXT BALL...", (w // 2, h // 2), 1.8, WHITE, 3, anchor="center")
    elif s.phase == "MATCH_OVER":
        title = "GAME OVER" if s.mode == "survival" else "MATCH OVER"
        canvas.draw_text(frame, title, (w // 2, h // 2 - 60), 2.6, RED, 5, anchor="center")
        if s.mode == "survival":
            canvas.draw_text(frame, f"streak {s.streak}   best {s.record}", (w // 2, h // 2), 1.5, WHITE, 3,
                             anchor="center")
        canvas.draw_text(frame, "SPACE to play again", (w // 2, h - 120), 1.2, WHITE, 2, anchor="center")


def _draw_leaderboard(frame, s):
    """A panel on the left (clear of the prompts in the middle); the player's own row is green."""
    rows = s.leaderboard[:5]
    if not rows:
        return
    x0, y0, pw, ph = 24, 100, 380, 64 + 40 * len(rows)
    canvas.dim_rect(frame, x0, y0, pw, ph)
    canvas.draw_text(frame, "BEST STREAKS" if s.mode == "survival" else "MATCH WINS", (x0 + 14, y0 + 36), 0.8, AMBER, 2)
    for i, (name, score) in enumerate(rows):
        mine = bool(s.player_name) and name.lower() == s.player_name.lower()
        color = GREEN if mine else WHITE if i == 0 else GREY
        y = y0 + 80 + 40 * i
        canvas.draw_text(frame, f"{i + 1}. {name}", (x0 + 14, y), 0.9, color, 2)
        canvas.draw_text(frame, str(score), (x0 + pw - 14, y), 0.9, color, 2, anchor="right")


def _draw_xray(frame, s, w, h):
    x0, y = w - 470, 110
    canvas.draw_text(frame, "X-RAY: why that counted", (x0, y), 0.7, AMBER, 2)
    for gate in s.gates:
        y += 34
        color = GREEN if gate.passed else RED
        canvas.draw_text(frame, f"{gate.name} {'OK' if gate.passed else 'X '}  {gate.note[:36]}", (x0, y), 0.55,
                         color, 1)


def _draw_swing(frame, s, w, h):
    if not s.swing_trace:
        return
    x0, y0, pw, ph = 24, h - 215, 360, 110
    canvas.draw_trace(frame, x0, y0, pw, ph, s.swing_trace, s.swing_scale, s.swing_threshold, GREY, GREEN)
    canvas.draw_text(frame, "IMU SWING", (x0 + 8, y0 + 22), 0.55, GREY, 1)


def _draw_footer(frame, s, w, h):
    parts = []
    if s.last_kmh is not None:
        parts.append(f"{s.last_kmh:.0f} km/h")
    for extra in (s.spin_text, s.last_label.upper()):
        if extra:
            parts.append(extra)
    if parts:
        canvas.draw_text(frame, "  ".join(parts), (w // 2, h - 70), 1.6, WHITE, 3, anchor="center")
    if s.message:
        canvas.draw_text(frame, s.message, (w // 2 - 120, 260), 1.7, AMBER, 4, anchor="center")
    canvas.draw_text(frame, s.keys_hint, (24, h - 20), 0.55, GREY, 1)
