"""HUD composition: one OpenCV frame from a HudState (large glyphs, readable at 1.8 m).

The x-ray panel lists the judge gates J1-J6 with OK/X and a note, so the screen
explains every decision ("hand 0.9 SW from the ball") -- the policy made visible.
"""

from dataclasses import dataclass, field

import cv2

import math

from pingpong import canvas, levels

GREEN, RED, AMBER, WHITE, GREY = (80, 220, 80), (70, 70, 240), (40, 170, 255), (255, 255, 255), (170, 170, 170)
LOW_BATTERY = 20                 # percent: the hub's number goes amber below this
PLANE_CENTRE_Y = 0.60            # the hand plane's middle, as a fraction of the frame height
PLANE_SW_PX = 0.17               # pixels per shoulder width on the hand plane, as a fraction of the frame height
TARGET_R = 16                    # px: the ring where the ball will arrive; it must end up inside your paddle's face
CPU_PADDLE_R = 80                # px at full size: the computer's paddle (shrunk by the court's perspective)
CPU_PADDLE_LIFT_M = 0.04         # its face is centred this far above the far edge of the table


def plane_xy(ab, box_sw, w, h):
    """Reach-box coordinates (a, b) -> pixels on the hand plane, where the paddle dot and the target ring live.

    The same pixels per shoulder width across and up: the judge measures the hand in shoulder widths, so a ring of
    radius R shoulder widths is a circle and 'the dot is inside the ring' is exactly 'the judge will say hit'.  (The
    court's own mapping squashed height 2.5x against width, and rings that looked like they touched were missed.)"""
    s = PLANE_SW_PX * h
    return round(w / 2 + (ab[0] - 0.5) * box_sw[0] * s), round(PLANE_CENTRE_Y * h - (ab[1] - 0.5) * box_sw[1] * s)


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
    hub_battery: int | None = None   # percent, None until the hub has reported
    message: str = ""
    gates: tuple = ()
    show_xray: bool = False
    flash: tuple | None = None       # (BGR colour, alpha)
    keys_hint: str = "SPACE start/swing   1-3 level   M mode   X x-ray   D motors   S sound   Q quit"
    leaderboard: tuple = field(default_factory=tuple)   # ((name, score), ...) top rows for the end screen
    swing_trace: tuple = ()          # recent signed forward gyro rate in dps (the IMU, made visible)
    swing_label: str = "IMU SWING"   # "CAMERA SWING" when the camera's hand speed is the swing sensor
    swing_scale: float = 1200.0      # dps that fills the trace panel (the player's hard-swing rate)
    swing_threshold: float = 0.0     # dps below which a swing does not count (T_PK)
    player_name: str = ""            # highlights the player's own row on the leaderboard
    box_sw: tuple = (2.0, 1.4)       # the reach box's width and height in shoulder widths (sets the hand plane's shape)
    radius_sw: float = 0.55          # the level's hit radius: the target ring is exactly this big
    reach: float = 1.0               # the share of the reach box the balls arrive in
    cpu_swing: float | None = None   # 0..1 while the computer's paddle is hitting the ball, else None
    paddle_angle: float = 0.0        # degrees, clockwise as the player sees it: how far the hub is turned in the hand


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
    _draw_cpu_paddle(frame, s, w, h)
    if s.phase in ("COUNTDOWN", "RALLY"):
        _draw_arrival_window(frame, s, w, h)
    if s.paddle_ab is not None:
        # your paddle: its face is the judge's hit zone (a circle of the level's radius), centred on your hand
        px, py = plane_xy(s.paddle_ab, s.box_sw, w, h)
        canvas.draw_paddle(frame, px, py, round(s.radius_sw * PLANE_SW_PX * h), angle_deg=s.paddle_angle)
        cv2.circle(frame, (px, py), 4, WHITE, -1, cv2.LINE_AA)
    target = None
    if s.arrival_ab is not None and s.phase == "RALLY":
        target = plane_xy(s.arrival_ab, s.box_sw, w, h)             # where the ball will arrive: get the paddle over it
        cv2.circle(frame, target, TARGET_R, AMBER, 3, cv2.LINE_AA)
        cv2.circle(frame, target, 3, AMBER, -1, cv2.LINE_AA)
    if s.ball is not None:
        canvas.draw_ball(frame, *s.ball, toward=target)


def _draw_cpu_paddle(frame, s, w, h):
    """The computer's paddle at the far end: it waits where it hit, chases your shot, and flicks when it hits."""
    px, py, scale = canvas.project(s.cpu_x_m, 0.0, CPU_PADDLE_LIFT_M, w, h)
    angle = 0.0 if s.cpu_swing is None else 55.0 * math.sin(math.pi * s.cpu_swing)
    canvas.draw_paddle(frame, px, py, round(CPU_PADDLE_R * scale), rubber=canvas.RUBBER_BLUE, angle_deg=angle)


def _draw_arrival_window(frame, s, w, h):
    """Where balls can arrive: the level's share of the reach box, so the player sees what they have to reach."""
    half_w = 0.5 * s.reach * s.box_sw[0] * PLANE_SW_PX * h
    half_h = 0.5 * s.reach * s.box_sw[1] * PLANE_SW_PX * h
    cx, cy = plane_xy((0.5, 0.5), s.box_sw, w, h)
    cv2.rectangle(frame, (round(cx - half_w), round(cy - half_h)), (round(cx + half_w), round(cy + half_h)), GREY, 1)


def _draw_top_bar(frame, s, w, h):
    mode = levels.MODE_NAMES.get(s.mode, s.mode.upper())
    canvas.draw_text(frame, f"{mode}  {s.level_name.upper()}", (24, 44), 1.0, WHITE, 2)
    battery = "" if s.hub_battery is None else f" {s.hub_battery}%"
    healthy = (s.mqtt_status, s.hub_status) == ("ok", "ok") and (s.hub_battery is None or s.hub_battery >= LOW_BATTERY)
    canvas.draw_text(frame, f"MQTT {s.mqtt_status.upper()}   HUB {s.hub_status.upper()}{battery}", (w - 24, 44), 0.8,
                     GREEN if healthy else AMBER, 2, anchor="right")
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
    x0, y0, pw, ph = 24, 150, 340, 64 + 40 * len(rows)
    canvas.dim_rect(frame, x0, y0, pw, ph)
    canvas.draw_text(frame, "BEST STREAKS" if s.mode == "survival" else "MATCH WINS", (x0 + 14, y0 + 36), 0.8, AMBER, 2)
    for i, (name, score) in enumerate(rows):
        mine = bool(s.player_name) and name.lower() == s.player_name.lower()
        color = GREEN if mine else WHITE if i == 0 else GREY
        y = y0 + 80 + 40 * i
        canvas.draw_text(frame, f"{i + 1}. {name}", (x0 + 14, y), 0.9, color, 2)
        canvas.draw_text(frame, str(score), (x0 + pw - 14, y), 0.9, color, 2, anchor="right")


def _draw_xray(frame, s, w, h):
    pw = 520
    x0, y0 = w - pw - 24, 124
    rows = []
    for gate in s.gates:
        text = f"{gate.name} {'OK' if gate.passed else 'X '}  {gate.note}".rstrip()
        lines, scale = canvas.fit_text(text, pw, max_scale=0.65, min_scale=0.65, thickness=1, max_lines=2)
        rows.append((GREEN if gate.passed else RED, lines, scale))
    canvas.dim_rect(frame, x0 - 14, y0 - 40, pw + 28, 60 + 28 * sum(len(lines) for _, lines, _ in rows))
    canvas.draw_text(frame, "X-RAY: why that counted", (x0, y0), 0.8, AMBER, 2)
    y = y0 + 6
    for color, lines, scale in rows:
        for line in lines:
            y += 28
            canvas.draw_text(frame, line, (x0, y), scale, color, 1)


def _draw_swing(frame, s, w, h):
    if not s.swing_trace:
        return
    x0, y0, pw, ph = 24, h - 215, 360, 110
    canvas.draw_trace(frame, x0, y0, pw, ph, s.swing_trace, s.swing_scale, s.swing_threshold, GREY, GREEN)
    canvas.draw_text(frame, s.swing_label, (x0 + 8, y0 + 22), 0.55, GREY, 1)


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
        # in the lobby the message is a standing notice (UNCALIBRATED, what the broker holds): below the start prompt
        y = 470 if s.phase == "LOBBY" else 260
        canvas.draw_fitted(frame, s.message, w // 2, y, w - 120, max_scale=1.7, min_scale=0.75, color=AMBER)
    canvas.draw_text(frame, s.keys_hint, (24, h - 20), 0.55, GREY, 1)
