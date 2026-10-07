"""HUD composition: one OpenCV frame from a HudState (large glyphs, readable at 1.8 m).

The court is scene.py (the table in perspective, your paddle, the computer's, the ball and its shadow); this adds the
score panels, the prompts, the x-ray panel that lists the judge gates J1-J6 with OK/X and a note (so the screen explains
every decision: "hand 0.9 SW from the ball" -- the policy made visible) and the camera as a small picture in the corner.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from pingpong import canvas, court3d, levels, scene

GREEN, RED, AMBER, WHITE, GREY = (80, 220, 80), (70, 70, 240), (40, 170, 255), (255, 255, 255), (170, 170, 170)
LOW_BATTERY = 20                 # percent: the hub's number goes amber below this
PIP_SIZE = (256, 144)            # the camera picture in the corner (px)


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
    ball: tuple | None = None        # (x, y, z) metres in physics.py's world: where the ball is now
    paddle: tuple | None = None      # (x, y, z): your paddle, the hand's place on the table plus the stroke
    rest: tuple | None = None        # (x, y, z): where the hand has the paddle (the stroke starts and ends here)
    paddle_angle: float = 0.0        # degrees, clockwise as the player sees it: how far the hub is turned in the hand
    reach_m: float = 0.30            # how far from the paddle (across the table) a ball can still be hit
    zone: tuple | None = None        # (far z, near z): the stretch of table over which the incoming ball can be hit
    cpu_x_m: float = 0.0
    cpu_swing: float | None = None   # 0..1 while the computer's paddle is hitting the ball, else None
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
    hand_img: tuple | None = None    # your hand in the camera picture: (across, down) as fractions, as the mirror shows it


@lru_cache(maxsize=4)
def _camera(size):
    return court3d.Camera.for_frame(*size)


def render(state, size=(1280, 720), background=None):
    w, h = size
    frame = np.empty((h, w, 3), dtype=np.uint8)
    scene.draw_scene(frame, _camera((w, h)), state)
    _draw_score(frame, state, w, h)
    _draw_top_bar(frame, state, w, h)
    _draw_phase(frame, state, w, h)
    if state.phase == "MATCH_OVER":
        _draw_leaderboard(frame, state)
    if state.show_xray and state.gates:
        _draw_xray(frame, state, w, h)
    _draw_swing(frame, state, w, h)
    if background is not None:
        _draw_pip(frame, background, state, w, h)
    _draw_footer(frame, state, w, h)
    if state.flash:
        canvas.tint(frame, state.flash[0], state.flash[1])
    return frame


def _draw_score(frame, s, w, h):
    """Two panels on the sides of the table, like the arcade original's score boxes."""
    if s.mode == "survival":
        left, right = ("STREAK", s.streak, WHITE), ("BEST", max(s.record, s.streak), AMBER)
    else:
        left, right = ("YOU", s.player_points, WHITE), ("CPU", s.cpu_points, WHITE)
    if s.phase == "MATCH_OVER" and s.leaderboard:
        left = None                                          # the end screen's board takes the left side
    for panel, x in ((left, 16), (right, w - 246)):
        if panel is None:
            continue
        label, value, color = panel
        canvas.panel(frame, x, 110, 230, 128)
        canvas.draw_text(frame, label, (x + 115, 146), 0.8, GREY, 2, anchor="center")
        canvas.draw_text(frame, str(value), (x + 115, 222), 3.0, color, 6, anchor="center")
    if s.mode != "survival":
        canvas.draw_text(frame, f"first to {s.target}   streak {s.streak}   best {max(s.record, s.streak)}",
                         (w // 2, 100), 0.7, GREY, 2, anchor="center")


def _draw_top_bar(frame, s, w, h):
    mode = levels.MODE_NAMES.get(s.mode, s.mode.upper())
    canvas.draw_text(frame, f"{mode}  {s.level_name.upper()}", (24, 44), 1.0, WHITE, 2)
    battery = "" if s.hub_battery is None else f" {s.hub_battery}%"
    healthy = (s.mqtt_status, s.hub_status) == ("ok", "ok") and (s.hub_battery is None or s.hub_battery >= LOW_BATTERY)
    canvas.draw_text(frame, f"MQTT {s.mqtt_status.upper()}   HUB {s.hub_status.upper()}{battery}", (w - 24, 44), 0.8,
                     GREEN if healthy else AMBER, 2, anchor="right")


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
    x0, y0 = w - pw - 24, 270
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
    x0, y0, pw, ph = 16, h - 190, 290, 96
    canvas.draw_trace(frame, x0, y0, pw, ph, s.swing_trace, s.swing_scale, s.swing_threshold, GREY, GREEN)
    canvas.draw_text(frame, s.swing_label, (x0 + 8, y0 + 20), 0.5, GREY, 1)


def _draw_pip(frame, camera, s, w, h):
    """The camera, mirrored, in the bottom corner, with a ring on your hand: the proof you are being seen."""
    pw, ph = PIP_SIZE
    x0, y0 = w - pw - 24, h - ph - 60
    frame[y0:y0 + ph, x0:x0 + pw] = cv2.resize(camera, (pw, ph), interpolation=cv2.INTER_AREA)
    if s.hand_img is not None:
        cv2.circle(frame, (round(x0 + s.hand_img[0] * pw), round(y0 + s.hand_img[1] * ph)), 9, AMBER, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), GREY, 1)


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
