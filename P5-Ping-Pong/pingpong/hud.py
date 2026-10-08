"""HUD composition: one OpenCV frame from a HudState, in the bright rounded style of a console sports game (large glyphs,
readable at 1.8 m).

The court is scene.py (the table on its terrace, your paddle, the opponent's, the ball and its shadow); this puts a card for each
side at the top (a portrait, a name, the score), the game and level in the middle, pops for the speed of a hit and the countdown,
banners for a point or a record, the x-ray panel that lists the judge gates J1-J6 with a tick or a cross and a note (so the
screen explains every decision: "hand 0.9 SW from the ball"), and the camera in a rounded window in the corner.  The screens
around the game (the intro, the title, the menus, the face-off, the results) are screens.py and intro.py.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from pingpong import anim, cast, characters, court3d, fonts, holdstart, levels, scene, screens, ui
from pingpong.cast import rgb
from pingpong.uistate import Results, UiState

GREEN, RED, AMBER, WHITE, GREY = (80, 220, 80), (70, 70, 240), (40, 170, 255), (255, 255, 255), (170, 170, 170)
NAVY, SKY, CORAL, ORANGE, GOLD = rgb(24, 48, 96), rgb(54, 160, 255), rgb(255, 104, 104), rgb(255, 146, 30), rgb(255, 196, 40)
MINT, PURPLE, SLATE = rgb(54, 190, 112), rgb(150, 100, 240), rgb(112, 126, 150)
LOW_BATTERY = 20                 # percent: the hub's number goes amber below this
PING_OK_MS = 250                 # a line slower than this (there and back) is shown as a worry in an online game
PIP_SIZE = (256, 144)            # the camera picture in the corner (px)
CARD_W, CARD_H = 318, 100
LABEL_COLOR = {"perfect": GOLD, "good": MINT, "early": ORANGE, "late": ORANGE}


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
    player_name: str = ""            # whose portrait is on the left card, and whose row is highlighted on the leaderboard
    hand_img: tuple | None = None    # your hand in the camera picture: (across, down) as fractions, as the mirror shows it
    start_button: tuple | None = None   # (how far the hold has come 0..1, hand on it) while the START button is shown, else None
    cursor: tuple | None = None      # (a, b): where the hand points over the screen (across, up; 0..1) while the button is shown
    cpu_mood: str = "happy"          # the opponent's face: happy | grin | cheer | sad | surprised | smug | neutral
    anim_t: float = 0.0              # seconds on the game's clock: the blinking and bobbing of the people
    countdown_t: float = 0.0         # seconds since the countdown's digit changed
    point_for: str = ""              # who won the point just played: "player" | "cpu" (while it is shown)
    screen: str = "GAME"             # GAME, or one of the screens round it: INTRO | TITLE | MODE | OPPONENT | VS | RESULTS
    ui: UiState | None = None        # what that screen is told (flow.py)
    results: Results | None = None   # the results screen's words and numbers
    opponent_name: str = ""          # a friend being played online (their avatar stands behind the table); "" against the computer
    ping_ms: float | None = None     # how long a message takes there and back (online play), None until the first answer


@lru_cache(maxsize=4)
def _camera(size):
    return court3d.Camera.for_frame(*size)


def render(state, size=(1280, 720), background=None):
    w, h = size
    frame = np.empty((h, w, 3), dtype=np.uint8)
    if state.screen != "GAME":
        screens.render(frame, state, background)
        return frame
    scene.draw_scene(frame, _camera((w, h)), state)
    _score_cards(frame, state, w)
    _chips(frame, state, w)
    _status(frame, state, h)
    _phase(frame, state, w, h)
    _hit_pop(frame, state, w, h)
    _message(frame, state, w, h)
    if state.show_xray and state.gates:
        _xray(frame, state, w)
    _swing(frame, state, h)
    if background is not None:
        _pip(frame, background, state, w, h)
    if state.phase in ("LOBBY", "MATCH_OVER"):
        fonts.draw(frame, state.keys_hint, w // 2, h - 24, 20, WHITE, outline=NAVY, outline_px=3, opacity=0.85)
    if state.flash:
        _tint(frame, state.flash[0], state.flash[1])
    return frame


def _tint(frame, color, alpha):
    cv2.addWeighted(np.full_like(frame, color), alpha, frame, 1.0 - alpha, 0, frame)


# --- the cards, the chips -------------------------------------------------------------------------------------------------------------
def _card(frame, x, y, side, look, name, label, value, color, mood, t):
    ui.panel(frame, x, y, CARD_W, CARD_H, radius=32, fill=WHITE, opacity=1.0, border=color, border_px=4)
    left = side == "left"
    cx = x + 58 if left else x + CARD_W - 58
    if look is None:
        ui.panel(frame, cx - 36, y + 14, 72, 72, radius=36, fill=(rgb(255, 244, 205), rgb(255, 226, 150)), opacity=1.0, shadow=False)
        ui.trophy(frame, cx, y + 50, 26, GOLD)
    else:
        characters.portrait(frame, cx, y + CARD_H / 2, 76, look, mood=mood, t=t)
    tx = x + 112 if left else x + CARD_W - 112
    anchor = "lm" if left else "rm"
    fonts.draw(frame, name, tx, y + 27, 22, SLATE, anchor=anchor)
    fonts.draw(frame, str(value), tx, y + 66, 52, NAVY if color == SKY else color, anchor=anchor)
    vw = fonts.measure(str(value), 52)[0]
    fonts.draw(frame, label, tx + (vw + 10 if left else -(vw + 10)), y + 80, 17, SLATE, anchor=anchor)


def _score_cards(frame, s, w):
    me = cast.player_look(s.player_name or "player")
    opp = cast.opponent_look(s.level_name, s.opponent_name)
    if s.mode == "survival":
        _card(frame, 20, 16, "left", me, me.name, "STREAK", s.streak, SKY, "happy", s.anim_t)
        _card(frame, w - 20 - CARD_W, 16, "right", None, "BEST", "STREAK", max(s.record, s.streak), ORANGE, "happy", s.anim_t)
    else:
        _card(frame, 20, 16, "left", me, me.name, "POINTS", s.player_points, SKY, "happy", s.anim_t)
        _card(frame, w - 20 - CARD_W, 16, "right", opp, opp.name, "POINTS", s.cpu_points, CORAL, s.cpu_mood, s.anim_t)


def _pill_text(frame, cx, y, text, size, color=NAVY, fill=WHITE, opacity=0.92, pad=26):
    tw, _ = fonts.measure(text, size)
    ui.panel(frame, cx - tw / 2 - pad, y, tw + 2 * pad, size + 22, radius=(size + 22) // 2, fill=fill, opacity=opacity)
    fonts.draw(frame, text, cx, y + (size + 22) / 2 + 1, size, color, anchor="mm")


def _chips(frame, s, w):
    """Which game, against which level, in a tag under your card."""
    name = "ONLINE" if s.opponent_name else levels.MODE_NAMES.get(s.mode, s.mode.upper())
    middle = f"FIRST TO {s.target}   \u2022   " if s.mode == "match" else ""
    text = f"{name}   \u2022   {middle}{s.level_name.upper()}"
    tw, _ = fonts.measure(text, 21)
    ui.panel(frame, 24, 124, tw + 44, 38, radius=19, fill=(rgb(255, 255, 255), rgb(232, 242, 255)), opacity=0.95, shadow=False)
    fonts.draw(frame, text, 46, 143, 21, NAVY, anchor="lm")


def _status(frame, s, h):
    battery = "" if s.hub_battery is None else f" {s.hub_battery}%"
    hub_ok = s.hub_status == "ok" and (s.hub_battery is None or s.hub_battery >= LOW_BATTERY)
    if s.opponent_name:                                              # a friend: how good the line is matters, the score topic does not
        ping = "..." if s.ping_ms is None else f"{s.ping_ms:.0f} ms"
        healthy = hub_ok and s.ping_ms is not None and s.ping_ms < PING_OK_MS
        text = f"ONLINE   PING {ping}    HUB {s.hub_status.upper()}{battery}"
    else:
        healthy = (s.mqtt_status, s.hub_status) == ("ok", "ok") and hub_ok
        text = f"MQTT {s.mqtt_status.upper()}    HUB {s.hub_status.upper()}{battery}"
    tw, _ = fonts.measure(text, 18)
    ui.panel(frame, 20, h - 52, tw + 52, 34, radius=17, fill=WHITE, opacity=0.86, shadow=False)
    cv2.circle(frame, (40, h - 35), 7, MINT if healthy else rgb(255, 170, 30), -1, cv2.LINE_AA)
    fonts.draw(frame, text, 56, h - 34, 18, NAVY if healthy else rgb(190, 110, 0), anchor="lm")


# --- what is happening ----------------------------------------------------------------------------------------------------------------
def _banner(frame, w, cy, text, color, size=64):
    size = fonts.fit_size(text, w - 220, max_size=size, min_size=26)
    tw, _ = fonts.measure(text, size)
    ui.panel(frame, w / 2 - tw / 2 - 48, cy - size * 0.75, tw + 96, size * 1.5, radius=int(size * 0.75),
             fill=(color, tuple(int(c * 0.78) for c in color)), opacity=1.0, border=WHITE, border_px=5)
    fonts.draw(frame, text, w / 2, cy + 2, size, WHITE, anchor="mm", outline=tuple(int(c * 0.5) for c in color), outline_px=5)


def _phase(frame, s, w, h):
    if s.phase == "LOBBY":
        _lobby(frame, s, w, h)
    elif s.phase == "COUNTDOWN" and s.countdown is not None:
        t = anim.pop(min(1.0, s.countdown_t / 0.4))
        color = {3: CORAL, 2: ORANGE}.get(s.countdown, MINT)
        r = round(104 * min(1.2, max(0.2, t)))
        cx, cy = w // 2, h // 2 + 30
        ui.glow(frame, cx, cy, r * 1.9, color, 0.45)
        cv2.circle(frame, (cx, cy), r, WHITE, -1, cv2.LINE_AA)
        ui.ring(frame, cx, cy, r - 7, 1.0 - min(1.0, s.countdown_t), color, 14, track=tuple(int(c * 0.25 + 190) for c in color))
        fonts.draw(frame, str(s.countdown), cx, cy + 4, round(150 * min(1.2, max(0.3, t))), NAVY, anchor="mm")
    elif s.phase == "POINT_OVER":
        mine = s.point_for != "cpu"
        _banner(frame, w, h // 2 - 10, "NICE POINT!" if mine else "POINT TO THE CPU", MINT if mine else CORAL)
    elif s.phase == "MATCH_OVER":
        _over(frame, s, w, h)


def _lobby(frame, s, w, h):
    _start_button(frame, s, w, h)
    ui.panel(frame, w / 2 - 330, h / 2 - 120, 660, 190, radius=40, fill=WHITE, opacity=0.93)
    if s.start_button is not None:
        fonts.draw(frame, "HOLD THE HUB ON START", w / 2, h / 2 - 52, 52, NAVY)
        fonts.draw(frame, "top right, 1.5 s   (or show the START card, or press SPACE)", w / 2, h / 2 + 20, 24, SLATE)
    else:
        fonts.draw(frame, "SHOW THE START CARD", w / 2, h / 2 - 52, 56, NAVY)
        fonts.draw(frame, "or press SPACE", w / 2, h / 2 + 20, 34, SLATE)


def _start_button(frame, s, w, h):
    """The START button in the top right, and a pointer where the hand points over the screen: hold the hub on the button to start."""
    if s.start_button is None:
        return
    progress, on = s.start_button
    x, y, bw, bh = (f * v for f, v in zip(holdstart.BUTTON, (w, h, w, h)))
    ui.pill(frame, x + bw / 2, y + bh / 2, bw, bh * 0.62, "START", fill=(rgb(90, 214, 130), rgb(40, 168, 92)), fill_progress=rgb(255, 214, 70),
            progress=progress, outline=WHITE, scale=1.08 if on else 1.0)
    fonts.draw(frame, "hold the hub here", x + bw / 2, y + bh * 0.9, 18, WHITE, outline=NAVY, outline_px=3)
    if s.cursor is not None:
        ui.cursor(frame, min(1.0, max(0.0, s.cursor[0])) * (w - 1), (1.0 - min(1.0, max(0.0, s.cursor[1]))) * (h - 1), progress=progress)


def _over(frame, s, w, h):
    title = "GAME OVER" if s.mode == "survival" else "MATCH OVER"
    _start_button(frame, s, w, h)
    ui.panel(frame, w / 2 - 330, h / 2 - 150, 660, 250, radius=40, fill=WHITE, opacity=0.94)
    fonts.draw(frame, title, w / 2, h / 2 - 78, 70, CORAL, outline=WHITE, outline_px=4)
    if s.mode == "survival":
        fonts.draw(frame, f"streak {s.streak}    best {s.record}", w / 2, h / 2 + 4, 44, NAVY)
    else:
        fonts.draw(frame, f"{s.player_points}  -  {s.cpu_points}", w / 2, h / 2 + 4, 54, NAVY)
    fonts.draw(frame, "hold the hub on START to play again" if s.start_button is not None else "SPACE to play again", w / 2, h / 2 + 70, 26,
               SLATE)
    _leaderboard(frame, s)


def _leaderboard(frame, s):
    """A panel on the left (clear of the prompts in the middle); the player's own row is green."""
    rows = s.leaderboard[:5]
    if not rows:
        return
    x0, y0, pw, ph = 24, 150, 340, 64 + 40 * len(rows)
    ui.panel(frame, x0, y0, pw, ph, radius=26, fill=WHITE, opacity=0.94)
    fonts.draw(frame, "BEST STREAKS" if s.mode == "survival" else "MATCH WINS", x0 + 20, y0 + 32, 24, ORANGE, anchor="lm")
    for i, (name, score) in enumerate(rows):
        mine = bool(s.player_name) and name.lower() == s.player_name.lower()
        color = MINT if mine else NAVY if i == 0 else SLATE
        y = y0 + 76 + 40 * i
        fonts.draw(frame, f"{i + 1}. {name}", x0 + 20, y, 26, color, anchor="lm")
        fonts.draw(frame, str(score), x0 + pw - 20, y, 26, color, anchor="rm")


def _hit_pop(frame, s, w, h):
    """The speed of the last hit and how good it was, in a bar at the bottom: the speed on the left, a coloured tag on the right."""
    if s.last_kmh is None and not s.last_label:
        return
    label = s.last_label.split()[0] if s.last_label else ""
    color = LABEL_COLOR.get(label, CORAL)
    cy = h - 70
    ui.panel(frame, w / 2 - 250, cy - 34, 500, 68, radius=34, fill=WHITE, opacity=0.95)
    if s.last_kmh is not None:
        fonts.draw(frame, f"{s.last_kmh:.0f} km/h", w / 2 - 224, cy + 1, 38, NAVY, anchor="lm")
    if s.last_label:
        ui.pill(frame, w / 2 + 128, cy, 200, 50, s.last_label.upper(), fill=(color, tuple(int(c * 0.8) for c in color)), size=26,
                outline=WHITE, gloss=False, shadow=False)
        if label == "perfect":
            for dx in (-1, 1):
                ui.star(frame, w / 2 + 128 + dx * 118, cy - 24, 17, GOLD, outline=rgb(230, 140, 20), outline_px=2, angle_deg=dx * 12)
    if s.spin_text:
        _pill_text(frame, w / 2, cy - 92, s.spin_text, 22, color=PURPLE)


def _message(frame, s, w, h):
    if not s.message:
        return
    if s.message == "GO!":                                                # the first ball: a big word in the middle
        ui.glow(frame, w / 2, h / 2 + 30, 230, MINT, 0.5)
        fonts.draw(frame, "GO!", w / 2, h / 2 + 30, 170, WHITE, outline=NAVY, outline_px=12, shadow=(0, 9, NAVY, 0.3))
        return
    color = PURPLE if "RECORD" in s.message else rgb(255, 160, 20)
    if s.phase == "LOBBY":                                                # a standing notice: below the start prompt
        lines = fonts.wrap_lines(s.message, w - 160, 30)
        ui.panel(frame, 60, 470, w - 120, 22 + 40 * len(lines), radius=30, fill=WHITE, opacity=0.93, border=color, border_px=4)
        for i, line in enumerate(lines):
            fonts.draw(frame, line, w / 2, 490 + 40 * i, 30, rgb(150, 90, 0), anchor="mm")
    else:
        _banner(frame, w, 150, s.message, color, size=54)


def _xray(frame, s, w):
    pw = 540
    rows = []
    for gate in s.gates:
        text = f"{'OK' if gate.passed else 'NO'}  {gate.name}  {gate.note}".rstrip()
        rows.append((MINT if gate.passed else CORAL, fonts.wrap_lines(text, pw - 40, 20, 2)))
    x0, y0 = w - pw - 24, 134
    ph = 60 + 28 * sum(len(lines) for _, lines in rows)
    ui.panel(frame, x0, y0, pw, ph, radius=26, fill=WHITE, opacity=0.95)
    fonts.draw(frame, "X-RAY: why that counted", x0 + 20, y0 + 30, 24, ORANGE, anchor="lm")
    y = y0 + 38
    for color, lines in rows:
        for line in lines:
            y += 28
            fonts.draw(frame, line, x0 + 20, y, 20, color if color == CORAL else NAVY, anchor="lm")


def _swing(frame, s, h):
    if not s.swing_trace:
        return
    x0, y0, pw, ph = 20, h - 196, 300, 120
    ui.panel(frame, x0, y0, pw, ph, radius=22, fill=WHITE, opacity=0.9)
    fonts.draw(frame, s.swing_label, x0 + 16, y0 + 20, 17, SLATE, anchor="lm")
    top, bottom = y0 + 34, y0 + ph - 12
    mid = (top + bottom) // 2
    scale = max(1.0, s.swing_scale)
    cv2.line(frame, (x0 + 12, mid), (x0 + pw - 12, mid), rgb(200, 210, 225), 1, cv2.LINE_AA)
    if s.swing_threshold > 0:
        ty = round(mid - min(1.0, s.swing_threshold / scale) * (mid - top))
        cv2.line(frame, (x0 + 12, ty), (x0 + pw - 12, ty), rgb(255, 150, 30), 1, cv2.LINE_AA)
    n = len(s.swing_trace)
    pts = [(x0 + 12 + (i * (pw - 25)) // max(1, n - 1), round(mid - max(-1.0, min(1.0, v / scale)) * (mid - top)))
           for i, v in enumerate(s.swing_trace)]
    if n == 1:
        cv2.circle(frame, pts[0], 3, NAVY, -1, cv2.LINE_AA)
    for (a, va), (b, vb) in zip(zip(pts, s.swing_trace), zip(pts[1:], s.swing_trace[1:])):
        cv2.line(frame, a, b, MINT if max(va, vb) >= s.swing_threshold > 0 else NAVY, 2, cv2.LINE_AA)


def _pip(frame, camera, s, w, h):
    """The camera, mirrored, in a rounded window in the bottom corner, with a ring on your hand: the proof you are being seen."""
    pw, ph = PIP_SIZE
    x0, y0 = w - pw - 24, h - ph - 60
    picture = cv2.resize(camera, (pw, ph), interpolation=cv2.INTER_AREA)
    if s.hand_img is not None:
        cv2.circle(picture, (round(s.hand_img[0] * pw), round(s.hand_img[1] * ph)), 9, AMBER, 2, cv2.LINE_AA)
    ui.panel(frame, x0 - 5, y0 - 5, pw + 10, ph + 10, radius=24, fill=WHITE, opacity=1.0)
    fonts.blit(frame, picture, ui.rounded_mask(pw, ph, 20), x0, y0)
