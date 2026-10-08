"""The scene: the table in perspective on a seaside terrace, with shadows, the net, two paddles, the opponent and the ball.

Everything is placed in the world of physics.py (metres) and drawn through a court3d.Camera, so depth reads the way it
does in the arcade original: things shrink and climb the picture towards the far end, the ball's shadow on the table
shows how high it is, and your paddle is an object standing in the scene (it gets smaller and higher as it reaches
forward) rather than a sticker on the screen.  Flat bright colours, no textures.  The terrace around the table, the sea
and the crowd are resort.py; the opponent is characters.py.
"""

import math
from functools import lru_cache

import cv2
import numpy as np

from pingpong import canvas, cast, characters, physics, resort
from pingpong.cast import rgb

# BGR palette
TABLE, TABLE_FRONT, LINE = rgb(34, 100, 190), rgb(20, 62, 132), (248, 248, 248)
LEG, SHADOW = rgb(120, 138, 160), (0, 0, 0)
NET, TAPE = rgb(34, 44, 74), (250, 250, 250)
BALL, BALL_RIM = rgb(255, 176, 32), rgb(176, 84, 8)
REACH = (40, 170, 255)                  # the ring on the table: how far from your paddle a ball can still be hit

PADDLE_FACE_M = 0.12                    # the face of your paddle (the ring around it is the real reach)
CPU_FACE_M = 0.11
BALL_R_M = 0.05
CPU_Y_M = physics.STRIKE_Y_M
CPU_Z_M = physics.CPU_Z_M + 0.05
OPPONENT_Z_M = 3.7                      # the opponent stands behind its end of the table, clear of its own paddle
LEG_H_M = -resort.DECK_Y                # the table stands on the deck
NET_OVERHANG_M = 0.10
HW, L = physics.HALF_WIDTH_M, physics.TABLE_LEN_M


def _pts(cam, *world):
    return np.array([cam.project(*p)[:2] for p in world], dtype=np.int32)


def _blend_poly(frame, pts, color, alpha):
    """A translucent filled polygon (only the part of the frame it covers is touched)."""
    h, w = frame.shape[:2]
    pts = np.asarray(pts, dtype=np.int32)
    x0, y0 = max(0, int(pts[:, 0].min()) - 2), max(0, int(pts[:, 1].min()) - 2)
    x1, y1 = min(w, int(pts[:, 0].max()) + 3), min(h, int(pts[:, 1].max()) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.fillPoly(layer, [pts - np.array([x0, y0], dtype=np.int32)], color, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)


@lru_cache(maxsize=4)
def _still_life(cam, w, h):
    """The terrace and the table: they never change, so they are drawn once and copied into every frame."""
    layer = np.empty((h, w, 3), dtype=np.uint8)
    resort.draw_backdrop(layer, cam, 0.0)
    draw_table(layer, cam)
    return layer


def draw_scene(frame, cam, s):
    h, w = frame.shape[:2]
    frame[:] = _still_life(cam, w, h)
    if s.paddle is not None:
        _zone(frame, cam, s)
    if s.ball is not None:
        _ball_shadow(frame, cam, s.ball)
    items = [(CPU_Z_M, "cpu"), (physics.NET_Z_M, "net")]
    if s.ball is not None:
        items.append((s.ball[2] - 0.15, "ball"))
    if s.paddle is not None:
        items.append((s.paddle[2], "paddle"))
    for _, kind in sorted(items, key=lambda item: -item[0]):
        if kind == "cpu":
            _cpu_paddle(frame, cam, s)
        elif kind == "net":
            draw_net(frame, cam)
        elif kind == "ball":
            _ball(frame, cam, s.ball)
        else:
            _player_paddle(frame, cam, s)


# --- the table ----------------------------------------------------------------------------------------------------------------
def draw_table(frame, cam):
    for x in (-HW + 0.12, HW - 0.12):
        for z in (0.14, L - 0.14):
            (x0, y0, sc), (x1, y1, _) = cam.project(x, -0.06, z), cam.project(x, -LEG_H_M, z)
            cv2.line(frame, (round(x0), round(y0)), (round(x1), round(y1)), LEG, max(3, round(0.05 * sc)), cv2.LINE_AA)
    cv2.fillPoly(frame, [_pts(cam, (-HW, 0, 0), (HW, 0, 0), (HW, 0, L), (-HW, 0, L))], TABLE, cv2.LINE_AA)
    cv2.fillPoly(frame, [_pts(cam, (-HW, 0, 0), (HW, 0, 0), (HW, -0.06, 0), (-HW, -0.06, 0))], TABLE_FRONT, cv2.LINE_AA)
    corners = _pts(cam, (-HW, 0, 0), (HW, 0, 0), (HW, 0, L), (-HW, 0, L))
    cv2.polylines(frame, [corners], True, LINE, 3, cv2.LINE_AA)
    mid = _pts(cam, (0, 0, 0), (0, 0, L))
    cv2.line(frame, tuple(int(v) for v in mid[0]), tuple(int(v) for v in mid[1]), LINE, 2, cv2.LINE_AA)


def draw_net(frame, cam):
    w = HW + NET_OVERHANG_M
    quad = _pts(cam, (-w, 0, physics.NET_Z_M), (w, 0, physics.NET_Z_M), (w, physics.NET_H_M, physics.NET_Z_M),
                (-w, physics.NET_H_M, physics.NET_Z_M))
    _blend_poly(frame, quad, NET, 0.55)
    for k in range(1, 28):                                     # the mesh
        x = -w + 2 * w * k / 28
        a, b = cam.project(x, 0, physics.NET_Z_M)[:2], cam.project(x, physics.NET_H_M, physics.NET_Z_M)[:2]
        cv2.line(frame, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), NET, 1, cv2.LINE_AA)
    top = _pts(cam, (-w, physics.NET_H_M, physics.NET_Z_M), (w, physics.NET_H_M, physics.NET_Z_M))
    cv2.line(frame, tuple(int(v) for v in top[0]), tuple(int(v) for v in top[1]), TAPE, 3, cv2.LINE_AA)
    for x in (-w, w):
        a, b = cam.project(x, 0, physics.NET_Z_M), cam.project(x, physics.NET_H_M + 0.03, physics.NET_Z_M)
        cv2.line(frame, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), (30, 30, 30), 3, cv2.LINE_AA)


# --- shadows and rings on the table -------------------------------------------------------------------------------------
def _on_table(x, z):
    return abs(x) <= HW + 0.02 and -0.02 <= z <= L + 0.02


def _ball_shadow(frame, cam, ball):
    x, _, z = ball
    if _on_table(x, z):
        _blend_poly(frame, cam.ground_circle(x, z, BALL_R_M * 1.15), SHADOW, 0.75)


def _zone(frame, cam, s):
    """Your paddle's reach on the table: an oval round where your hand has it, as long as the ball can be hit over (a
    ball whose shadow is inside it can be hit; over the paddle's own line it is exactly on time), and the paddle's shadow."""
    rest = s.rest or s.paddle
    x, _, z = rest
    z_far, z_near = s.zone if s.zone is not None else (z + 0.45, z - 0.15)
    z_far, z_near = min(z_far, physics.NET_Z_M), max(z_near, -0.45)         # (a ball is hit on your side of the net)
    if z_far - z_near > 0.1:
        ring = np.array(cam.ground_ellipse(x, (z_far + z_near) / 2, s.reach_m, (z_far - z_near) / 2), dtype=np.int32)
        _blend_poly(frame, ring, REACH, 0.12)
        cv2.polylines(frame, [ring], True, REACH, 2, cv2.LINE_AA)
        line = _pts(cam, (x - s.reach_m, 0, z), (x + s.reach_m, 0, z))
        cv2.line(frame, tuple(int(v) for v in line[0]), tuple(int(v) for v in line[1]), REACH, 2, cv2.LINE_AA)
    px, _, pz = s.paddle
    if _on_table(px, pz):
        _blend_poly(frame, cam.ground_circle(px, pz, PADDLE_FACE_M * 1.1), SHADOW, 0.55)


# --- the things standing in the scene -------------------------------------------------------------------------------------
def _ball(frame, cam, ball):
    px, py, sc = cam.project(*ball)
    r = max(4, round(BALL_R_M * sc))
    cv2.circle(frame, (round(px), round(py)), r + 2, BALL_RIM, -1, cv2.LINE_AA)
    cv2.circle(frame, (round(px), round(py)), r, BALL, -1, cv2.LINE_AA)
    cv2.circle(frame, (round(px - 0.35 * r), round(py - 0.35 * r)), max(1, round(0.22 * r)), (235, 250, 255), -1, cv2.LINE_AA)


def _player_paddle(frame, cam, s):
    px, py, sc = cam.project(*s.paddle)
    r = max(6, round(PADDLE_FACE_M * sc))
    skin = cast.player_look(s.player_name).skin if s.player_name else canvas.SKIN             # your own hand, once you have a name
    canvas.draw_paddle(frame, round(px), round(py), r, angle_deg=s.paddle_angle, hand=True, skin=skin)


def _cpu_pose(cam, s):
    """Where the computer's paddle is on the picture: (x, y, face radius, angle), and the opponent that holds it."""
    px, py, sc = cam.project(s.cpu_x_m, CPU_Y_M, CPU_Z_M)
    r = max(5, round(CPU_FACE_M * sc))
    angle = 0.0 if s.cpu_swing is None else 55.0 * math.sin(math.pi * s.cpu_swing)
    return px, py, r, angle, cast.opponent_look(s.level_name, s.opponent_name)


def draw_opponent(frame, cam, s, *, clip_rows=None):
    """The opponent behind the far end of the table: its body (to the first `clip_rows` rows of the picture, when the table's
    far edge is where it ends), its face, and its racket arm reaching the paddle's grip."""
    px, py, r, angle, look = _cpu_pose(cam, s)
    if cam.depth_of(s.cpu_x_m, CPU_Y_M, OPPONENT_Z_M) < 1.0:
        return
    characters.draw_opponent(frame, cam, look, x_m=s.cpu_x_m, z_m=OPPONENT_Z_M,
                             grip_px=canvas.paddle_grip(px, py, r, angle, handle_up=True), mood=s.cpu_mood, t=s.anim_t, body_rows=clip_rows)


def draw_cpu_paddle(frame, cam, s):
    px, py, r, angle, look = _cpu_pose(cam, s)
    canvas.draw_paddle(frame, round(px), round(py), r, rubber=canvas.RUBBER_BLACK, angle_deg=angle, handle_up=True, hand=True,
                       skin=look.skin)


def _cpu_paddle(frame, cam, s):
    """The opponent and its paddle, where the computer has it: it waits where it hit, chases your shot, and flicks when it hits;
    its face shows how it feels about the score.  Its body ends at the table's far edge."""
    draw_opponent(frame, cam, s, clip_rows=round(cam.project(0.0, 0.0, L)[1]))
    draw_cpu_paddle(frame, cam, s)
