"""The intro: out of a flash of sunlight, the camera comes in from high above the clouds, sweeps over the island and its palms, along
the pier and down onto the terrace, and settles exactly where the game looks at the court from; the title comes up on that picture.

The whole flight is drawn live by the game's own renderer (resort.py, scene.py), so it is the very court you are about to play on,
and it can be written out as a video to show somebody (./pp intro --video).  While the camera is far away the world is drawn at half
size and enlarged (soft and quick, like a postcard of a sunny day); the last seconds are drawn at full size.
"""

import dataclasses

import cv2
import numpy as np

from pingpong import anim, court3d, flow, fonts, resort, scene, ui
from pingpong.cast import rgb

LENGTH_S = flow.INTRO_S
ARRIVE_S = 9.0                      # the camera is on the game's own view from here to the end
NAVY, WHITE, ORANGE, ORANGE_DARK = rgb(24, 48, 96), (255, 255, 255), rgb(255, 168, 40), rgb(240, 120, 20)
BAR_FRACTION = 0.09                 # the black bars of the cinema picture, top and bottom, until they slide away
SKIP_AFTER_S = 1.5

# time s, x, y, z (metres), yaw, pitch (degrees), the row of the optical centre in a 720 px picture
KEYS = (
    (0.0, -46.0, 72.0, -236.0, 14.0, 17.0, 330.0),
    (2.6, -24.0, 40.0, -146.0, 10.0, 18.0, 342.0),
    (5.0, -9.0, 15.0, -66.0, 5.0, 16.0, 352.0),
    (7.2, -3.0, 4.8, -14.0, 1.5, 24.0, 372.0),
    (ARRIVE_S, *court3d.POS, 0.0, court3d.PITCH_DEG, court3d.CENTRE_Y),
)


def _hermite(values, times, t):
    """A smooth curve through the key values: it leaves the first at rest, comes to rest at the last, and in between its slope is
    the average slope across each key (so it never overshoots a straight descent)."""
    t = min(max(t, times[0]), times[-1])
    i = max(0, min(len(times) - 2, int(np.searchsorted(times, t, side="right")) - 1))
    dt = times[i + 1] - times[i]
    u = (t - times[i]) / dt

    def tangent(k):
        if k == 0 or k == len(times) - 1:
            return 0.0
        return (values[k + 1] - values[k - 1]) / (times[k + 1] - times[k - 1])

    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * values[i] + h10 * dt * tangent(i) + h01 * values[i + 1] + h11 * dt * tangent(i + 1)


def camera_at(t_s, w, h):
    """The camera t_s seconds into the flight, for a picture w by h."""
    times = [k[0] for k in KEYS]
    x, y, z, yaw, pitch, cy = (_hermite([k[column] for k in KEYS], times, t_s) for column in range(1, 7))
    return dataclasses.replace(court3d.Camera.for_frame(w, h), pos=(x, y, z), yaw_deg=yaw, pitch_deg=pitch, cy=cy * h / court3d.REF_H)


def _world(frame, cam, s, t_s):
    """The island, the terrace and the court drawn through the camera, in the order they stand."""
    resort.draw_backdrop(frame, cam, t_s)
    scene.draw_opponent(frame, cam, s)
    scene.draw_table(frame, cam)
    scene.draw_net(frame, cam)
    scene.draw_cpu_paddle(frame, cam, s)
    resort.draw_foreground(frame, cam, t_s)


def _flash(frame, t_s):
    alpha = 0.9 * (1.0 - anim.smoothstep(t_s / 0.9))
    if alpha > 0.01:
        cv2.addWeighted(np.full_like(frame, 255), alpha, frame, 1.0 - alpha, 0, frame)


def _logo(frame, t_s):
    """The name of the game over the first shots, popping in and fading out as the camera comes down."""
    w, h = frame.shape[1], frame.shape[0]
    shown = anim.pop(anim.progress(t_s, start=0.5, length=0.6)) * (1.0 - anim.smoothstep(anim.progress(t_s, start=2.8, length=0.7)))
    if shown < 0.02:
        return
    opacity, k = min(1.0, shown), max(0.05, shown)
    fonts.draw(frame, "PING-PONG", w / 2, h * 0.36, round(h * 0.17 * k), WHITE, outline=NAVY, outline_px=max(2, round(12 * k * h / 720)),
               shadow=(0, round(9 * h / 720), NAVY, 0.3), opacity=opacity)
    ui.pill(frame, w / 2, h * 0.36 + 0.14 * h * k + 0.05 * h, 0.25 * w * k, 0.095 * h * k, "ISLAND", fill=(ORANGE, ORANGE_DARK), outline=WHITE,
            size=round(0.06 * h * k), opacity=opacity)


def _bars(frame, t_s):
    height = round(BAR_FRACTION * frame.shape[0] * (1.0 - anim.smoothstep(anim.progress(t_s, start=ARRIVE_S, length=LENGTH_S - ARRIVE_S))))
    if height > 0:
        frame[:height] = 0
        frame[-height:] = 0
    return height


def _skip_hint(frame, u, bar):
    if u.t_s < SKIP_AFTER_S:
        return
    w, h = frame.shape[1], frame.shape[0]
    text = "hold the hub in the top right to skip"
    size = round(0.032 * h)
    tw = fonts.measure(text, size)[0]
    ui.pill(frame, w - 30 - (tw + 56) / 2, bar + 0.07 * h, tw + 56, 0.068 * h, text, fill=(rgb(255, 255, 255), rgb(224, 236, 250)),
            fill_progress=ORANGE, progress=u.progress if u.hover == "skip" else 0.0, text=NAVY, size=size, gloss=False, opacity=0.9)


def render(frame, state):
    """The intro as it looks `state.ui.t_s` seconds in."""
    u, (w, h) = state.ui, (frame.shape[1], frame.shape[0])
    t = u.t_s
    s = dataclasses.replace(state, phase="LOBBY", ball=None, paddle=None, rest=None, zone=None, cpu_swing=None, cpu_mood="happy", screen="GAME",
                            ui=None, results=None, cpu_x_m=0.0, anim_t=t)
    if t >= ARRIVE_S:
        scene.draw_scene(frame, camera_at(ARRIVE_S, w, h), s)
    else:
        scale = anim.lerp(0.5, 1.0, anim.smoothstep(anim.progress(t, start=6.5, length=2.5)))
        iw, ih = max(2, round(w * scale)), max(2, round(h * scale))
        picture = np.empty((ih, iw, 3), dtype=np.uint8)
        _world(picture, camera_at(t, iw, ih), s, t)
        cv2.resize(picture, (w, h), dst=frame, interpolation=cv2.INTER_LINEAR if scale < 1.0 else cv2.INTER_AREA)
    _logo(frame, t)
    bar = _bars(frame, t)
    _skip_hint(frame, u, bar)
    _flash(frame, t)


def export_video(path, *, size=(1280, 720), fps=30, seconds=None, start_s=0.0):
    """Write the intro (or `seconds` of it from start_s) as a video file; -> how many frames."""
    from pingpong import hud
    from pingpong.uistate import UiState

    n = round((LENGTH_S - start_s if seconds is None else seconds) * fps)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"cannot write a video to {path}")
    try:
        for k in range(n):
            t = start_s + k / fps
            writer.write(hud.render(hud.HudState(screen="INTRO", ui=UiState(screen="INTRO", t_s=t), level_name="Club", anim_t=t), size=size))
    finally:
        writer.release()
    return n
