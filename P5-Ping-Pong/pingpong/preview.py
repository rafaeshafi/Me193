"""A live, mirrored camera window for the bench tools, so you can see whether you are in position.

macOS draws an OpenCV window only while the MAIN thread keeps calling cv2.waitKey: a window that is created and then
left alone while the tool waits at an Enter prompt (or sleeps through a measurement) stays blank and keeps the Python
icon bouncing in the Dock.  So every wait goes through Preview.update(): the tool's environment calls it from
sleep(), and the prompts below poll the terminal instead of blocking in input().

The window takes the keyboard focus when it opens, so a keypress may land in it instead of the terminal: the prompts
take their answer from either place.
"""

import select
import sys
import time

import cv2

from pingpong import canvas
from pingpong.pose import MIN_VISIBILITY

WIDTH = 960
GREEN, RED, ORANGE = (80, 220, 80), (60, 60, 230), (0, 160, 255)
PARTS = (("head", (0,)), ("shoulders", (11, 12)), ("hips", (23, 24)))     # what has to be in view to play
JOINTS = (0, 11, 12, 15, 16, 23, 24)                                      # head, shoulders, wrists, hips
WAIT_HINT = "   >> press ENTER when ready"
ENTER_KEYS = {13: "", 10: "", 32: ""}                                     # Return / Space in the window
YES_NO_KEYS = {ord("y"): "y", ord("n"): "n"}
NO_KEY = 255                                                              # waitKey() & 0xFF when nothing was pressed


def _seen(point, min_visibility):
    return point.visibility >= min_visibility and 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0


def missing_parts(landmarks, min_visibility=MIN_VISIBILITY):
    """The body parts not clearly in the picture; [] means head, shoulders and hips are all in view."""
    if landmarks is None:
        return ["person"]
    return [name for name, ids in PARTS if not all(_seen(landmarks[i], min_visibility) for i in ids)]


def compose(frame, caption, landmarks, width=WIDTH):
    """The camera frame as a mirror shows it (moving right moves right), the joints the game reads, a caption on
    top and at the bottom whether you are in position."""
    h, w = frame.shape[:2]
    image = cv2.flip(cv2.resize(frame, (width, round(h * width / w))), 1)
    ih = image.shape[0]
    if landmarks is not None:
        for i in JOINTS:
            point = landmarks[i]
            color = GREEN if _seen(point, MIN_VISIBILITY) else RED
            cv2.circle(image, (round((1.0 - point.x) * width), round(point.y * ih)), 7, color, -1, cv2.LINE_AA)
    canvas.draw_fitted(image, caption, width // 2, 34, width - 40, max_scale=0.9, min_scale=0.6, thickness=2,
                       max_lines=2)
    missing = missing_parts(landmarks)
    if not missing:
        text, color = "IN POSITION", GREEN
    elif missing == ["person"]:
        text, color = "NO PERSON IN VIEW", ORANGE
    else:
        text, color = "NOT IN POSITION: can't see your " + ", ".join(missing), ORANGE
    canvas.draw_text(image, text, (width // 2, ih - 22), 1.0, color, 2, anchor="center")
    return image


class Preview:
    """The window: call update() from the main thread as often as you can; it redraws at most `fps` times a second."""

    def __init__(self, show, *, wait_key=None, now=time.monotonic, fps=25.0):
        self.show, self.now, self.period = show, now, 1.0 / fps
        self.wait_key = wait_key or (lambda ms: cv2.waitKey(ms))        # looked up per call, so a test can swap it
        self.vision, self.caption, self.live, self._next = None, "", True, 0.0

    @property
    def active(self):
        return self.vision is not None

    def attach(self, vision):
        self.vision = vision

    def update(self):
        """Draw the newest frame and let the window handle events -> the key pressed in it (NO_KEY if none).

        A window that cannot draw is dropped, never allowed to stop the measurement it only decorates."""
        if self.vision is None:
            return NO_KEY
        try:
            t = self.now()
            if self.live and t >= self._next:
                frame = self.vision.latest_frame()
                if frame is not None:
                    self.show(compose(frame, self.caption, self.vision.last_landmarks))
                    self._next = t + self.period
            return self.wait_key(1) & 0xFF
        except cv2.error as exc:
            self.vision = None
            print(f"preview window disabled: {exc}")
            return NO_KEY

    def show_still(self, image):
        """A picture that stays up (the HUD to judge): live drawing stops."""
        self.live = False
        self.show(image)
        self.wait_key(1)


def make_prompts(window, *, stdin=None, poll_s=0.03, out=print):
    """-> (prompt, ask) for a bench tool's steps.  With the window up, both keep it drawing while they wait and take
    the answer from the terminal (a typed line) or from the window (Enter / Space, or y / n); without one they are
    plain input() questions."""

    def wait(text, accept, hint):
        instruction = text.replace("Press Enter.", "").strip()
        stream = stdin or sys.stdin
        out(text + "\n> ", end="", flush=True)
        window.caption = instruction + hint
        while True:
            key = window.update()
            if key in accept:
                answer = accept[key]
                out("")
                break
            if select.select([stream], [], [], poll_s)[0]:
                answer = stream.readline()
                break
        window.caption = instruction
        return answer

    def prompt(text):
        if window.active:
            wait(text, ENTER_KEYS, WAIT_HINT)
        else:
            input(text + "\n> ")

    def ask(text):
        answer = wait(text, YES_NO_KEYS, "") if window.active else input(text + "\n> ")
        return answer.strip().lower() == "y"

    return prompt, ask
