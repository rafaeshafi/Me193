"""AprilTag (36h11) detection and the voting that turns frames into START / LEVEL events.

A single detection frame is never trusted: a card counts only when it is in >= 4 of the last 6 looks AND has been held up for
HOLD_S (2 s) since the first look that saw it.  It counts once (it must be taken away to count again, plus a 1.5 s lockout), and
only in the lobby / after a finished game.  Card 0 (START) starts a game with the choices so far; a LEVEL card (1-3) says
which opponent, and the game starts against it.  Changing the card in the hand starts the hold again.  Unallocated ids (anything
but 0, 1-3 here) are ignored.  Detection itself is classical vision: threshold -> square contours -> decode the Hamming-protected
bit grid.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2
from cv2 import aruco

from pingpong.events import TagEvent

START_ID = 0
LEVEL_IDS = (1, 2, 3)
CARD_IDS = (START_ID,) + LEVEL_IDS
HOLD_S = 2.0                                   # how long a card is held up before it counts
HOLD_PHASES = ("LOBBY", "MATCH_OVER")
STALE_S = 0.5                                  # looks come several times a second while the lobby is up: none for this long ends a hold


@dataclass(frozen=True)
class Tag:
    id: int
    corners: tuple          # four (x, y) points
    center: tuple
    angle_deg: float        # angle of the top edge (a roll dial, if ever wanted)


class TagDetector:
    def __init__(self, family=aruco.DICT_APRILTAG_36h11, inverted=True):
        params = aruco.DetectorParameters()
        params.detectInvertedMarker = inverted                 # also tags shown on a dark-mode phone
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(family), params)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        tags = []
        for tag_id, c in zip(ids.flatten(), corners):
            pts = c.reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            dx, dy = pts[1] - pts[0]
            tags.append(Tag(int(tag_id), tuple(map(tuple, pts.tolist())), (float(cx), float(cy)),
                            math.degrees(math.atan2(dy, dx))))
        return tags


class TagVoter:
    def __init__(self, window=6, need=4, hold_s=HOLD_S, lockout_s=1.5):
        self.need, self.hold_ns, self.lockout_ns = need, round(hold_s * 1e9), round(lockout_s * 1e9)
        self._frames = deque(maxlen=window)              # (time, the ids seen) of the latest looks
        self._card = self._since = None                  # the card voted now, and when the first look that saw it was
        self._fired = None                               # the card that has counted and has not been taken away since
        self._lockout_until = 0
        self._hold = None                                # (card, since, last look): what the screen is shown, swapped whole

    def update(self, t_ns, ids, phase):
        """Feed the tag ids seen in one frame; returns the TagEvents (if any) it triggers."""
        self._frames.append((t_ns, frozenset(ids)))
        card = self._voted()
        if card is None:
            self._card = self._since = self._fired = None
        elif card != self._card:
            self._card = card
            self._since = next(t for t, seen in self._frames if card in seen)
        events = self._count(t_ns, phase)
        shown = self._card is not None and self._card != self._fired and phase in HOLD_PHASES
        self._hold = (self._card, self._since, t_ns) if shown else None
        return events

    def _voted(self):
        """The card in >= `need` of the latest looks (the one with most votes); None when there is none, or a tie."""
        votes = Counter(i for _, seen in self._frames for i in seen if i in CARD_IDS)
        ranked = sorted(((n, i) for i, n in votes.items() if n >= self.need), reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return None                                  # nothing voted, or a tie: do not guess
        return ranked[0][1]

    def _count(self, t_ns, phase):
        if self._card is None or self._card == self._fired:
            return []
        if phase not in HOLD_PHASES:
            self._fired = self._card                     # held through a game: take it away first
            return []
        if t_ns < self._lockout_until or t_ns - self._since < self.hold_ns:
            return []
        self._fired, self._lockout_until = self._card, t_ns + self.lockout_ns
        return [TagEvent("START" if self._card == START_ID else "LEVEL", self._card, t_ns)]

    def hold(self, now_ns):
        """(card, how far the hold has come 0..1) while a card is being held up for a game, else None.  The screen draws at 60 Hz
        and looks come at ~10, so the bar is timed from `now_ns`, not from the last look."""
        hold = self._hold
        if hold is None or now_ns - hold[2] > round(STALE_S * 1e9):
            return None
        return hold[0], min(1.0, max(0.0, (now_ns - hold[1]) / self.hold_ns))
