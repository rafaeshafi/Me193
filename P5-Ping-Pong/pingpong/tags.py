"""AprilTag (36h11) detection and the voting that turns frames into START / LEVEL events.

A single detection frame is never trusted: START needs the card in >= 4 of the last
6 frames AND present continuously for 0.4 s, fires once per presentation (the card
must be removed to re-arm, plus a 1.5 s lockout), and only in the lobby / after a
finished game.  LEVEL cards (1-3) latch once voted, only between rallies.  Unallocated
ids (anything but 0, 1-3 here) are ignored.  Detection itself is classical vision:
threshold -> square contours -> decode the Hamming-protected bit grid.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2
from cv2 import aruco

from pingpong.events import TagEvent

START_ID = 0
LEVEL_IDS = (1, 2, 3)
START_PHASES = ("LOBBY", "MATCH_OVER")
LEVEL_PHASES = ("LOBBY", "POINT_OVER", "MATCH_OVER")


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
    def __init__(self, window=6, need=4, hold_s=0.4, lockout_s=1.5):
        self.need, self.hold_ns, self.lockout_ns = need, round(hold_s * 1e9), round(lockout_s * 1e9)
        self._frames = deque(maxlen=window)
        self._start_since = None
        self._armed = True
        self._lockout_until = 0
        self._level = None

    def update(self, t_ns, ids, phase):
        """Feed the tag ids seen in one frame; returns the TagEvents (if any) it triggers."""
        self._frames.append(frozenset(ids))
        votes = Counter(i for frame in self._frames for i in frame)
        events = []
        events += self._start(t_ns, votes, phase)
        events += self._level_card(t_ns, votes, phase)
        return events

    def _start(self, t_ns, votes, phase):
        present = votes[START_ID] >= self.need
        if not present:
            self._start_since, self._armed = None, True
            return []
        if phase not in START_PHASES:
            self._start_since, self._armed = None, False        # held through a rally: remove it first
            return []
        if self._start_since is None:
            self._start_since = t_ns
        if self._armed and t_ns >= self._lockout_until and t_ns - self._start_since >= self.hold_ns:
            self._armed, self._lockout_until = False, t_ns + self.lockout_ns
            return [TagEvent("START", START_ID, t_ns)]
        return []

    def _level_card(self, t_ns, votes, phase):
        if phase not in LEVEL_PHASES:
            return []
        ranked = sorted(((votes[i], i) for i in LEVEL_IDS if votes[i] >= self.need), reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return []                                           # nothing voted, or a tie: do not guess
        level = ranked[0][1]
        if level == self._level:
            return []
        self._level = level
        return [TagEvent("LEVEL", level, t_ns)]
