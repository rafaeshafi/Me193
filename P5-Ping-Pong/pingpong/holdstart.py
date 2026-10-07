"""Start the game by holding the hub on a button in the top right of the screen: no keyboard, you stand 1.8 m from it.

Your hand is a cursor over the whole screen: its place in the calibrated reach box (a across, b up, both 0..1) is where the
cursor stands, so reaching to the top right of your reach is reaching the button.  Holding it there for HOLD_S starts the game
(from the lobby, or from the end screen to play again).  Pure logic with the clock passed in; hud.py draws the button where
BUTTON says and the Session decides when it counts.
"""

HOLD_S = 1.5
GRACE_S = 0.3             # the hand (or its reading) may drop out this long without the hold starting over
MAX_STEP_S = 0.1          # a stall between two updates is not time held
FRESH_S = 0.2             # a reading older than this is no reading (so a lost hand cannot finish a hold from its last place)
MIN_CONF = 0.5
PHASES = ("LOBBY", "MATCH_OVER")
# x, y, width, height as fractions of the screen (y down): 1034, 96, 230 x 132 px at 1280 x 720, where the right score panel sits
BUTTON = (1034 / 1280, 96 / 720, 230 / 1280, 132 / 720)
A_MIN = BUTTON[0]                                    # the hand counts as on the button from its lower left corner up and right,
B_MIN = 1.0 - (BUTTON[1] + BUTTON[3])                # so reaching past it (up, or out of the picture) is fine


def hand_ab(box, poses, now_ns):
    """The newest reading's (a, b) in the reach box, or None when there is none, it is unsure, or it is old."""
    if not poses:
        return None
    pose = poses[-1]
    if pose.conf < MIN_CONF or now_ns - pose.t_scene_ns > round(FRESH_S * 1e9):
        return None
    return box.to_ab(pose.u, pose.v)


class HoldStart:
    def __init__(self, hold_s=HOLD_S, grace_s=GRACE_S):
        self.hold_ns, self.grace_ns, self.max_step_ns = round(hold_s * 1e9), round(grace_s * 1e9), round(MAX_STEP_S * 1e9)
        self.reset()

    def reset(self):
        self._t = None
        self._held_ns = self._away_ns = 0
        self.armed = False            # the hand has been somewhere else since the screen came up: only then is a hold on purpose
        self.inside = False

    @staticmethod
    def on_button(ab):
        return ab is not None and ab[0] >= A_MIN and ab[1] >= B_MIN

    def progress(self):
        return min(1.0, self._held_ns / self.hold_ns)

    def update(self, now_ns, ab, active):
        """-> True at the moment the hold is complete.  ab: the hand's (a, b) in the reach box, None when it is not seen;
        active: the screen is one the button works on (the lobby, the end screen)."""
        if not active:
            self.reset()
            return False
        dt = 0 if self._t is None else max(0, now_ns - self._t)
        self._t = now_ns
        self.inside = self.on_button(ab)
        if not self.inside:
            self.armed = self.armed or ab is not None
            self._away_ns += dt
            if self._away_ns > self.grace_ns:
                self._held_ns = 0
            return False
        self._away_ns = 0
        if not self.armed:
            return False
        self._held_ns += min(dt, self.max_step_ns)
        if self._held_ns >= self.hold_ns:
            self.reset()
            return True
        return False
