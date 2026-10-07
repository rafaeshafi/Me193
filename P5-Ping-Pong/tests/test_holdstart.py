"""Start by holding the hub on the button in the top right of the screen: no keyboard, you are 1.8 m from it.

The hand is a cursor over the whole screen: its place in the calibrated reach box, (a, b) in 0..1 across and up, is where
the cursor stands, so reaching to the top right of your reach is reaching the button."""

import pytest

from pingpong import holdstart
from pingpong.holdstart import HoldStart

S = 1_000_000_000
STEP = S // 60
CORNER, MIDDLE = (0.95, 0.9), (0.5, 0.5)


class Clockwork:
    """Feeds one reading at 60 Hz and says when (seconds into the feed) the hold completed."""

    def __init__(self, hold=None):
        self.hold, self.t = hold or HoldStart(), 0

    def feed(self, ab, seconds, active=True):
        fired = None
        for _ in range(round(seconds * 60)):
            self.t += STEP
            if self.hold.update(self.t, ab, active) and fired is None:
                fired = self.t / S
        return fired


def test_the_button_is_the_top_right_of_the_reach_and_overshooting_it_still_counts():
    on = HoldStart.on_button
    assert on((0.95, 0.9)) and on((1.3, 1.4)) and on((0.82, 0.70))
    assert not on((0.5, 0.9)) and not on((0.95, 0.5)) and not on((0.79, 0.9)) and not on((0.95, 0.66)) and not on(None)


def test_the_button_is_drawn_where_the_hold_counts():
    x, y, w, h = holdstart.BUTTON                                  # fractions of the screen, y down
    assert holdstart.A_MIN == pytest.approx(x) and holdstart.B_MIN == pytest.approx(1.0 - (y + h))
    assert x + w <= 1.0 and y > 84 / 720                           # inside the frame, below the top bar


def test_holding_the_hand_on_the_button_for_one_and_a_half_seconds_starts_the_game():
    c = Clockwork()
    assert c.feed(MIDDLE, 0.3) is None                             # the hand was somewhere else first
    assert c.feed(CORNER, 1.4) is None
    fired = c.feed(CORNER, 0.3)
    assert fired is not None and 1.69 <= fired <= 1.82             # 0.3 s + 1.5 s, to the frame


def test_the_progress_fills_while_holding_and_empties_when_the_hand_goes_away_for_good():
    c = Clockwork()
    c.feed(MIDDLE, 0.2)
    c.feed(CORNER, 0.75)
    assert c.hold.progress() == pytest.approx(0.5, abs=0.03) and c.hold.inside
    c.feed(MIDDLE, 0.5)                                            # longer than the grace
    assert c.hold.progress() == 0.0 and not c.hold.inside


def test_a_flicker_shorter_than_the_grace_neither_undoes_the_hold_nor_counts_as_holding():
    c = Clockwork()
    c.feed(MIDDLE, 0.2)
    assert c.feed(CORNER, 1.0) is None
    assert c.feed(None, 0.2) is None                               # the reading dropped for 0.2 s
    assert c.feed(CORNER, 0.4) is None                             # 1.4 s of holding so far
    fired = c.feed(CORNER, 0.2)
    assert fired is not None and fired == pytest.approx(0.2 + 1.0 + 0.2 + 0.5 + 0.0, abs=0.08)


def test_leaving_for_longer_than_the_grace_starts_over():
    c = Clockwork()
    c.feed(MIDDLE, 0.2)
    c.feed(CORNER, 1.2)
    c.feed(MIDDLE, 0.5)
    assert c.feed(CORNER, 1.4) is None                             # not 1.2 + 1.4
    assert c.feed(CORNER, 0.3) is not None


def test_a_hand_that_is_not_seen_is_a_hand_that_left():
    c = Clockwork()
    c.feed(MIDDLE, 0.2)
    c.feed(CORNER, 1.0)
    c.feed(None, 0.6)
    assert c.hold.progress() == 0.0


def test_a_hand_already_on_the_button_when_the_screen_comes_up_has_to_leave_and_come_back_first():
    # the end screen shows your result: a hand that happened to end the last game up there must not restart it
    c = Clockwork()
    assert c.feed(CORNER, 5.0) is None and c.hold.progress() == 0.0
    c.feed(MIDDLE, 0.1)
    assert c.feed(CORNER, 1.7) is not None


def test_only_the_lobby_and_the_end_screen_count_and_leaving_them_forgets_everything():
    c = Clockwork()
    c.feed(MIDDLE, 0.2)
    c.feed(CORNER, 1.0)
    assert c.feed(CORNER, 3.0, active=False) is None               # a game is on
    assert c.hold.progress() == 0.0
    assert c.feed(CORNER, 2.0) is None                             # and it needs arming again afterwards


def test_a_stall_between_two_updates_is_not_held_time():
    hold = HoldStart()
    hold.update(0, MIDDLE, True)
    hold.update(S // 60, CORNER, True)
    assert hold.update(5 * S, CORNER, True) is False               # five seconds with no frame in between
    assert hold.progress() < 0.2
