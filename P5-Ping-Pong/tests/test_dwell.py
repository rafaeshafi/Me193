"""Pointing with the hub and holding on a button to press it: the engine behind every menu (and the START button).

The hand is a cursor over the whole screen: its place in the reach box, (a, b) in 0..1 across and up, is where the cursor stands.
A target is a rectangle of the screen (fractions, y down) with the time it takes to press it."""

import pytest

from pingpong import dwell

S = 1_000_000_000
STEP = S // 60
LEFT = ("left", (0.1, 0.4, 0.4, 0.7), 1.0)                      # id, (x0, y0, x1, y1), hold seconds
RIGHT = ("right", (0.6, 0.4, 0.9, 0.7), 1.5)
CORNER = ("corner", (0.8, 0.0, 1.0, 0.3), 1.0)
AT_LEFT, AT_RIGHT, ELSEWHERE = (0.25, 0.45), (0.75, 0.45), (0.5, 0.9)        # (a, b): b is up, so y = 1 - b


class Clockwork:
    def __init__(self, targets=(LEFT, RIGHT)):
        self.pointer, self.targets, self.t = dwell.Pointer(), targets, 0

    def feed(self, ab, seconds, active=True):
        """-> the id pressed during this stretch, if any (and when, in seconds into the feed)."""
        fired = None
        for _ in range(round(seconds * 60)):
            self.t += STEP
            got = self.pointer.update(self.t, ab, self.targets, active)
            if got and fired is None:
                fired = (got, self.t / S)
        return fired


def test_a_target_is_a_rectangle_of_the_screen_and_the_hand_is_a_cursor_over_the_whole_screen():
    hit = dwell.hit_test
    assert hit(AT_LEFT, (LEFT, RIGHT)) == "left" and hit(AT_RIGHT, (LEFT, RIGHT)) == "right"
    assert hit(ELSEWHERE, (LEFT, RIGHT)) is None and hit(None, (LEFT, RIGHT)) is None
    assert hit((0.25, 0.7), (LEFT, RIGHT)) is None                 # b = 0.7 is y = 0.3: above the rectangle


def test_overshooting_the_edge_of_the_reach_counts_as_the_edge_so_a_button_in_the_corner_can_be_reached():
    assert dwell.hit_test((1.4, 1.3), (CORNER,)) == "corner"       # past the top right of the reach is the top right of the screen
    assert dwell.hit_test((0.95, 0.95), (CORNER,)) == "corner"
    assert dwell.hit_test((1.4, -0.4), (CORNER,)) is None          # ... and past the bottom right is not on it


def test_the_first_target_wins_where_two_overlap():
    both = (("a", (0.0, 0.0, 0.6, 1.0), 1.0), ("b", (0.4, 0.0, 1.0, 1.0), 1.0))
    assert dwell.hit_test((0.5, 0.5), both) == "a" and dwell.hit_test((0.9, 0.5), both) == "b"


def test_holding_on_a_target_for_its_time_presses_it_and_says_which():
    c = Clockwork()
    assert c.feed(ELSEWHERE, 0.3) is None                          # the hand was somewhere else first
    assert c.feed(AT_LEFT, 0.9) is None
    got = c.feed(AT_LEFT, 0.3)
    assert got is not None and got[0] == "left" and 1.25 <= got[1] <= 1.35


def test_each_target_has_its_own_time():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    assert c.feed(AT_RIGHT, 1.4) is None
    assert c.feed(AT_RIGHT, 0.2)[0] == "right"


def test_the_progress_and_the_hovered_target_follow_the_hand():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    assert c.pointer.hovered is None and c.pointer.progress == 0.0
    c.feed(AT_RIGHT, 0.75)
    assert c.pointer.hovered == "right" and c.pointer.progress == pytest.approx(0.5, abs=0.03)
    c.feed(ELSEWHERE, 0.5)                                         # longer than the grace
    assert c.pointer.hovered is None and c.pointer.progress == 0.0


def test_sliding_from_one_target_to_another_starts_the_new_hold_from_nothing():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    c.feed(AT_LEFT, 0.8)
    assert c.pointer.progress > 0.7
    c.feed(AT_RIGHT, 0.1)
    assert c.pointer.hovered == "right" and c.pointer.progress < 0.15
    assert c.feed(AT_RIGHT, 1.0) is None and c.feed(AT_RIGHT, 0.5)[0] == "right"


def test_a_flicker_shorter_than_the_grace_neither_undoes_the_hold_nor_counts_as_holding():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    assert c.feed(AT_LEFT, 0.6) is None
    assert c.feed(None, 0.2) is None                               # the reading dropped for 0.2 s
    assert c.feed(AT_LEFT, 0.2) is None                            # 0.8 s of holding so far
    assert c.feed(AT_LEFT, 0.3)[0] == "left"


def test_a_hand_that_is_not_seen_for_longer_than_the_grace_is_a_hand_that_left():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    c.feed(AT_LEFT, 0.8)
    c.feed(None, 0.6)
    assert c.pointer.progress == 0.0


def test_a_hand_already_on_a_target_when_the_screen_comes_up_has_to_leave_and_come_back_first():
    c = Clockwork()
    assert c.feed(AT_LEFT, 5.0) is None and c.pointer.progress == 0.0
    c.feed(ELSEWHERE, 0.1)
    assert c.feed(AT_LEFT, 1.2)[0] == "left"


def test_a_new_screen_has_to_be_armed_again_even_if_the_hand_was_not_on_a_target_before():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    c.pointer.reset()                                              # the screen changed
    assert c.feed(AT_LEFT, 3.0) is None
    c.feed(ELSEWHERE, 0.1)
    assert c.feed(AT_LEFT, 1.2)[0] == "left"


def test_an_inactive_screen_forgets_everything_and_needs_arming_afterwards():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    c.feed(AT_LEFT, 0.8)
    assert c.feed(AT_LEFT, 3.0, active=False) is None
    assert c.pointer.progress == 0.0
    assert c.feed(AT_LEFT, 2.0) is None


def test_a_stall_between_two_updates_is_not_held_time():
    p = dwell.Pointer()
    p.update(0, ELSEWHERE, (LEFT,), True)
    p.update(S // 60, AT_LEFT, (LEFT,), True)
    assert p.update(5 * S, AT_LEFT, (LEFT,), True) is None         # five seconds with no frame in between
    assert p.progress < 0.2


def test_after_a_press_the_pointer_has_to_be_armed_again_so_one_hold_is_one_press():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.2)
    assert c.feed(AT_LEFT, 1.2)[0] == "left"
    assert c.feed(AT_LEFT, 3.0) is None


def test_a_restart_starts_the_hold_again_from_nothing_but_the_hand_stays_armed():
    c = Clockwork()
    c.feed(ELSEWHERE, 0.3)
    assert c.feed(AT_LEFT, 0.7) is None and c.pointer.progress > 0.5
    c.pointer.restart()
    assert c.pointer.progress == 0.0 and c.pointer.armed and c.pointer.hovered == "left"
    assert c.feed(AT_LEFT, 0.7) is None
    assert c.feed(AT_LEFT, 0.5)[0] == "left"
