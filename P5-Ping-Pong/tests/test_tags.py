"""TagVoter (4-of-6 vote, hold, lockout, phase gating) and the 36h11 detector."""

import cv2
import numpy as np
import pytest

from pingpong.tags import TagDetector, TagVoter
from tools import make_cards

MS = 1_000_000
FPS = 15
DT = int(1e9 / FPS)


def run(voter, frames, phase="LOBBY", t0=0):
    """frames: list of id-sets, one per detection frame at 15 fps; returns [(t, event)]."""
    out = []
    for i, ids in enumerate(frames):
        t = t0 + i * DT
        for e in voter.update(t, set(ids), phase):
            out.append((t, e))
    return out


def seen(n, tag=0):
    return [{tag}] * n


def gap(n):
    return [set()] * n


S = 1_000_000_000
HOLD_FRAMES = 40                                                    # ~2.7 s of looks: comfortably more than the hold


@pytest.mark.parametrize("card, role", [(0, "START"), (1, "LEVEL"), (2, "LEVEL"), (3, "LEVEL")])
def test_a_card_counts_once_it_has_been_held_up_for_two_seconds(card, role):
    events = run(TagVoter(), gap(5) + seen(HOLD_FRAMES, card))
    assert [(e.role, e.value) for _, e in events] == [(role, card)]
    held_for = events[0][0] - 5 * DT                                # from the first look that saw it
    assert 2.0 * S <= held_for <= 2.0 * S + 2 * DT


def test_a_card_shown_for_less_than_two_seconds_never_counts():
    assert run(TagVoter(), gap(5) + seen(28) + gap(10)) == []       # ~1.9 s
    assert run(TagVoter(), gap(5) + seen(10, 2) + gap(10)) == []    # a flash


def test_a_card_swept_across_the_view_is_not_a_card_held_up():
    # in view for under a second, three times over, each time taken away: the holds never add up
    assert run(TagVoter(), (gap(10) + seen(14, 3)) * 3) == []


def test_one_dropped_frame_does_not_break_a_hold():
    frames = gap(4) + seen(10) + [set()] + seen(30)
    assert [e.role for _, e in run(TagVoter(), frames)] == ["START"]


def test_a_card_counts_once_per_presentation_and_again_after_it_has_been_taken_away():
    one = seen(HOLD_FRAMES) + gap(40)                               # ~2.7 s shown, ~2.7 s away
    events = run(TagVoter(), gap(4) + one * 3)
    assert [e.role for _, e in events] == ["START"] * 3


def test_the_same_level_card_counts_every_time_it_is_held_up_and_another_one_counts_as_itself():
    frames = seen(HOLD_FRAMES, 1) + gap(10) + seen(HOLD_FRAMES, 1) + gap(10) + seen(HOLD_FRAMES, 3)
    assert [(e.role, e.value) for _, e in run(TagVoter(), frames)] == [("LEVEL", 1), ("LEVEL", 1), ("LEVEL", 3)]


def test_changing_the_card_in_the_hand_starts_the_hold_again():
    frames = seen(25, 1) + seen(HOLD_FRAMES + 10, 2)                # 1.7 s of card 1, then card 2 straight after
    events = run(TagVoter(), frames)
    assert [(e.role, e.value) for _, e in events] == [("LEVEL", 2)]
    assert events[0][0] - 25 * DT >= 2.0 * S                        # card 2's own two seconds, not card 1's


def test_two_level_cards_at_once_with_equal_votes_do_not_guess():
    assert run(TagVoter(), [{1, 2}] * 60) == []


def test_cards_are_ignored_outside_the_lobby_and_one_held_through_a_game_must_be_taken_away_first():
    voter = TagVoter()
    for k, phase in enumerate(("COUNTDOWN", "RALLY", "POINT_OVER")):
        assert run(voter, seen(50, 2), phase=phase, t0=k * 50 * DT) == []
    assert run(voter, seen(50, 2), phase="LOBBY", t0=150 * DT) == []        # still the same card: it must be taken away first
    again = run(voter, gap(10) + seen(HOLD_FRAMES, 2), phase="LOBBY", t0=200 * DT)
    assert [(e.role, e.value) for _, e in again] == [("LEVEL", 2)]


@pytest.mark.parametrize("card, role", [(0, "START"), (3, "LEVEL")])
def test_a_card_also_starts_another_game_after_a_finished_one(card, role):
    events = run(TagVoter(), gap(4) + seen(HOLD_FRAMES, card), phase="MATCH_OVER")
    assert [(e.role, e.value) for _, e in events] == [(role, card)]


def test_unallocated_ids_are_rejected():
    assert run(TagVoter(), seen(60, 7) + seen(60, 99)) == []


# --- what the screen is told while the card is held -------------------------------------------------------------------------------------------
def test_the_hold_says_which_card_and_how_far_the_two_seconds_have_come():
    voter = TagVoter()
    assert voter.hold(0) is None
    run(voter, gap(3))
    assert voter.hold(3 * DT) is None                               # nothing in view
    run(voter, seen(15, 2), t0=3 * DT)                              # one second of card 2
    card, progress = voter.hold(18 * DT)
    assert card == 2 and 0.40 < progress < 0.60
    run(voter, seen(7, 2), t0=18 * DT)                              # a little over one and a half seconds
    card, progress = voter.hold(25 * DT)
    assert card == 2 and 0.70 < progress < 0.85


def test_the_hold_shows_between_the_looks_so_the_bar_moves_smoothly():
    voter = TagVoter()
    run(voter, seen(10, 1))                                         # looks come ten or fifteen times a second; the screen draws at 60
    a, b = voter.hold(10 * DT)[1], voter.hold(10 * DT + DT // 2)[1]
    assert b > a


def test_the_hold_is_gone_once_the_card_has_counted_or_has_been_taken_away_or_nobody_looks_any_more():
    voter = TagVoter()
    run(voter, seen(HOLD_FRAMES, 3))                                # it has counted
    assert voter.hold(HOLD_FRAMES * DT) is None
    run(voter, gap(10), t0=HOLD_FRAMES * DT)                        # taken away
    assert voter.hold((HOLD_FRAMES + 10) * DT) is None
    run(voter, seen(15, 2), t0=(HOLD_FRAMES + 10) * DT)
    last = (HOLD_FRAMES + 25) * DT
    assert voter.hold(last) is not None
    assert voter.hold(last + S) is None                             # the looks stopped (the game began): no bar left on the screen


def test_a_card_that_is_not_in_the_game_shows_no_hold():
    voter = TagVoter()
    run(voter, seen(30, 7))
    assert voter.hold(30 * DT) is None


def make_frame(tag_id, tag_px, frame_w=960, frame_h=540, ambient=190):
    page = make_cards.make_card_page(tag_id)
    ys, xs = np.where(page < 128)
    tag = page[ys.min():ys.max() + 1, xs.min():xs.max() + 1]            # the black-bordered tag only
    quiet = max(tag_px // 8, 8)
    tag = cv2.resize(tag, (tag_px, tag_px), interpolation=cv2.INTER_AREA)
    frame = np.full((frame_h, frame_w), 255, dtype=np.uint8)
    y0, x0 = frame_h // 3, frame_w // 3
    frame[y0 - quiet:y0 + tag_px + quiet, x0 - quiet:x0 + tag_px + quiet] = 255
    frame[y0:y0 + tag_px, x0:x0 + tag_px] = tag
    frame = (frame.astype(np.float32) * ambient / 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("tag_id", [0, 1, 2, 3])
def test_detector_reads_all_four_printed_cards(tag_id):
    tags = TagDetector().detect(make_frame(tag_id, 160))
    assert [t.id for t in tags] == [tag_id]
    assert 0 <= tags[0].center[0] < 960 and len(tags[0].corners) == 4


def test_detector_finds_a_tag_that_is_only_63_px_wide_in_a_960_px_frame():
    # a 15 cm card at 1.8 m seen by a 65 degree camera at 960 px is about this big
    tags = TagDetector().detect(make_frame(2, 63))
    assert [t.id for t in tags] == [2]


def test_detector_reports_nothing_in_an_empty_frame():
    blank = np.full((540, 960, 3), 128, dtype=np.uint8)
    assert TagDetector().detect(blank) == []
