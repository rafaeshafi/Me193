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


def test_start_needs_a_vote_and_a_continuous_hold_of_four_tenths_of_a_second():
    events = run(TagVoter(), gap(5) + seen(12))
    assert [e.role for _, e in events] == ["START"]
    t_fire = events[0][0]
    first_present = 5 * DT
    assert t_fire - first_present >= 400 * MS


def test_a_short_flash_never_triggers_start():
    assert run(TagVoter(), gap(5) + seen(3) + gap(10)) == []        # ~0.2 s
    assert run(TagVoter(), gap(5) + seen(5) + gap(10)) == []        # ~0.33 s: still under the hold


def test_one_dropped_frame_does_not_break_a_hold():
    frames = gap(4) + seen(4) + [set()] + seen(8)
    assert [e.role for _, e in run(TagVoter(), frames)] == ["START"]


def test_start_fires_once_per_presentation_and_rearms_after_removal_and_lockout():
    one = seen(12) + gap(40)                                        # ~0.8 s shown, ~2.7 s away
    events = run(TagVoter(), gap(4) + one * 3)
    assert [e.role for _, e in events] == ["START"] * 3


def test_start_is_ignored_outside_the_lobby_and_a_card_held_through_a_rally_does_not_fire_after():
    voter = TagVoter()
    assert run(voter, seen(30), phase="RALLY") == []
    assert run(voter, seen(30), phase="LOBBY", t0=40 * DT) == []        # still held: must be removed first
    assert [e.role for _, e in run(voter, gap(10) + seen(12), phase="LOBBY", t0=80 * DT)] == ["START"]


def test_start_also_restarts_a_finished_game():
    assert [e.role for _, e in run(TagVoter(), gap(4) + seen(12), phase="MATCH_OVER")] == ["START"]


@pytest.mark.parametrize("tag, level", [(1, 1), (2, 2), (3, 3)])
def test_level_cards_latch_once_voted_in_the_lobby(tag, level):
    events = run(TagVoter(), gap(3) + seen(10, tag))
    assert [(e.role, e.value) for _, e in events] == [("LEVEL", level)]


def test_the_same_level_card_does_not_re_announce_but_a_new_one_does():
    frames = seen(8, 1) + gap(8) + seen(8, 1) + gap(8) + seen(8, 3)
    assert [(e.role, e.value) for _, e in run(TagVoter(), frames)] == [("LEVEL", 1), ("LEVEL", 3)]


def test_level_cards_do_nothing_mid_rally_or_countdown():
    voter = TagVoter()
    assert run(voter, seen(20, 2), phase="RALLY") == []
    assert run(voter, seen(20, 2), phase="COUNTDOWN", t0=30 * DT) == []


def test_two_level_cards_at_once_with_equal_votes_do_not_guess():
    both = [{1, 2}] * 12
    assert run(TagVoter(), both) == []


def test_unallocated_ids_are_rejected():
    assert run(TagVoter(), seen(30, 7) + seen(30, 99)) == []


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
