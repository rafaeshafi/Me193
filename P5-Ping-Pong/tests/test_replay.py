"""replay: a recorded session fed back through the real pipeline on a simulated clock.

Two uses: prove a recording reproduces the decisions made live, and ask counterfactuals ("what
if the late window were 0.4 s?") on real data without anybody swinging again.
"""

import pytest

from pingpong import fakerig, recorder, replay, sessionreport
from tools import replay as replay_tool

DECISIONS = ("serve", "hit", "miss", "fault", "rally_end", "game_over", "point")


def decisions(loaded):
    return [(e["k"], e["t"]) for e in loaded.events if e["k"] in DECISIONS]


def verdict_kinds(loaded):
    return [e["d"]["verdict"]["kind"] for e in loaded.events if e["k"] == "verdict"]


def record(tmp_path, name="orig", until=None, **kw):
    rig = fakerig.FakeRig(record_dir=tmp_path / name, **kw)
    rig.run(until=until or (lambda: rig.game.tracker.streak >= 8), max_s=120)
    rig.close()
    return recorder.load(tmp_path / name)


def test_a_replay_reproduces_every_decision_of_the_live_session_at_the_same_instants(tmp_path):
    original = record(tmp_path)
    result = replay.replay(original, record_dir=tmp_path / "again")
    again = recorder.load(tmp_path / "again")
    assert decisions(again) == decisions(original)
    assert verdict_kinds(again) == verdict_kinds(original)
    assert result.hits == 8 and result.record == 8


def test_a_replay_reproduces_pauses_and_the_rally_that_carried_on(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "orig", stale_ms=300.0)
    rig.run(until=lambda: rig.game.phase == "RALLY")
    now = rig.now_s()
    rig.hub_blackouts.append((now + 0.2, now + 1.4))
    rig.run(until=lambda: rig.game.tracker.streak >= 5, max_s=60)
    rig.close()
    original = recorder.load(tmp_path / "orig")
    replay.replay(original, record_dir=tmp_path / "again")
    again = recorder.load(tmp_path / "again")
    assert decisions(again) == decisions(original)
    pauses = lambda s: [(e["d"]["reasons"]) for e in s.events if e["k"] == "pause"]    # noqa: E731
    assert pauses(again) == pauses(original) and ["hub"] in pauses(original)


def test_a_wider_window_turns_a_late_miss_into_a_hit_on_the_same_recorded_swing(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "late", timing_s=0.25)               # Rookie accepts +180 ms
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=60)
    rig.close()
    late = recorder.load(tmp_path / "late")
    assert sessionreport.summarize(late)["hits"] == 0 and "REJECTED" in verdict_kinds(late)
    fixed = replay.replay(late, overrides={"level": {"late_s": 0.40}, "judge": {"d95_s": 0.20}})
    assert fixed.hits >= 1                                                           # same swing, wider window


def test_a_stricter_swing_threshold_turns_a_hit_into_a_non_event(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "orig")
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    rig.close()
    strict = replay.replay(recorder.load(tmp_path / "orig"), overrides={"swing": {"t_pk": 900.0}})
    assert strict.hits == 0                                                          # a 600 dps swing no longer counts


def test_a_replay_can_never_publish_anything(tmp_path):
    original = record(tmp_path)
    result = replay.replay(original)
    assert result.rig.mqtt_client is None and result.rig.session.game.publisher is None


def test_unknown_override_names_are_rejected_loudly(tmp_path):
    original = record(tmp_path)
    for bad in ({"level": {"nonsense": 1}}, {"judge": {"nonsense": 1}}, {"swing": {"nonsense": 1}}, {"wat": {}}):
        with pytest.raises(ValueError, match="nonsense|wat"):
            replay.replay(original, overrides=bad)


def test_set_arguments_become_typed_overrides():
    assert replay.parse_overrides(["level.late_s=0.4", "swing.t_pk=150", "judge.d95_s=0.15"]) == {
        "level": {"late_s": 0.4}, "swing": {"t_pk": 150.0}, "judge": {"d95_s": 0.15}}
    with pytest.raises(ValueError, match="section.name=value"):
        replay.parse_overrides(["level.late_s"])


def test_the_tool_prints_the_original_and_the_replayed_headlines(tmp_path, capsys):
    record(tmp_path, name="orig")
    assert replay_tool.main([str(tmp_path / "orig"), "--set", "level.late_s=0.3"]) == 0
    out = capsys.readouterr().out
    assert "original" in out and "replayed" in out and "hits 8" in out
    assert replay_tool.main([str(tmp_path / "orig"), "--set", "level.nonsense=1"]) == 2
    assert replay_tool.main([str(tmp_path / "missing")]) == 1


def test_the_tool_selftest_is_green():
    assert replay_tool.main(["--selftest"]) == 0


def test_a_session_played_with_spin_replays_exactly_because_the_recorded_probabilities_are_reused(tmp_path):
    top = lambda feat: {"flat": 0.0, "top": 1.0, "back": 0.0}                     # noqa: E731
    rig = fakerig.FakeRig(record_dir=tmp_path / "orig", spin_probs_fn=top)
    rig.run(until=lambda: rig.game.tracker.streak >= 6, max_s=90)
    rig.close()
    original = recorder.load(tmp_path / "orig")
    swings = [e for e in original.events if e["k"] == "swing"]
    assert swings and all(e["d"]["spin_probs"]["top"] == 1.0 for e in swings)
    result = replay.replay(original, record_dir=tmp_path / "again")
    assert decisions(recorder.load(tmp_path / "again")) == decisions(original)
    assert result.hits == 6
    assert any(e["d"]["topspin"] > 0.3 for e in original.events if e["k"] == "hit")      # spin really was in play


def test_a_session_against_the_learning_opponent_is_flagged_because_its_serves_cannot_be_reproduced(tmp_path):
    import random

    from pingpong import qbandit

    rig = fakerig.FakeRig(level=2, record_dir=tmp_path / "orig", learner=qbandit.QBandit(rng=random.Random(1)))
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    rig.close()
    original = recorder.load(tmp_path / "orig")
    assert original.meta["learn"] is True
    assert any("learning opponent" in w for w in replay.replay(original).warnings)
    plain = fakerig.FakeRig(record_dir=tmp_path / "plain")
    plain.run(until=lambda: plain.game.tracker.streak >= 3, max_s=60)
    plain.close()
    assert replay.replay(recorder.load(tmp_path / "plain")).warnings == []


def test_a_replay_never_starts_a_game_by_itself_from_a_hand_that_sits_on_the_start_button(tmp_path):
    # games start in a replay exactly where the recording says; the hold-to-start logic is for live play only
    loaded = record(tmp_path, until=lambda: True)
    assert replay.replay(loaded).rig.session.hold_start is None

