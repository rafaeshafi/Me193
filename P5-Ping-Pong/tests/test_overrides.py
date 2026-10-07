"""--set section.name=value: tune the swing detector, the judge or the levels from the command line.

The same settings work in `./pp play --set ...` (live tuning without editing code) and `./pp replay --set ...`
(the same recorded swings judged with other numbers); a live session records what it was run with, so its replay
starts from the same settings.
"""

import pytest

import play
from pingpong import fakerig, levels, live, overrides, recorder, replay, sessionreport
from pingpong.sources_fake import FakeEnv


def args_for(*extra):
    return play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-store", *extra])


def test_set_arguments_become_typed_checked_settings():
    parsed = overrides.parse(["level.late_s=0.4", "swing.t_pk=150", "judge.d95_s=0.15"])
    assert parsed == {"level": {"late_s": 0.4}, "swing": {"t_pk": 150}, "judge": {"d95_s": 0.15}}
    assert overrides.check(parsed) is parsed
    with pytest.raises(ValueError, match="section.name=value"):
        overrides.parse(["level.late_s"])
    with pytest.raises(ValueError, match="unknown section"):
        overrides.check({"ball": {"speed": 3}})
    with pytest.raises(ValueError, match="radius_sw"):                       # the message names the real settings
        overrides.check({"level": {"radius": 0.7}})


def test_later_settings_win_and_sections_merge():
    base = {"level": {"late_s": 0.3, "radius_sw": 0.6}, "swing": {"t_pk": 150.0}}
    merged = overrides.merge(base, {"level": {"late_s": 0.4}, "judge": {"d95_s": 0.2}})
    assert merged == {"level": {"late_s": 0.4, "radius_sw": 0.6}, "swing": {"t_pk": 150.0}, "judge": {"d95_s": 0.2}}
    assert base["level"]["late_s"] == 0.3                                      # the inputs are not touched


def test_a_level_setting_changes_every_level_and_survives_a_change_of_level():
    rig = fakerig.FakeRig(level=1)
    overrides.apply(rig.rig, {"level": {"radius_sw": 0.9}})
    game = rig.game
    assert game.level.radius_sw == 0.9 and game.level.name == "Rookie"
    game.set_level(levels.LEVELS[3])                                           # a tag or key changes the level
    assert game.level.name == "Pro" and game.level.radius_sw == 0.9
    assert levels.LEVELS[3].radius_sw != 0.9                                   # the table itself is never edited


def test_the_arrival_window_can_be_tuned_live_with_set_level_reach():
    rig = fakerig.FakeRig(level=1)
    overrides.apply(rig.rig, {"level": {"reach": 0.4}})
    assert rig.game.level.reach == 0.4 and levels.LEVELS[1].reach == 0.6
    plans = [rig.game.policy.serve(rig.game.level, 0.5, 0, 0.5, False) for _ in range(50)]
    assert all(abs(a - 0.5) <= 0.35 * 0.4 + 1e-9 for a, _ in (p.aim_ab for p in plans))


def test_judge_and_swing_settings_reach_the_judge_and_the_detector():
    rig = fakerig.FakeRig()
    overrides.apply(rig.rig, {"judge": {"refractory_s": 0.5}, "swing": {"t_pk": 222.0}})
    assert rig.game.judge.refractory_s == 0.5
    assert rig.rig.imu.detector.p.t_pk == 222.0 and rig.game.judge.t_pk == 222.0      # J3 follows the detector


def test_a_live_session_takes_the_settings_and_a_bad_one_is_a_clear_error(tmp_path):
    rig = live.build_live(args_for("--no-record", "--set", "level.late_s=0.4", "--set", "judge.d95_s=0.2"), FakeEnv(),
                          player_root=tmp_path)
    assert rig.session.game.level.late_s == 0.4 and rig.session.game.judge.d95_s == 0.2
    with pytest.raises(live.LiveSetupError, match="--set"):
        live.build_live(args_for("--no-record", "--set", "level.nonsense=1"), FakeEnv(), player_root=tmp_path)


def test_a_recorded_session_remembers_its_settings_and_the_report_says_so(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "s", overrides={"level": {"late_s": 0.4}})
    assert rig.game.level.late_s == 0.4
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    rig.close()
    loaded = recorder.load(tmp_path / "s")
    assert loaded.meta["overrides"] == {"level": {"late_s": 0.4}}
    assert "level.late_s=0.4" in sessionreport.format_report(sessionreport.summarize(loaded))


def test_a_replay_starts_from_the_recorded_settings_and_a_new_set_wins(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "s", overrides={"level": {"late_s": 0.4, "radius_sw": 0.7}})
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    rig.close()
    loaded = recorder.load(tmp_path / "s")
    same = replay.replay(loaded)
    assert same.rig.session.game.level.late_s == 0.4 and same.hits == 3
    changed = replay.replay(loaded, overrides={"level": {"late_s": 0.25}})
    level = changed.rig.session.game.level
    assert level.late_s == 0.25 and level.radius_sw == 0.7                      # merged, the new value wins
