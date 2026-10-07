"""sessionreport: a recorded session -> the numbers that tune the game (and the explain-back)."""

import pytest

from pingpong import fakerig, recorder, sessionreport
from tools import report as report_tool


def record(tmp_path, name="run", **kw):
    rig = fakerig.FakeRig(record_dir=tmp_path / name, **kw)
    rig.run(until=lambda: rig.game.tracker.streak >= 5, max_s=120)
    return rig


def summarise(tmp_path, name="run"):
    return sessionreport.summarize(recorder.load(tmp_path / name))


def test_a_clean_session_reports_hits_timing_speed_and_hub_health(tmp_path):
    rig = record(tmp_path)
    rig.close()
    s = summarise(tmp_path)
    assert s["hits"] == 5 and s["balls"] >= 5 and s["swings"] >= 5 and s["best_streak"] == 5
    assert s["level"] == "Rookie" and s["mode"] == "survival" and s["misses"] == 0
    assert s["timing_ms"]["n"] == 5 and abs(s["timing_ms"]["mean"]) < 60.0          # swings land near the ball
    assert 55 < s["hub"]["hz"] < 70 and s["hub"]["worst_gap_ms"] < 40
    assert s["pose"]["n"] > 100 and s["w_pk"]["median"] == pytest.approx(600.0, rel=0.05)
    assert s["gate_failures"] == {} and s["pauses"] == []
    assert s["kmh"]["max"] > 0 and sum(s["labels"].values()) == 5


def test_a_missed_ball_ends_survival_and_is_counted(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "run", timing_s=0.6)
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=60)
    rig.close()
    s = summarise(tmp_path)
    assert s["hits"] == 0 and s["misses"] == 1 and s["game_over"] is True


def test_rejected_swings_are_broken_down_by_the_gate_that_failed(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "run")
    rig.run(until=lambda: rig.game.phase == "RALLY")
    now = rig.now_s()
    rig.pose_blackouts.append((now + 0.45, now + 1.0))                  # hand unseen while the ball arrives
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=60)
    rig.close()
    s = summarise(tmp_path)
    assert s["rejected"] >= 1 and s["gate_failures"].get("J2", 0) >= 1
    assert any("pose frame" in note for note in s["gate_notes"]["J2"])


def test_pauses_are_listed_with_their_reason_and_how_long_they_lasted(tmp_path):
    rig = fakerig.FakeRig(record_dir=tmp_path / "run", stale_ms=300.0)
    rig.run(until=lambda: rig.game.phase == "RALLY")
    now = rig.now_s()
    rig.hub_blackouts.append((now + 0.2, now + 1.4))
    rig.run(until=lambda: rig.game.tracker.streak >= 2, max_s=60)
    rig.close()
    pauses = summarise(tmp_path)["pauses"]
    assert pauses and pauses[0]["reasons"] == ["hub"] and 0.5 < pauses[0]["seconds"] < 1.3


def test_the_rig_adds_a_summary_event_with_loop_timing_when_it_closes(tmp_path):
    rig = record(tmp_path)
    rig.close()
    summary = [e for e in recorder.load(tmp_path / "run").events if e["k"] == "summary"]
    assert summary and summary[0]["d"]["loop"]["n"] > 100 and summary[0]["d"]["impacts"] >= 5
    assert summarise(tmp_path)["loop_p95_ms"] >= 0.0


def test_the_text_report_names_the_numbers_a_person_looks_for(tmp_path):
    rig = record(tmp_path)
    rig.close()
    text = sessionreport.format_report(summarise(tmp_path))
    for needle in ("hits 5", "best streak 5", "Hz", "timing", "J1", "Rookie"):
        assert needle in text, needle


def test_the_report_tool_picks_the_newest_session_in_a_folder_or_a_given_one(tmp_path, capsys):
    rig = record(tmp_path, name="20260101-000000-a")
    rig.close()
    rig2 = fakerig.FakeRig(record_dir=tmp_path / "20260102-000000-b", level=2)
    rig2.run(until=lambda: rig2.game.tracker.streak >= 2, max_s=60)
    rig2.close()
    assert report_tool.main(["--root", str(tmp_path)]) == 0
    assert "Club" in capsys.readouterr().out                            # the newest one
    assert report_tool.main([str(tmp_path / "20260101-000000-a")]) == 0
    assert "Rookie" in capsys.readouterr().out
    assert report_tool.main(["--root", str(tmp_path / "nothing")]) == 1


def test_the_tool_selftest_is_green():
    assert report_tool.main(["--selftest"]) == 0
