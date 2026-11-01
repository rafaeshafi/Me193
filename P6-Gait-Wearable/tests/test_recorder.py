import csv

import pytest

from recorder import GAP_S, HEADER, SCRIPT, Recorder


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def read(path):
    with open(path) as f:
        return list(csv.reader(f))


SAMPLE = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_guided_script_prompts_and_finishes_on_the_last_batch(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    assert not r.active and r.prompt() == ""
    r.start_guided(who="rafae")
    assert r.active and r.path.name.endswith("-guided-rafae.csv")
    assert r.prompt() == "GET READY %ds" % GAP_S
    clk.t += GAP_S
    assert r.prompt() == "STAND 20s"
    clk.t += 19.5
    assert r.prompt() == "STAND 1s"
    clk.t += 0.5 + GAP_S                      # into WALK
    assert r.prompt().startswith("WALK")
    total = sum(GAP_S + s for _, s in SCRIPT)
    clk.t = 1000.0 + total + 0.1
    assert r.prompt() == "" and r.active       # script over, but the file waits for the last batch
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)
    assert not r.active


def test_samples_get_the_label_of_their_own_time(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    r.start_single("WALK", seconds=10)
    # a 10-sample batch ending 45 ms after the gap ends: samples at -45..-5 ms are in the gap, 5..45 ms are WALK
    clk.t += GAP_S + 0.045
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)
    rows = read(r.path)
    assert rows[0] == HEADER
    labels = [row[1] for row in rows[1:]]
    assert labels == [""] * 5 + ["WALK"] * 5
    assert float(rows[-1][0]) > float(rows[1][0])
    assert r.rows == 10


def test_status_ahead_of_the_samples_does_not_move_the_labels(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    r.start_single("WALK", seconds=10)
    clk.t += GAP_S + 0.05                          # the clock is 50 ms into WALK
    assert r.status()["label"] == "WALK"           # the phone page asks first...
    r.on_samples(clk.t - 0.1, [SAMPLE] * 10, rate=100)   # ...then the batch that ended inside the gap arrives
    assert [row[1] for row in read(r.path)[1:]] == [""] * 10
    clk.t = 1000.0 + GAP_S + 10 + 1.0              # a second past the end
    assert r.prompt() == "" and r.status()["active"] and r.active
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)   # nothing to write, closes the file
    assert not r.active and r.rows == 10 and r.sessions() == [r.path.name]


def test_single_rejects_unknown_label_and_stop_closes(tmp_path):
    r = Recorder(tmp_path, clock=Clock(), rolling=False)
    try:
        r.start_single("FLY")
        assert False, "expected ValueError"
    except ValueError:
        pass
    r.start_single("RUN")
    assert r.active and r.sessions() == []           # the file being written is not offered
    r.stop()
    assert not r.active and r.sessions() == [r.path.name]


def test_rolling_log_runs_only_while_idle(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=True)
    r.on_samples(clk.t, [SAMPLE] * 3, rate=100)
    files = list(tmp_path.glob("rolling-*.csv"))
    assert len(files) == 1
    r.start_single("STAND")
    clk.t += 1.0
    r.on_samples(clk.t, [SAMPLE] * 3, rate=100)      # goes to the session, not the rolling log
    r.stop()
    r.close()
    rows = read(files[0])
    assert len(rows) == 4 and rows[1][1] == "-"
    assert r.sessions() == [r.path.name]             # rolling logs are not sessions


from recorder import STEP_ACTIVITIES, parse_counted


def test_parse_counted():
    assert parse_counted("20261101-101500-steps-walk-42-Rafae.csv") == ("WALK", 42, "Rafae")
    assert parse_counted("20261101-101500-steps-bike-0.csv") == ("BIKE", 0, "")
    assert parse_counted("20261101-101500-steps-walk-42-Rafae-2.csv") == ("WALK", 42, "Rafae")
    assert parse_counted("20261101-101500-steps-walk-Rafae.csv") is None      # stopped without a count
    assert parse_counted("20261026-135026-guided-Rafae.csv") is None
    assert parse_counted("20261101-101500-steps-hop-5.csv") is None            # not an activity
    assert STEP_ACTIVITIES == ("WALK", "RUN", "STAND", "BIKE", "MIXED")


def test_counted_take_is_renamed_with_its_count(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    r.start_counted("walk", who="Rafae")
    assert r.active and r.path.name.endswith("-steps-walk-Rafae.csv")
    st = r.status()
    assert st["kind"] == "counted" and st["elapsed_s"] == 0.0 and st["label"] == "WALK" and st["prompt"].startswith("WALK")
    clk.t += 12.3
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)
    assert r.status()["elapsed_s"] == 12.3
    rows = read(r.path)
    assert [row[1] for row in rows[1:]] == ["WALK"] * 10          # no get-ready gap: every row carries the activity
    r.stop(steps=42)
    assert not r.active
    assert r.path.name.endswith("-steps-walk-42-Rafae.csv") and r.path.exists()
    assert r.sessions() == [r.path.name]
    assert r.status()["kind"] is None and r.status()["elapsed_s"] is None


def test_counted_take_without_a_count_keeps_its_name(tmp_path):
    r = Recorder(tmp_path, clock=Clock(), rolling=False)
    r.start_counted("MIXED")
    name = r.path.name
    r.stop()
    assert r.path.name == name and name.endswith("-steps-mixed.csv") and parse_counted(name) is None


def test_steps_are_ignored_for_other_kinds_and_bad_activities_rejected(tmp_path):
    r = Recorder(tmp_path, clock=Clock(), rolling=False)
    r.start_single("WALK", seconds=10)
    name = r.path.name
    r.stop(steps=7)
    assert r.path.name == name
    with pytest.raises(ValueError):
        r.start_counted("HOP")
