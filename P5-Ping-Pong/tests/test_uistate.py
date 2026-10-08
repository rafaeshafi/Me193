"""The words and numbers of the results screen, made from a finished game's summary."""

from pingpong import uistate

MATCH = {"mode": "match", "level": "Club", "target": 7, "streak": 12, "record": 12, "player_points": 7, "cpu_points": 4, "winner": "player",
         "hits": 31, "misses": 4, "faults": 0, "max_kmh": 73.6, "duration_s": 161.4}
RALLY = {"mode": "survival", "level": "Rookie", "target": 7, "streak": 17, "record": 17, "player_points": 0, "cpu_points": 0, "winner": None,
         "hits": 17, "misses": 1, "faults": 0, "max_kmh": 0.0, "duration_s": 59.9}


def stat(results, label):
    return dict(results.stats)[label]


def test_a_won_match_says_you_win_and_a_lost_one_says_the_computer_did():
    won = uistate.results_from_summary(MATCH, new_record=False)
    assert won.won is True and won.title == "YOU WIN!" and not won.new_record
    lost = uistate.results_from_summary(dict(MATCH, winner="cpu", player_points=2, cpu_points=7), new_record=False)
    assert lost.won is False and lost.title == "THE CPU WINS"


def test_a_rally_is_game_over_unless_it_set_a_record():
    plain = uistate.results_from_summary(RALLY, new_record=False)
    assert plain.won is None and plain.title == "GAME OVER" and not plain.new_record
    record = uistate.results_from_summary(RALLY, new_record=True)
    assert record.title == "NEW RECORD!" and record.new_record


def test_a_match_record_is_still_a_win_or_a_loss_with_the_record_noted():
    r = uistate.results_from_summary(MATCH, new_record=True)
    assert r.title == "YOU WIN!" and r.won is True and r.new_record


def test_the_numbers_are_hits_the_longest_rally_the_top_speed_and_the_time():
    r = uistate.results_from_summary(MATCH, new_record=False)
    assert [label for label, _ in r.stats] == ["HITS", "LONGEST RALLY", "TOP SPEED", "TIME"]
    assert stat(r, "HITS") == "31" and stat(r, "LONGEST RALLY") == "12" and stat(r, "TOP SPEED") == "74 km/h" and stat(r, "TIME") == "2:41"


def test_a_game_with_no_speed_or_a_very_short_one_still_reads_sensibly():
    r = uistate.results_from_summary(RALLY, new_record=False)
    assert stat(r, "TOP SPEED") == "-" and stat(r, "TIME") == "0:59"
    assert stat(uistate.results_from_summary(dict(RALLY, duration_s=3.2), new_record=False), "TIME") == "0:03"
    assert stat(uistate.results_from_summary(dict(RALLY, duration_s=3725.0), new_record=False), "TIME") == "62:05"
