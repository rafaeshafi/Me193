from pingpong import feedback, haptics, levels
from pingpong.events import GameEvent

ROOKIE, CLUB, PRO = (levels.LEVELS[i] for i in (1, 2, 3))


def ev(kind, **data):
    return GameEvent(kind=kind, t_ns=0, data=data)


def test_every_mapped_pattern_exists_in_the_haptics_table():
    for kind, data in [("hit", {"label": "perfect"}), ("hit", {"label": "good"}), ("hit", {"label": "early"}),
                       ("hit", {"label": "late"}), ("fault", {}), ("miss", {}), ("record", {}),
                       ("serve", {}), ("point", {"scorer": "player"}), ("point", {"scorer": "cpu"})]:
        name = feedback.pattern_for(ev(kind, **data), ROOKIE)
        assert name is None or name in haptics.PATTERNS, (kind, data, name)


def test_hit_labels_map_to_their_own_cues():
    assert feedback.pattern_for(ev("hit", label="perfect"), ROOKIE) == "hit_perfect"
    assert feedback.pattern_for(ev("hit", label="good"), ROOKIE) == "hit_good"
    assert feedback.pattern_for(ev("hit", label="early"), ROOKIE) == "hit_early"
    assert feedback.pattern_for(ev("hit", label="late"), ROOKIE) == "hit_late"


def test_early_and_late_motor_cues_degrade_when_the_flight_is_too_short_to_feel_them():
    # Pro flies in 0.43 s: no free time for two-tick cues, so they become beep + light only
    assert feedback.pattern_for(ev("hit", label="early"), PRO) == "hit_good"
    assert feedback.pattern_for(ev("hit", label="late"), PRO) == "hit_good"
    assert feedback.pattern_for(ev("hit", label="early"), CLUB) == "hit_early"      # 0.60 s: fine
    assert feedback.pattern_for(ev("hit", label="perfect"), PRO) == "hit_perfect"   # the thump always plays


def test_misses_and_faults_buzz_and_records_celebrate():
    assert feedback.pattern_for(ev("miss"), CLUB) == "fault"
    assert feedback.pattern_for(ev("fault", fault="out"), CLUB) == "fault"
    assert feedback.pattern_for(ev("record", value=7), CLUB) == "record"


def test_points_are_cued_by_who_scored():
    assert feedback.pattern_for(ev("point", scorer="player"), CLUB) == "point_won"
    assert feedback.pattern_for(ev("point", scorer="cpu"), CLUB) == "fault"


def test_boring_events_make_no_haptic():
    assert feedback.pattern_for(ev("verdict"), CLUB) is None
    assert feedback.pattern_for(ev("rally_end"), CLUB) is None
    assert feedback.pattern_for(ev("unknown_kind"), CLUB) is None


def test_play_submits_in_event_order_and_survives_a_dropped_pattern():
    class Recorder:
        def __init__(self):
            self.names = []

        def submit(self, name, fire_at_ns=None):
            self.names.append(name)
            return name != "hit_good"            # simulate one pattern being dropped

    rec = Recorder()
    feedback.play([ev("hit", label="good"), ev("record", value=1), ev("verdict")], CLUB, rec)
    assert rec.names == ["hit_good", "record"]
