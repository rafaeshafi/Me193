"""The whole live pipeline on fake hardware: a scripted player plays through the real code.

Fake hub notifications -> the real legoeducation parser -> HubLink -> ImuWorker -> SwingDetector,
fake camera frames -> VisionWorker (landmarks placed where the player's hand is) -> TagVoter,
haptics -> ActuatorCore -> fake hub, score -> fake broker.  Only the hardware and the clock are fake.
"""

import pytest

import config
from pingpong import fakerig, pose
from pingpong.sources_fake import body_landmarks

OFFICIAL = config.SCORE_TOPIC


def payloads(rig, topic=OFFICIAL):
    return [p["payload"] for p in rig.client.published if p["topic"] == topic]


def to_rally(rig):
    rig.run(until=lambda: rig.game.phase == "RALLY")
    return rig.now_s()


def test_body_landmarks_decode_to_exactly_the_requested_paddle_point():
    for hand in ("right", "left"):
        for u, v in ((0.0, 0.0), (0.8, -0.3), (-1.2, 0.5)):
            lm = body_landmarks(u, v, hand=hand)
            got = pose.paddle_uv(lm, hand, 640, 360)
            assert got[0] == pytest.approx(u, abs=1e-9) and got[1] == pytest.approx(v, abs=1e-9)


def test_a_scripted_player_returns_ten_balls_from_start_card_to_the_published_score():
    rig = fakerig.FakeRig(level=1)
    rig.run(until=lambda: rig.game.tracker.streak >= 10, max_s=120)
    assert rig.game.tracker.streak == 10 and rig.game.phase in ("RALLY", "POINT_OVER")
    assert payloads(rig) == [f"{n}.0" for n in range(1, 11)]
    assert rig.rig.n_impacts == rig.player.n_swings == 10          # one detected swing per ball, no phantoms
    assert rig.game.paused is False


def test_the_start_card_is_what_starts_the_game_nothing_else_does():
    rig = fakerig.FakeRig(cards=[])
    rig.run(seconds=6.0)
    assert rig.game.phase == "LOBBY" and payloads(rig) == []


def test_a_level_card_shown_before_the_start_card_sets_the_ball_speed_tier():
    rig = fakerig.FakeRig(level=1, cards=[(0.2, 1.0, 2), (1.2, 2.8, 0)])
    rig.run(until=lambda: rig.game.phase == "RALLY")
    assert rig.game.level.name == "Club"


def test_a_fake_source_never_reaches_the_official_topic():
    rig = fakerig.FakeRig(source="fake")
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    assert payloads(rig) == [] and payloads(rig, config.DEMO_SCORE_TOPIC) == ["1.0", "2.0", "3.0"]


def test_every_counted_hit_gets_a_cue_on_the_hub_and_the_motors_use_batched_writes():
    rig = fakerig.FakeRig()
    rig.run(until=lambda: rig.game.tracker.streak >= 5, max_s=60)
    calls = [c[0] for c in rig.dev.calls]
    assert calls.count("beep") >= 5 and calls.count("light_color") >= 5
    assert calls.count("begin_batch") == calls.count("end_batch")           # never a left-open batch


def test_haptic_vibration_does_not_fake_a_swing_because_the_blank_windows_are_wired():
    rig = fakerig.FakeRig(vibration=True)
    rig.run(until=lambda: rig.game.tracker.streak >= 6, max_s=90)
    assert rig.dev.n_motor_pulses > 0                                        # the hub really shook ...
    assert rig.rig.n_impacts == rig.player.n_swings                          # ... and nothing counted for it


def test_a_motor_pulse_blanks_the_swing_detector_through_the_assembled_wiring():
    # Direct check of the wiring, with the control: the same pulse WITHOUT the blank window makes
    # the detector fire, so the simulated vibration is strong enough to matter.
    results = {}
    for wired in (True, False):
        rig = fakerig.FakeRig(vibration=True, cards=[])
        if not wired:
            rig.session.actuator.on_blank = None
        rig.run(seconds=1.0)                                     # let the detector finish its warm-up
        rig.session.actuator.submit("hit_perfect")               # a 60 ms thump: the fake IMU shakes
        rig.run(seconds=0.6)
        results[wired] = rig.rig.n_impacts
    assert results == {True: 0, False: 1}


def test_a_hub_blackout_during_a_ball_pauses_the_game_and_the_rally_carries_on():
    rig = fakerig.FakeRig(stale_ms=300.0)
    now = to_rally(rig)
    rig.hub_blackouts.append((now + 0.2, now + 1.6))
    rig.run(until=lambda: rig.game.tracker.streak >= 5, max_s=90)
    assert "hub" in rig.pause_reasons_seen
    assert rig.game.tracker.streak >= 5 and payloads(rig)[:5] == [f"{n}.0" for n in range(1, 6)]


def test_a_pose_blackout_during_a_ball_pauses_the_game_and_the_rally_carries_on():
    rig = fakerig.FakeRig()
    now = to_rally(rig)
    rig.pose_blackouts.append((now + 0.2, now + 1.8))
    rig.run(until=lambda: rig.game.tracker.streak >= 5, max_s=90)
    assert "pose" in rig.pause_reasons_seen
    assert rig.game.tracker.streak >= 5


def test_a_late_swing_inside_the_window_counts_but_a_very_late_one_is_a_miss_and_ends_survival():
    late = fakerig.FakeRig(timing_s=0.10)                       # Rookie accepts up to +180 ms
    late.run(until=lambda: late.game.tracker.streak >= 3, max_s=60)
    assert late.game.tracker.streak >= 3
    missed = fakerig.FakeRig(timing_s=0.60)
    missed.run(until=lambda: missed.game.phase == "MATCH_OVER", max_s=60)
    assert missed.game.phase == "MATCH_OVER" and missed.game.tracker.record == 0


def test_a_swing_weaker_than_the_threshold_is_never_a_hit():
    rig = fakerig.FakeRig(w_pk=90.0)                            # T_PK is 0.6 * omega_lo = 180 dps
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=60)
    assert rig.game.tracker.record == 0 and payloads(rig) == []


def test_a_match_on_the_real_pipeline_plays_to_the_target_and_the_player_wins_it():
    rig = fakerig.FakeRig(level=1, mode="match", target=3, seed=3)
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=600)
    assert rig.game.phase == "MATCH_OVER" and rig.game.player_points == 3 and rig.game.cpu_points == 0


def test_closing_the_rig_stops_the_motors_and_says_offline():
    rig = fakerig.FakeRig()
    rig.run(until=lambda: rig.game.tracker.streak >= 2, max_s=60)
    rig.close()
    names = [c[0] for c in rig.dev.calls]
    assert names[-2:] == ["motor_stop", "disconnect"]
    assert ("publish", config.STATUS_TOPIC, "offline") in rig.client.log
    assert rig.client.log[-1] == ("loop_stop",)
