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


def test_the_hud_gets_the_imu_trace_of_the_swing_that_was_just_judged():
    rig = fakerig.FakeRig()
    rig.run(until=lambda: rig.game.tracker.streak >= 2, max_s=60)
    state = rig.rig.hud_state()
    assert max(state.swing_trace) > 0.8 * 600.0
    assert state.swing_threshold == pytest.approx(0.6 * 300.0)


def test_shaking_the_hub_locks_the_paddle_so_the_ball_arriving_meanwhile_is_not_a_hit():
    # The shake is on the hub's y axis, which the swing detector does not look at: only the FFT
    # monitor sees it.  A perfectly good swing at the right moment is rejected by gate J6.
    rig = fakerig.FakeRig()
    rig.run(until=lambda: rig.game.phase == "COUNTDOWN")
    rig.shake_windows.append((rig.now_s(), rig.now_s() + 8.0))
    rig.run(until=lambda: rig.game.phase == "MATCH_OVER", max_s=30)
    assert rig.game.tracker.record == 0 and payloads(rig) == []
    j6 = [g for g in rig.session.hud_state().gates if g.name == "J6"]
    assert j6 and not j6[0].passed and "shak" in j6[0].note


def test_after_the_shaking_stops_the_lock_expires_and_hits_count_again():
    rig = fakerig.FakeRig()
    rig.run(until=lambda: rig.game.phase == "COUNTDOWN")
    rig.shake_windows.append((rig.now_s(), rig.now_s() + 1.5))          # over before the first ball arrives
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    assert rig.game.tracker.streak >= 3


def test_a_recorded_fake_session_holds_the_streams_a_replay_needs(tmp_path):
    from pingpong import recorder

    rig = fakerig.FakeRig(seed=5, record_dir=tmp_path / "run")
    rig.run(until=lambda: rig.game.tracker.streak >= 3, max_s=60)
    rig.close()
    loaded = recorder.load(tmp_path / "run")
    assert loaded.meta["seed"] == 5 and loaded.meta["units"]["gyro_per_dps"] == fakerig.GPD
    assert loaded.meta["t0_ns"] == rig.origin_ns and loaded.meta["calibration"]["swing"]["omega_hi"] == 1200.0
    assert len(loaded.imu) > 300 and len(loaded.poses) > 100
    kinds = [e["k"] for e in loaded.events]
    assert kinds.count("serve") >= 3 and kinds.count("hit") == 3 and kinds.count("swing") >= 3
    assert "tag" in kinds and "phase" in kinds
    verdicts = [e for e in loaded.events if e["k"] == "verdict"]
    assert [g["name"] for g in verdicts[0]["d"]["verdict"]["gates"]] == ["J1", "J2", "J3", "J4", "J5", "J6"]


def test_the_whole_pipeline_plays_with_a_learning_opponent_and_the_learner_learns():
    import random

    from pingpong import qbandit

    learner = qbandit.QBandit(rng=random.Random(1))
    rig = fakerig.FakeRig(level=2, learner=learner)
    rig.run(until=lambda: rig.game.tracker.streak >= 8, max_s=120)
    assert rig.game.tracker.streak >= 8 and float(abs(learner.table).sum()) > 0.0


@pytest.mark.parametrize("level", (1, 2, 3))
def test_a_hand_that_really_swings_through_the_ball_is_still_judged_a_hit_for_an_ordinary_swing(level):
    # The scripted hand used to sit perfectly still while the gyro swung.  A real hand covers a shoulder width or
    # more in the 300 ms the pose gate looks at, so the gate has to cope with a moving hand.
    rig = fakerig.FakeRig(level=level, hand_motion=True, w_pk=600.0)
    rig.run(until=lambda: rig.game.tracker.streak >= 6, max_s=90)
    assert rig.game.tracker.streak >= 6


def test_the_hardest_swing_carries_the_hand_out_of_the_pro_radius_and_a_wider_radius_fixes_it():
    # A known limit, not a bug: at ~11 shoulder widths a second the hand leaves Pro's 0.35 radius before the peak.
    # The tuning knob is --set level.radius_sw, and it is enough.
    hard = fakerig.FakeRig(level=3, hand_motion=True, w_pk=1100.0)
    hard.run(until=lambda: hard.game.phase == "MATCH_OVER", max_s=60)
    assert hard.game.tracker.record == 0
    wide = fakerig.FakeRig(level=3, hand_motion=True, w_pk=1100.0, overrides={"level": {"radius_sw": 0.9}})
    wide.run(until=lambda: wide.game.tracker.streak >= 5, max_s=90)
    assert wide.game.tracker.streak >= 5
