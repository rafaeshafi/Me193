"""The whole live chain in contact mode on fake hardware: a hand that moves into the ball hits it, and a flick of the hub that the IMU
feels around the contact is the ball's spin."""

import play
from pingpong import contact, live, profile
from pingpong.calibration import SwingCalibration
from pingpong.paddle import ReachBox
from pingpong.sources_fake import FakeEnv, FakeLandmarker
from pingpong.tilt import TiltCalibration

import pytest


def tilted_player(root):
    cal = profile.Calibration(swing=SwingCalibration((0.0, 1.0, 0.0), 280.0, 1100.0), box=ReachBox(-1.0, 1.0, -0.5, 0.5),
                              shoulder_w=0.2, tilt=TiltCalibration((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)))
    profile.save("flicker", cal, root=root)


def build(tmp_path, *settings):
    tilted_player(tmp_path)
    env = FakeEnv(hz=64.0)
    args = play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-record", "--no-store", "--no-audio",
                            "--no-publish", "--player", "flicker", *[a for s in settings for a in ("--set", s)]])
    holder = {}

    def hand(t_ns):
        """The hand goes to where the ball will be and waits there (nothing else: no swing, no key)."""
        game = holder["rig"].session.game
        if game.incoming is not None and game.incoming_leg is not None:
            leg = game.incoming_leg
            return contact.ball_u(game.judge.box, leg.position(leg.arrival_ns)[0]), 0.0
        return 0.0, 0.0

    env.make_landmarker = lambda model="lite": FakeLandmarker(hand, env.clock, full_body=True)
    holder["rig"] = rig = live.build_live(args, env, player_root=tmp_path)
    return env, rig


def run(env, rig, seconds, until=None):
    for _ in range(round(seconds * 60)):
        env.sleep(1 / 60)
        rig.vision.step()
        rig.pump()
        if until is not None and until():
            return True
    return False


def test_a_hand_that_goes_to_the_ball_hits_it_with_no_swing_through_the_whole_live_chain(tmp_path):
    env, rig = build(tmp_path)
    assert rig.session.game.hit_mode == "contact" and rig.session.game.on_swing is not None
    rig.session.on_start()
    assert run(env, rig, 12.0, until=lambda: rig.session.game.tracker.streak >= 3)
    assert rig.n_impacts == 0                                                  # the hub saw no swing at all: it was never needed
    rig.close()


def test_a_flick_of_the_hub_around_the_contact_is_the_balls_spin_through_the_whole_live_chain(tmp_path):
    env, rig = build(tmp_path, "shot.flick_warmup=1")
    game, seen = rig.session.game, []
    # quiet until the first hit has taught what is usual; after it the hub tips its front down at a rate that stays high
    env.scenario = lambda now: (0, 0, 1000, 20, 4500 if game.tracker.streak >= 1 else 20, 0)
    rig.session.on_start()
    original = game._launch

    def spy(**kw):
        events = original(**kw)
        seen.extend(e for e in events if e.kind == "hit")
        return events

    game._launch = spy
    assert run(env, rig, 14.0, until=lambda: len(seen) >= 3)
    assert seen[0].data["topspin"] == 0.0                                      # the first hit learns what is usual
    assert seen[1].data["topspin"] == pytest.approx(1.0, abs=0.01)             # then a tipped-down hub is topspin
    assert seen[1].data["mode"] == "contact"
    rig.close()


def test_a_recorded_contact_session_replays_to_the_same_hits_at_the_same_instants(tmp_path):
    from pingpong import recorder, replay

    tilted_player(tmp_path)
    env = FakeEnv(hz=64.0)
    args = play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-store", "--no-audio", "--no-publish",
                            "--player", "flicker", "--set", "shot.flick_warmup=1"])
    holder = {}

    def hand(t_ns):
        game = holder["rig"].session.game
        if game.incoming is not None and game.incoming_leg is not None:
            leg = game.incoming_leg
            return contact.ball_u(game.judge.box, leg.position(leg.arrival_ns)[0]), 0.0
        return 0.0, 0.0

    env.make_landmarker = lambda model="lite": FakeLandmarker(hand, env.clock, full_body=True)
    rig = live.build_live(args, env, player_root=tmp_path, record_root=tmp_path / "rec")
    holder["rig"] = rig
    game = rig.session.game
    env.scenario = lambda now: (0, 0, 1000, 20, 3000 if game.tracker.streak >= 1 else 20, 0)
    rig.session.on_start()
    assert run(env, rig, 14.0, until=lambda: game.tracker.streak >= 3)
    run(env, rig, 0.5)                                                           # a tail: a replay stops with the recorded data
    rig.close()
    loaded = recorder.load(next((tmp_path / "rec").iterdir()))
    assert loaded.meta["hit_mode"] == "contact"
    live_hits = [(e["t"], round(e["d"]["topspin"], 3)) for e in loaded.events if e["k"] == "hit"]
    again = replay.replay(loaded)
    replayed = [(t, round(d["topspin"], 3)) for k, t, d in again.events if k == "hit"]
    assert len(live_hits) >= 3 and len(replayed) >= len(live_hits)
    for (t_live, spin_live), (t_again, spin_again) in zip(live_hits, replayed):
        assert spin_again == spin_live                                           # the flick is read the same way
        assert abs(t_again - t_live) <= 25_000_000                               # a replay pumps on its own 60 Hz grid: within a frame

