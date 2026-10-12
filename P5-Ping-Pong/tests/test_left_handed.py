"""A left-handed player through the real code: the calibration tool saves the left hand and a reach box on the player's left, the camera then
follows the left hand, and everything that reads the hand (START, a rally, the swing chain) works from that box."""

import dataclasses

import pytest

import config
import play
from pingpong import contact, fakerig, live, profile
from pingpong.paddle import ReachBox
from pingpong.sources_fake import FakeEnv, FakeLandmarker
from test_build_live import left_handed_player
from test_calibrate_swing import LEFT_CORNERS, fake_env
from tools import calibrate_swing as tool

LEFT_BOX = ReachBox(-1.8, -0.2, -0.9, 0.7)                  # what left_handed_player() saves: out to the player's left, not across the body


def cli(*extra):
    return play.parse_args(["--card-color", "red", "--card-serial", "1131", "--no-record", "--no-store", "--no-audio",
                            "--no-publish", "--player", "arbert", *extra])


def run(env, rig, seconds, until=None):
    for _ in range(round(seconds * 60)):
        env.sleep(1 / 60)
        rig.vision.step()
        rig.pump()
        if until is not None and until():
            return True
    return False


def left_hand_that_goes_to_the_ball(holder):
    """The hand goes to where the ball will be and waits there; between balls it rests in the middle of the reach box."""
    def hand(t_ns):
        game = holder["rig"].session.game
        if game.incoming is not None and game.incoming_leg is not None:
            leg = game.incoming_leg
            return contact.ball_u(game.judge.box, leg.position(leg.arrival_ns)[0]), 0.0
        return LEFT_BOX.to_uv(0.5, 0.5)
    return hand


def build(tmp_path, *extra):
    left_handed_player(tmp_path)
    env, holder = FakeEnv(hz=64.0), {}
    env.make_landmarker = lambda model="lite": FakeLandmarker(left_hand_that_goes_to_the_ball(holder), env.clock,
                                                              full_body=True, hand="left")
    holder["rig"] = live.build_live(cli(*extra), env, player_root=tmp_path)
    return env, holder["rig"]


def test_a_left_handed_player_returns_balls_with_the_left_hand_through_the_whole_live_chain(tmp_path):
    env, rig = build(tmp_path)
    assert rig.vision.hand == "left" and rig.session.game.hit_mode == "contact"
    rig.session.on_start()
    assert run(env, rig, 14.0, until=lambda: rig.session.game.tracker.streak >= 3)
    rig.close()


def test_following_the_other_hand_finds_nobody_to_hit_the_ball(tmp_path):
    env, rig = build(tmp_path, "--hand", "right")                  # the same left-handed player, the camera on the right hand
    rig.session.on_start()
    assert not run(env, rig, 8.0, until=lambda: rig.session.game.tracker.streak >= 1)
    rig.close()


def test_a_left_handed_player_starts_a_game_by_holding_the_hub_at_the_top_right_of_their_own_reach(tmp_path):
    left_handed_player(tmp_path)
    env = FakeEnv(hz=64.0)
    t0 = env.clock.now_ns()
    reach = lambda t_ns: LEFT_BOX.to_uv(0.97, 0.95) if t_ns - t0 > 1e9 else LEFT_BOX.to_uv(0.5, 0.5)    # noqa: E731
    env.make_landmarker = lambda model="lite": FakeLandmarker(reach, env.clock, full_body=True, hand="left")
    rig = live.build_live(cli("--classic"), env, player_root=tmp_path)      # (a hand that starts on the button is not armed)
    assert run(env, rig, 6.0, until=lambda: rig.session.game.phase == "COUNTDOWN")
    rig.close()


def test_a_left_handed_scripted_player_returns_ten_balls_through_the_swing_chain_to_the_published_score():
    cal = dataclasses.replace(profile.Calibration.default(), hand="left", box=LEFT_BOX)
    rig = fakerig.FakeRig(level=1, calibration=cal)
    rig.run(until=lambda: rig.game.tracker.streak >= 10, max_s=120)
    published = [p["payload"] for p in rig.client.published if p["topic"] == config.SCORE_TOPIC]
    assert published == [f"{n}.0" for n in range(1, 11)] and rig.rig.n_impacts == rig.player.n_swings == 10


def test_what_the_calibration_tool_saves_for_a_left_handed_player_is_what_the_game_then_follows(tmp_path):
    env = fake_env(fakerig.CalibrationScript(hand="left", corners=LEFT_CORNERS), hand="left")
    args = tool.parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "arbert", "--hand", "left"])
    code = tool.run(env, args, profile_root=tmp_path, show=lambda f: None, wait_key=lambda ms: 255,
                    notify=lambda note: None, size=(320, 180), frame_hz=5.0)
    assert code == 0
    saved = profile.load("arbert", root=tmp_path)
    rig = live.build_live(cli(), FakeEnv(), player_root=tmp_path)             # no --hand: the profile says which
    assert rig.vision.hand == "left" and rig.session.game.judge.box == saved.box and saved.box.u_max < 0.0
    rig.close()
