import play


def test_selftest_runs_a_scripted_ten_hit_rally_and_checks_the_published_scores():
    assert play.main(["--selftest"]) == 0


def test_argument_defaults():
    args = play.parse_args([])
    assert args.level == 1 and args.mode == "survival" and args.fake is False and args.no_publish is False


def test_the_window_loop_pumps_shows_every_frame_and_quits_on_q():
    from pingpong import fakerig

    rig = fakerig.FakeRig()
    frames, keys_pressed = [], []

    def wait_key(ms):
        return ord("q") if len(frames) >= 3 else 255

    play.run_loop(rig.rig, show=frames.append, wait_key=wait_key)
    assert len(frames) == 3 and frames[0].shape == (play.H, play.W, 3)
    assert rig.rig.loop_stats()["n"] == 3


def test_the_r_key_asks_a_lost_hub_to_reconnect_instead_of_reaching_the_game_keys():
    from pingpong import fakerig

    rig = fakerig.FakeRig()
    asked = []
    rig.rig.reconnect_now = lambda: asked.append(True)
    presses = iter([ord("r"), ord("q")])
    play.run_loop(rig.rig, show=lambda frame: None, wait_key=lambda ms: next(presses))
    assert asked == [True]


def test_live_mode_without_a_hub_card_explains_itself_instead_of_crashing(capsys):
    code = play.main(["--no-window"])
    err = capsys.readouterr().err
    assert code == 2 and "card" in err.lower() and "scan_hubs" in err


def test_the_selftest_also_runs_the_whole_live_pipeline_on_fake_hardware(capsys):
    assert play.main(["--selftest"]) == 0
    assert "live pipeline selftest OK" in capsys.readouterr().out


def test_live_arguments():
    args = play.parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "Rafae"])
    assert (args.card_color, args.card_serial, args.player) == ("red", "1131", "Rafae")
    assert play.parse_args([]).player == "rafae"


def test_rally_is_what_the_screen_and_the_command_line_call_the_survival_mode():
    assert play.parse_args(["--mode", "rally"]).mode == "survival"
    assert play.parse_args([]).mode == "survival" and play.parse_args(["--mode", "match"]).mode == "match"


def test_board_prints_the_leaderboard_without_touching_any_hardware(tmp_path, capsys):
    from pingpong import store

    db = store.Store(tmp_path / "pp.db")
    db.record_game("rafae", dict(mode="survival", level="Club", target=7, streak=12, record=12, hits=12, misses=1,
                                 faults=0, max_kmh=30.0, duration_s=40.0, source="live"))
    db.close()
    assert play.main(["--board", "--db", str(tmp_path / "pp.db")]) == 0
    out = capsys.readouterr().out
    assert "RALLY" in out and "rafae" in out and "12" in out


def test_board_without_a_database_says_there_are_no_games_and_creates_nothing(tmp_path, capsys):
    assert play.main(["--board", "--db", str(tmp_path / "none.db")]) == 0
    assert "no games yet" in capsys.readouterr().out and not (tmp_path / "none.db").exists()


def test_the_fake_window_loop_plays_balls_from_the_mouse_and_the_keys():
    from pingpong import app
    from pingpong.clock import FakeClock

    clock = FakeClock(start_ns=1_000_000_000)
    session = app.make_session(clock=clock, client=None, source="fake", seed=1)
    mouse = {"xy": (play.W // 2, play.H // 2)}

    def mouse_xy():                                               # the "mouse" follows where the ball will arrive
        ball = session.game.incoming
        if ball is not None:
            mouse["xy"] = (ball.aim_ab[0] * play.W, (1.0 - ball.aim_ab[1]) * play.H)
        return mouse["xy"]

    def wait_key(ms):
        clock.advance_s(1 / 60)                                   # one frame passes per call
        game = session.game
        if game.tracker.streak >= 3:
            return ord("q")
        if game.phase == "LOBBY":
            return 32                                             # SPACE starts
        ball = game.incoming
        if game.phase == "RALLY" and ball is not None and clock.now_ns() >= ball.t_c_ns:
            return ord("k")                                       # a hard swing at the moment the ball arrives
        return 255

    frames = []
    play.fake_loop(session, show=frames.append, wait_key=wait_key, mouse_xy=mouse_xy)
    assert session.game.tracker.streak >= 3 and frames and frames[0].shape == (play.H, play.W, 3)


# --- one bad frame must not end the session ------------------------------------------------------------------------------
class FlakyRig:
    """A rig whose pump fails on chosen calls (a stand-in for a bug that only shows on hardware)."""

    def __init__(self, fail_on):
        self.calls, self.fail_on = 0, fail_on
        self.session = type("S", (), {"game": None})()

    def pump(self):
        self.calls += 1
        if self.fail_on(self.calls):
            raise ValueError(f"bad frame {self.calls}")

    def hud_state(self):
        from pingpong import hud

        return hud.HudState()

    def display_frame(self):
        return None


def test_a_frame_that_raises_is_reported_once_and_the_loop_carries_on():
    rig, logs, frames = FlakyRig(lambda n: n in (2, 3, 4)), [], []

    def wait_key(ms):
        return ord("q") if len(frames) >= 8 else 255

    play.run_loop(rig, show=frames.append, wait_key=wait_key, log=logs.append)
    assert rig.calls >= 8 and len(frames) >= 6
    assert len([m for m in logs if "bad frame" in m]) == 1                 # the same error is not printed three times


def test_a_loop_that_fails_every_frame_ends_with_the_error_instead_of_spinning_forever():
    rig = FlakyRig(lambda n: True)
    try:
        play.run_loop(rig, show=lambda f: None, wait_key=lambda ms: 255, log=lambda *_: None)
    except ValueError as exc:
        assert "bad frame" in str(exc) and rig.calls == play.MAX_BAD_FRAMES
    else:
        raise AssertionError("a permanently broken loop must not be swallowed")


def test_control_c_is_never_swallowed_by_the_frame_guard():
    import pytest

    class Impatient(FlakyRig):
        def pump(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        play.run_loop(Impatient(lambda n: False), show=lambda f: None, wait_key=lambda ms: 255, log=lambda *_: None)


def test_the_fake_game_starts_from_the_mouse_held_in_the_top_right_too():
    from pingpong.clock import FakeClock

    clock = FakeClock(start_ns=1_000_000_000)
    session = play.make_fake_session(play.parse_args(["--fake", "--no-audio"]), clock=clock)
    assert session.hold_start is not None
    calls = {"n": 0}

    def mouse_xy():
        calls["n"] += 1
        clock.advance_s(1 / 30)
        return (play.W * 0.5, play.H * 0.7) if calls["n"] < 10 else (play.W * 0.95, play.H * 0.12)    # then the top right

    keys = iter([255] * 100 + [ord("q")])
    play.fake_loop(session, show=lambda frame: None, wait_key=lambda ms: next(keys), mouse_xy=mouse_xy)
    assert session.game.phase in ("COUNTDOWN", "RALLY")


def test_live_play_hits_by_hand_contact_unless_asked_for_swings():
    assert play.parse_args([]).hit_mode == "contact"
    assert play.parse_args(["--hit-mode", "swing"]).hit_mode == "swing"

