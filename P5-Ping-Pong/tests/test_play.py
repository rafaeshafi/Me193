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


def test_board_prints_the_leaderboard_without_touching_any_hardware(tmp_path, capsys):
    from pingpong import store

    db = store.Store(tmp_path / "pp.db")
    db.record_game("rafae", dict(mode="survival", level="Club", target=7, streak=12, record=12, hits=12, misses=1,
                                 faults=0, max_kmh=30.0, duration_s=40.0, source="live"))
    db.close()
    assert play.main(["--board", "--db", str(tmp_path / "pp.db")]) == 0
    out = capsys.readouterr().out
    assert "SURVIVAL" in out and "rafae" in out and "12" in out


def test_board_without_a_database_says_there_are_no_games_and_creates_nothing(tmp_path, capsys):
    assert play.main(["--board", "--db", str(tmp_path / "none.db")]) == 0
    assert "no games yet" in capsys.readouterr().out and not (tmp_path / "none.db").exists()
