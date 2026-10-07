import play


def test_selftest_runs_a_scripted_ten_hit_rally_and_checks_the_published_scores():
    assert play.main(["--selftest"]) == 0


def test_live_mode_without_hardware_support_explains_itself_instead_of_crashing(capsys):
    code = play.main(["--no-window"])
    err = capsys.readouterr().err
    assert code != 0 and "live" in err.lower()


def test_argument_defaults():
    args = play.parse_args([])
    assert args.level == 1 and args.mode == "survival" and args.fake is False and args.no_publish is False
