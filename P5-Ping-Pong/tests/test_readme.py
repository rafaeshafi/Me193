"""The README must not drift from the code: its commands and flags are real."""

import re
from pathlib import Path

import play

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def test_every_pp_command_the_readme_names_exists():
    pp = (ROOT / "pp").read_text()
    commands = set(re.findall(r"\./pp (\w+)", README))
    assert {"ready", "play", "env_check", "calibrate_swing", "report", "replay"} <= commands
    for command in commands:
        assert command in pp or (ROOT / "tools" / f"{command}.py").exists(), f"./pp {command} is not a command"


def test_every_tool_command_the_readme_names_has_its_script():
    for command in set(re.findall(r"\./pp (\w+)", README)):
        if command not in ("ready", "play", "help"):
            assert (ROOT / "tools" / f"{command}.py").exists(), command


def test_every_flag_on_a_play_line_in_the_readme_is_a_real_flag():
    flags = {opt for action in play.make_parser()._actions for opt in action.option_strings}
    used = set()
    for line in README.splitlines():
        if "./pp play" in line:
            used |= set(re.findall(r"--[a-z][a-z-]*", line))
    assert {"--player", "--fake", "--no-publish", "--board", "--mode", "--level"} <= used
    assert used <= flags, f"README mentions flags play.py does not have: {sorted(used - flags)}"


def test_the_readme_has_the_questions_the_assignment_asks():
    for heading in ("(a) Describe the policy", "(b) What are the potential limitations", "(c) Which AI/ML algorithms",
                    "What we learned", "most proud of", "What took a while", "Most spectacular failure"):
        assert heading in README, heading


def test_the_readme_does_not_claim_algorithms_that_are_not_built():
    # (c) may only list what ships; the spin classifier and Q-learning are claimed only once their modules exist.
    section = README.split("### (c)")[1].split("### Reflection")[0].lower()
    claimed = section.split("not built (so not claimed)")[0]                  # the table, not the disclaimer line
    for name, module in (("logistic", "spin.py"), ("q-learning", "qbandit.py")):
        built = (ROOT / "pingpong" / module).exists()
        assert built or name not in claimed, f"README (c) claims {name} but pingpong/{module} does not exist"


# Every algorithm named in (c) must be greppable: the claim, the module and a symbol only that algorithm has.
ALGORITHMS = {
    "BlazePose": ("pose_features.py", "PoseLandmarker"),
    "AprilTag": ("tags.py", "aruco"),
    "One-Euro": ("oneeuro.py", "class OneEuro"),
    "Signed-axis swing detector": ("swing.py", "class SwingDetector"),
    "FFT": ("shake.py", "np.fft.rfft"),
    "PD controller": ("pd.py", "class PDController"),
    "Softmax": ("policy.py", "math.exp"),
    "Logistic Regression": ("spin.py", "LogisticRegression"),
    "Q-learning": ("qbandit.py", "def update"),
    "Cross-correlation": ("benchstats.py", "corrcoef"),
}


def test_every_algorithm_the_readme_names_is_in_the_code_where_it_says_it_is():
    section = README.split("### (c)")[1].split("### Reflection")[0]
    named = [name for name in ALGORITHMS if name.lower() in section.lower()]
    assert len(named) == len(ALGORITHMS), f"README (c) should name all of {sorted(ALGORITHMS)}; names {named}"
    for name, (module, symbol) in ALGORITHMS.items():
        source = (ROOT / "pingpong" / module).read_text()
        assert symbol in source, f"README (c) names {name} but {module} has no {symbol!r}"
        assert module in section, f"README (c) does not point at {module} for {name}"
