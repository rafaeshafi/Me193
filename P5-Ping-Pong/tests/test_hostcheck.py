"""BLE and the camera only work from an app macOS lets use them.

Verified 2026-10-06: running CoreBluetooth from the Claude desktop app's shell
aborts the process (TCC, responsible process "claude").  The tools must say so
instead of dying with exit code 134.
"""

import pytest

from pingpong import hostcheck


def test_claude_desktop_shell_is_rejected_with_an_actionable_message():
    ok, message = hostcheck.check_host({"__CFBundleIdentifier": "com.anthropic.claudefordesktop"}, "Bluetooth")
    assert ok is False
    assert "Terminal.app" in message and "Bluetooth" in message


@pytest.mark.parametrize("bundle", ["com.apple.Terminal", "com.googlecode.iterm2", "com.microsoft.VSCode"])
def test_real_terminals_are_accepted(bundle):
    ok, _ = hostcheck.check_host({"__CFBundleIdentifier": bundle}, "camera")
    assert ok is True


def test_override_variable_allows_any_host():
    env = {"__CFBundleIdentifier": "com.anthropic.claudefordesktop", "PP_ALLOW_HOST": "1"}
    ok, _ = hostcheck.check_host(env, "Bluetooth")
    assert ok is True


def test_missing_bundle_id_is_accepted_because_it_cannot_be_judged():
    ok, _ = hostcheck.check_host({}, "Bluetooth")
    assert ok is True


def test_require_exits_with_the_message_when_the_host_is_bad(capsys):
    with pytest.raises(SystemExit) as exc:
        hostcheck.require_host("Bluetooth", env={"__CFBundleIdentifier": "com.anthropic.claudefordesktop"})
    assert exc.value.code == 2
    assert "Terminal.app" in capsys.readouterr().err
