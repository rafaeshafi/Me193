"""Refuse to touch Bluetooth/camera from a host app macOS will kill for it.

CoreBluetooth and AVFoundation need the *responsible app* to carry a privacy
usage description.  Terminal.app, iTerm and VS Code do; the Claude desktop app's
shell does not, and the process then aborts (exit 134, TCC) with no message.
Tools that need BLE or the camera call require_host() first.
"""

import os
import sys

_BAD_HOSTS = ("com.anthropic.claudefordesktop",)


def check_host(env, what):
    """-> (ok, message).  Unknown/missing hosts are allowed: only known-bad ones are refused."""
    if env.get("PP_ALLOW_HOST") == "1":
        return True, "host check overridden by PP_ALLOW_HOST=1"
    bundle = env.get("__CFBundleIdentifier", "")
    if bundle in _BAD_HOSTS:
        return False, (
            f"{what} cannot be used from this app ({bundle}): macOS aborts the process because the "
            f"app has no {what} permission.\nRun this from Terminal.app instead "
            f"(System Settings > Privacy & Security > {what} must allow Terminal), e.g.\n"
            f"    cd ~/ME193/P5-Ping-Pong && ./pp <tool>")
    return True, f"host {bundle or 'unknown'} accepted"


def require_host(what, env=None):
    ok, message = check_host(os.environ if env is None else env, what)
    if not ok:
        print(message, file=sys.stderr)
        raise SystemExit(2)
