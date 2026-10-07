"""Interface-drift guard for modules written in parallel: everything must import,
and tools must only reach hardware through the shared pieces."""

import importlib
import pkgutil
from pathlib import Path

import pingpong

ROOT = Path(__file__).resolve().parents[1]


def _module_names():
    return [m.name for m in pkgutil.iter_modules(pingpong.__path__)]


def test_every_pingpong_module_imports_without_touching_hardware():
    names = _module_names()
    assert "events" in names and "hub" in names
    for name in names:
        importlib.import_module(f"pingpong.{name}")


def test_every_tool_imports_and_exposes_main():
    tools = sorted(p.stem for p in (ROOT / "tools").glob("*.py"))
    assert tools, "no tools found"
    for name in tools:
        module = importlib.import_module(f"tools.{name}")
        assert callable(getattr(module, "main", None)), f"tools/{name}.py needs main()"


def test_every_tool_with_a_selftest_flag_runs_it_green():
    # ./pp ready runs these too; keeping the check here makes pytest alone enough.
    for path in sorted((ROOT / "tools").glob("*.py")):
        if "--selftest" not in path.read_text():
            continue
        module = importlib.import_module(f"tools.{path.stem}")
        assert module.main(["--selftest"]) == 0, path.name


def test_the_hub_is_only_constructed_through_hublink_or_realenv():
    # Raw le.DoubleMotor() calls elsewhere would bypass the callback/teardown rules in hub.py.
    allowed = {"hub.py", "realenv.py"}
    offenders = []
    for path in (ROOT / "pingpong").glob("*.py"):
        if path.name not in allowed and "le.DoubleMotor(" in path.read_text():
            offenders.append(path.name)
    assert offenders == []
