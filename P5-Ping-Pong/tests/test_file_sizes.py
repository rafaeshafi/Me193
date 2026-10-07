"""~/CLAUDE.md: keep files under 500 lines (code; the copied design docs are reference text)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 500
SKIP_DIRS = {"my_env", "__pycache__", ".pytest_cache", "data", "recordings"}


def find_long_files(root, limit=LIMIT):
    long_files = []
    for path in Path(root).rglob("*.py"):
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        n = sum(1 for _ in path.open(encoding="utf-8"))
        if n >= limit:
            long_files.append((str(path.relative_to(root)), n))
    return long_files


def test_the_checker_flags_a_file_that_is_too_long(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n" * 10)
    (tmp_path / "big.py").write_text("x = 1\n" * 501)
    assert find_long_files(tmp_path) == [("big.py", 501)]


def test_every_python_file_in_the_project_is_under_500_lines():
    assert find_long_files(ROOT) == []
