"""Make the shared package, the training scripts and the board app importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "train", "tools", "app/python"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
