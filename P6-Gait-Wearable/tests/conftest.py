"""Make the board-side package and the training scripts importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("app/python", "train"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
