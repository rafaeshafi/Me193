"""Recorded IMU takes (swings, shakes, rest) as JSON lines.

One file per label under recordings/fixtures/; every line is one take.  The same
files feed the offline swing-detector tests, so a failed live attempt becomes a
reproducible test instead of a verbal description.
"""

import json
import re
from pathlib import Path

from pingpong.events import ImuSample

_LABEL = re.compile(r"^[A-Za-z0-9_-]+$")


def _path(directory, label):
    if not _LABEL.match(label):
        raise ValueError(f"bad fixture label {label!r}")
    return Path(directory) / f"{label}.jsonl"


def save_take(directory, label, index, samples, meta=None):
    path = _path(directory, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "label": label, "index": index, "meta": meta or {},
        "samples": [[s.t_ns, *s.g, *s.a, s.src] for s in samples],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return path


def load_takes(path):
    path = Path(path)
    if not path.exists():
        return []
    takes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        d["samples"] = [ImuSample(t_ns=int(r[0]), g=tuple(r[1:4]), a=tuple(r[4:7]), src=r[7])
                        for r in d["samples"]]
        takes.append(d)
    return takes


def load_all(directory):
    """{label: [take, ...]} for every fixture file in a directory."""
    return {p.stem: load_takes(p) for p in sorted(Path(directory).glob("*.jsonl"))}
