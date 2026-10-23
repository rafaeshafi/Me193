"""Decision trees exported from scikit-learn as JSON, evaluated here without scikit-learn, plus a majority vote."""
import json
from collections import Counter, deque


class Tree:
    """JSON: {"features": [...], "classes": [...], "root": node}; node = {"f", "t", "l", "r"} or {"label"}.
    Goes left when features[f] <= t, exactly as scikit-learn does."""

    def __init__(self, spec):
        self.features = list(spec["features"])
        self.classes = list(spec["classes"])
        self.root = spec["root"]
        self._check(self.root)

    def _check(self, node):
        """A bad export fails at load time, not on the first window."""
        if "label" in node:
            return
        if node["f"] not in self.features:
            raise ValueError("tree splits on %r, which is not in its feature list" % node["f"])
        self._check(node["l"])
        self._check(node["r"])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def predict(self, features):
        node = self.root
        while "label" not in node:
            node = node["l"] if features[node["f"]] <= node["t"] else node["r"]
        return node["label"]

    def depth(self):
        def d(node):
            return 0 if "label" in node else 1 + max(d(node["l"]), d(node["r"]))
        return d(self.root)


class Vote:
    """Majority of the last n labels; ties go to the most recent label among the tied."""

    def __init__(self, n=3):
        self._h = deque(maxlen=n)

    def push(self, label):
        self._h.append(label)
        counts = Counter(self._h)
        top = max(counts.values())
        for lab in reversed(self._h):
            if counts[lab] == top:
                return lab
