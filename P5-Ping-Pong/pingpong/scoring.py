"""Streak / record bookkeeping (the number published to ME193/Rogers/RafaeShafi).

streak = consecutive valid player returns in the current rally (resets on a miss)
record = best streak this run (never decreases; optionally seeded)
value()  = what goes on the wire, by config.RECORD_SCOPE:
    record_session / record_alltime -> max(record, streak)   (holds after a miss)
    live_streak                     -> the running streak
"""

import config


class ScoreTracker:
    def __init__(self, scope="record_session", seed_record=0):
        if scope not in config.RECORD_SCOPES:
            raise ValueError(f"scope must be one of {config.RECORD_SCOPES}, got {scope!r}")
        self.scope = scope
        self.streak = 0
        self.record = 0 if scope == "live_streak" else int(seed_record)
        self._last_id = None

    def on_valid_hit(self, hit_id):
        """Count one valid return; ids must increase, so a duplicate delivery is ignored."""
        if self._last_id is not None and hit_id <= self._last_id:
            return False
        self._last_id = hit_id
        self.streak += 1
        self.record = max(self.record, self.streak)
        return True

    def end_rally(self):
        final, self.streak = self.streak, 0
        return final

    def value(self):
        return self.streak if self.scope == "live_streak" else max(self.record, self.streak)
