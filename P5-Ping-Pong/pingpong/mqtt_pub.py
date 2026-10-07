"""ScorePublisher: the only code allowed to write to ME193/Rogers/RafaeShafi.

Rules (all unit-tested):
  * payload is f"{float(x):.1f}" -- paho str()s ints, so "23" would lose the ".0"
  * QoS 1 + retain, so a late-joining dashboard sees the value and offline
    publishes survive a reconnect
  * NOTHING is published at startup (no 0.0): launching or crash-restarting never
    resets the retained value; the first publish is the first hit
  * record scopes publish increases only (the value holds after a miss);
    live_streak publishes every change
  * only a LIVE session (hub IMU + camera, not fake/sim/replay/demo) may touch the
    official topic; fake/sim/demo values go to .../demo/score; replay publishes nothing
  * --no-publish silences everything; there is never a will on the score topic
"""

import math

import config

_DEMO_SOURCES = ("fake", "sim", "demo")


class ScorePublisher:
    def __init__(self, client, *, topic=config.SCORE_TOPIC, scope="record_session", source="live",
                 no_publish=False, status_topic=config.STATUS_TOPIC, demo_topic=config.DEMO_SCORE_TOPIC):
        if scope not in config.RECORD_SCOPES:
            raise ValueError(f"unknown scope {scope!r}")
        self.client = client
        self.scope = scope
        self.source = source
        self.no_publish = no_publish
        self.topic = topic
        self.status_topic = status_topic
        self.demo_topic = demo_topic
        self._last = None          # last value sent (None until the first publish)
        self._floor = 0            # resume: never publish at or below this
        self._current = 0

    @staticmethod
    def format(value):
        value = float(value)
        if math.isnan(value) or value < 0:
            raise ValueError(f"score must be a non-negative number, got {value!r}")
        return f"{value:.1f}"

    def _target(self):
        if self.source == "live":
            return self.topic
        return self.demo_topic if self.source in _DEMO_SOURCES else None

    def _send(self, value):
        topic = self._target()
        if topic is None or self.no_publish:
            return
        self.client.publish(topic, self.format(value), qos=1, retain=True)
        self._last = int(value)

    def update(self, value):
        """Feed the tracker's current value; publishes only what the scope says to."""
        payload_check = self.format(value)          # validates even when silent
        del payload_check
        self._current = int(value)
        if self.no_publish or self._target() is None:
            return
        if self.scope == "live_streak":
            if self._last is None and self._current == 0:
                return                               # no 0.0 before the first hit
            if self._current != self._last:
                self._send(self._current)
        else:
            if self._current > max(self._last or 0, self._floor):
                self._send(self._current)

    def on_connect(self):
        """Called on every (re)connect: announce online; republish the value only if above zero."""
        if self.no_publish:
            return
        self.client.publish(self.status_topic, "online", qos=1, retain=True)
        if self._last and self._last > 0:
            self._send(self._last)

    def close(self):
        """Session end: only live_streak can have an unsent final value."""
        if self.no_publish or self._target() is None:
            return
        if self.scope == "live_streak" and self._last is not None and self._current != self._last:
            self._send(self._current)

    def publish_new_record_zero(self):
        """--new-record (after an on-screen confirm): deliberately reset the retained value."""
        self._floor = 0
        self._send(0)

    def resume_from(self, retained_text):
        """--resume (crash recovery of the graded run): keep the retained best as the floor."""
        try:
            value = float(retained_text)
        except (TypeError, ValueError):
            return
        if math.isnan(value) or value < 0:
            return
        self._floor = max(self._floor, int(value))
