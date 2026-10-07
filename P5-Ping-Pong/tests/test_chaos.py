"""Chaos: the whole pipeline on fake hardware with random sensor trouble, checked against invariants.

Random hub and camera blackouts, shaking, weak or late swings, motor vibration and restarts, in many
different games.  Whatever happens: nothing throws, the published score only ever rises and never
exceeds the record, no game event happens while the game is paused, and balls are numbered in order.
"""

import random
import re

import pytest

import config
from pingpong import fakerig, recorder

S = 1_000_000_000


def chaos_session(seed, tmp_path, seconds=25.0):
    rng = random.Random(seed)
    rig = fakerig.FakeRig(
        level=rng.choice([1, 2, 3]), mode=rng.choice(["survival", "match"]), seed=seed, target=rng.choice([3, 5]),
        timing_s=rng.choice([0.0, 0.0, 0.05, 0.12, 0.3]), w_pk=rng.choice([600.0, 600.0, 900.0, 120.0]),
        vibration=rng.random() < 0.5, stale_ms=rng.choice([300.0, 500.0]), record_dir=tmp_path / f"s{seed}")
    t = 0.0
    while t < seconds:
        chunk = rng.uniform(0.5, 2.0)
        now = rig.now_s()
        roll = rng.random()
        if roll < 0.20:
            rig.hub_blackouts.append((now + rng.uniform(0, 0.5), now + rng.uniform(0.6, 2.2)))
        elif roll < 0.35:
            rig.pose_blackouts.append((now + rng.uniform(0, 0.5), now + rng.uniform(0.3, 2.0)))
        elif roll < 0.45:
            rig.shake_windows.append((now, now + rng.uniform(1.0, 2.5)))
        if rig.game.phase in ("LOBBY", "MATCH_OVER") and rng.random() < 0.8:
            rig.session.on_start()                                      # a key press: (re)start
        rig.run(seconds=chunk)
        t += chunk
    rig.close()
    return rig, recorder.load(tmp_path / f"s{seed}")


@pytest.mark.parametrize("seed", range(10))
def test_invariants_hold_under_random_sensor_trouble(seed, tmp_path):
    rig, loaded = chaos_session(seed, tmp_path)
    # 1. the published score rises strictly, as "N.0", and never exceeds the record
    sent = [p["payload"] for p in rig.client.published if p["topic"] == config.SCORE_TOPIC]
    assert all(re.fullmatch(r"\d+\.0", x) for x in sent)
    values = [float(x) for x in sent]
    assert values == sorted(set(values)), f"score published out of order: {values}"
    assert not values or values[-1] <= rig.game.tracker.record
    # 2. no game event inside a pause
    pauses, start = [], None
    for e in loaded.events:
        if e["k"] == "pause":
            if e["d"]["reasons"] and start is None:
                start = e["t"]
            elif not e["d"]["reasons"] and start is not None:
                pauses.append((start, e["t"]))
                start = None
    game_kinds = ("serve", "hit", "miss", "fault", "rally_end", "game_over", "point", "match_over", "verdict")
    for e in loaded.events:
        if e["k"] in game_kinds:
            assert not any(a < e["t"] < b for a, b in pauses), f"{e['k']} at {e['t']} happened during a pause"
    # 3. balls are numbered in order within the session
    serves = [e["d"]["ball_id"] for e in loaded.events if e["k"] == "serve"]
    assert serves == sorted(serves) and len(set(serves)) == len(serves)
    # 4. the record never decreases
    records = [e["d"]["record"] for e in loaded.events if e["k"] == "rally_end"]
    assert records == sorted(records)
