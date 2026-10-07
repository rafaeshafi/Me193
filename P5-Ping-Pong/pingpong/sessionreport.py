"""sessionreport: a recorded session -> the numbers that tune the game.

Pure functions over recorder.load(): summarize() returns a dict, format_report() makes the text
the report tool prints.  This is what turns "it did not register my swing" into "J2 failed on 4
of 9 swings: only 0 confident pose frames in the approach window".
"""

import re
from collections import Counter, defaultdict

import numpy as np

from pingpong import benchstats

S = 1_000_000_000
GATES = ("J1", "J2", "J3", "J4", "J5", "J6")


def _spread(values, *extra):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    out = {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
           "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}
    return out


def _cross_offsets(verdicts):
    """The camera-vs-IMU offsets (ms) that gate J4 measured on judged swings."""
    out = []
    for v in verdicts:
        for g in v["gates"]:
            match = g["name"] == "J4" and re.search(r"pose peak ([+-]\d+) ms", g["note"])
            if match:
                out.append(float(match.group(1)))
    return out


def _pauses(events, t0_ns, end_ns):
    """[{"reasons", "seconds", "start_s"}] from the pause transitions."""
    out, start, reasons = [], None, []
    for e in events:
        if e["k"] != "pause":
            continue
        now = e["d"]["reasons"]
        if now and start is None:
            start, reasons = e["t"], list(now)
        elif now:
            reasons = sorted(set(reasons) | set(now))
        elif start is not None:
            out.append({"reasons": reasons, "seconds": (e["t"] - start) / S, "start_s": (start - t0_ns) / S})
            start = None
    if start is not None:
        out.append({"reasons": reasons, "seconds": (end_ns - start) / S, "start_s": (start - t0_ns) / S})
    return out


def summarize(loaded):
    meta, events = loaded.meta, loaded.events
    kinds = Counter(e["k"] for e in events)
    verdicts = [e["d"]["verdict"] for e in events if e["k"] == "verdict"]
    judged = [v for v in verdicts if v["kind"] in ("HIT", "REJECTED")]
    gate_failures, gate_notes = Counter(), defaultdict(list)
    for v in (v for v in judged if v["kind"] == "REJECTED"):
        for g in v["gates"]:
            if not g["passed"]:
                gate_failures[g["name"]] += 1
                if g["note"] not in gate_notes[g["name"]]:
                    gate_notes[g["name"]].append(g["note"])
    gate_pass = {name: [sum(1 for v in judged for g in v["gates"] if g["name"] == name and g["passed"]),
                        sum(1 for v in judged for g in v["gates"] if g["name"] == name)] for name in GATES}
    hits = [e["d"] for e in events if e["k"] == "hit"]
    imu_t = [s.t_ns for s in loaded.imu]
    pose_t = [p.t_scene_ns for p in loaded.poses]
    t0 = meta.get("t0_ns", imu_t[0] if imu_t else 0)
    end = max([imu_t[-1] if imu_t else t0] + [e["t"] for e in events])
    summary = next((e["d"] for e in reversed(events) if e["k"] == "summary"), None)
    pose_span = (pose_t[-1] - pose_t[0]) / S if len(pose_t) > 1 else 0.0
    return {
        "player": meta.get("player"), "level": _level_name(meta, events), "mode": meta.get("mode"),
        "source": meta.get("source"), "seed": meta.get("seed"), "duration_s": (end - t0) / S,
        "balls": kinds["serve"], "swings": kinds["swing"], "hits": kinds["hit"], "misses": kinds["miss"],
        "faults": kinds["fault"], "rejected": sum(1 for v in verdicts if v["kind"] == "REJECTED"),
        "ignored": sum(1 for v in verdicts if v["kind"] == "IGNORED"),
        "best_streak": max([h.get("streak", 0) for h in hits], default=0),
        "game_over": kinds["game_over"] > 0, "gate_failures": dict(gate_failures),
        "gate_notes": {k: v[:3] for k, v in gate_notes.items()}, "gate_pass": gate_pass,
        "timing_ms": _spread([v["e_s"] * 1000.0 for v in verdicts if v["kind"] == "HIT"]),
        "w_pk": _spread([e["d"]["w_pk"] for e in events if e["k"] == "swing"]),
        "cross_ms": _spread(_cross_offsets(judged)), "camera_lag_s": meta.get("camera_lag_s"),
        "kmh": _spread([h["kmh"] for h in hits]), "labels": dict(Counter(h["label"] for h in hits)),
        "hub": benchstats.rate_stats(imu_t),
        "pose": {"n": len(pose_t), "fps": (len(pose_t) - 1) / pose_span if pose_span > 0 else 0.0},
        "pauses": _pauses(events, t0, end), "hub_trouble": sum(1 for e in events if e["k"] == "hub" and e["d"]["status"] != "ok"),
        "loop_p95_ms": summary["loop"]["p95_ms"] if summary else None,
    }


def _level_name(meta, events):
    from pingpong import levels

    tags = [e["d"]["level"] for e in events if e["k"] == "phase" and e["d"].get("phase") == "COUNTDOWN"]
    tag = tags[-1] if tags else meta.get("level", 1)
    return levels.LEVELS[tag].name


def format_report(s):
    t, w, k = s["timing_ms"], s["w_pk"], s["kmh"]
    lines = [f"Session: {s['player']} / {s['level']} {s['mode']} / source {s['source']} / {s['duration_s']:.1f} s",
             f"Balls {s['balls']} | swings {s['swings']} | hits {s['hits']} | misses {s['misses']} | faults "
             f"{s['faults']} | rejected {s['rejected']} | ignored {s['ignored']} | best streak {s['best_streak']}"]
    if t["n"]:
        lines.append(f"Hit timing vs the ball (ms, negative = early): mean {t['mean']:+.0f}, p10 {t['p10']:+.0f}, "
                     f"p90 {t['p90']:+.0f} (n {t['n']})")
    if w["n"]:
        lines.append(f"Swing peak (dps): median {w['median']:.0f}, p10 {w['p10']:.0f}, p90 {w['p90']:.0f}, "
                     f"max {w['max']:.0f}")
    if k["n"]:
        lines.append(f"Hit speed (km/h): mean {k['mean']:.0f}, max {k['max']:.0f}; quality "
                     + ", ".join(f"{name} {n}" for name, n in sorted(s["labels"].items())))
    lines.append("Gates passed/judged: " + "  ".join(f"{g} {p}/{n}" for g, (p, n) in s["gate_pass"].items()))
    for gate, count in sorted(s["gate_failures"].items()):
        lines.append(f"  rejected by {gate} x{count}: " + " | ".join(s["gate_notes"][gate]))
    cross = s["cross_ms"]
    if cross["n"]:
        lag = s["camera_lag_s"]
        advice = ""
        if lag is not None:
            if abs(cross["median"]) < 20.0:
                advice = f": keep CAMERA_LAG_S = {lag:.3f}"
            else:
                advice = f": try CAMERA_LAG_S = {max(0.0, lag + cross['median'] / 1000.0):.3f} (now {lag:.3f})"
        lines.append(f"Camera vs IMU (gate J4, {cross['n']} swings): the camera's hand-speed peak comes "
                     f"{cross['median']:+.0f} ms from the IMU peak (p10 {cross['p10']:+.0f}, p90 {cross['p90']:+.0f})"
                     + advice)
    hub, pose = s["hub"], s["pose"]
    lines.append(f"Hub: {hub['hz']:.1f} Hz, worst gap {hub['worst_gap_ms']:.0f} ms, {hub['gaps_over_100ms']} gaps over "
                 f"100 ms; pose {pose['fps']:.1f} fps ({pose['n']} readings)")
    for p in s["pauses"]:
        lines.append(f"  paused ({', '.join(p['reasons'])}) for {p['seconds']:.1f} s at {p['start_s']:.1f} s")
    if s["loop_p95_ms"] is not None:
        lines.append(f"Main loop p95: {s['loop_p95_ms']:.2f} ms")
    return "\n".join(lines)
