"""Guided hub bench: IMU rate (P1), units (P2), clipping (P3) and the swing fixtures.

Usage (Terminal.app only -- Bluetooth):
    ./pp bench_hub --guided                      # the full ~25 minute session
    ./pp bench_hub --only rate,faces,turns       # just the measurements
    ./pp bench_hub --only soft,hard              # re-record some fixtures
    ./pp bench_hub --selftest

Every take is announced, then a beep tells you to go.  Takes are saved under
recordings/fixtures/<label>.jsonl (the swing detector is tuned offline on them);
trustworthy numbers go to config_local.json (ACCEL_PER_G, GYRO_PER_DPS, HUB_FS_RAW,
HUB_RATE_HZ ...), untrustworthy ones are reported and NOT written.
"""

import argparse
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import config  # noqa: E402
from pingpong import benchstats, fixtures  # noqa: E402
from pingpong.hub import card_kwargs  # noqa: E402


@dataclass
class Step:
    label: str
    count: int
    secs: float
    text: str


STEPS = [
    Step("rate", 1, 30.0, "Hold the hub in your fist at the play position (~1.8 m from the laptop, body between "
                          "hub and laptop) and swing GENTLY for the whole window."),
    Step("faces", 6, 1.5, "Rest the hub FLAT on a table with a DIFFERENT face up each time (all six), and keep it still."),
    Step("turns", 3, 6.0, "Turn the hub ONE full 360 degrees about one axis, smoothly, between about 1 s and 5 s after the beep."),
    Step("max", 10, 2.5, "MAX-effort forward swing right after the beep."),
    Step("soft", 20, 2.5, "SOFT forward swing right after the beep."),
    Step("hard", 20, 2.5, "HARD forward swing right after the beep."),
    Step("backswing_only", 10, 2.5, "Draw the paddle BACK only (a backswing), then stop. NO forward swing."),
    Step("shakes", 10, 2.5, "SHAKE the hub back and forth for the whole window."),
    Step("waving", 1, 30.0, "Wave the hub back and forth at about 3 Hz, like a fan, for the whole window."),
]


LEAD_S = {"rate": 6.0}        # the play position is 1.8 m from the laptop where the command was typed
NEW_STEP_LEAD_S = 3.0         # a different kind of take: pick the hub up, get into position
TAKE_GAP_S = 1.0              # the next take of the same step: just a breath


def make_prompt(env, link, announced, out):
    """prompt(step, index): announce a step once, give time to get ready, then a beep means GO."""
    def prompt(step, index):
        if step.label not in announced:
            announced.add(step.label)
            lead = LEAD_S.get(step.label, NEW_STEP_LEAD_S)
            out(f"\n== {step.label} x{step.count} ({step.secs:.0f} s each): {step.text}")
            out(f"   starting in {lead:.0f} s: get into position, then follow the beeps")
            env.sleep(lead)
        else:
            env.sleep(TAKE_GAP_S)
        link.dev.beep(frequency=880, blocking=False)         # GO

    return prompt


def run_steps(env, link, steps, prompt, out, fixtures_dir):
    """Record every take of every step; returns {label: [samples, ...]}."""
    takes = {}
    for step in steps:
        for index in range(step.count):
            prompt(step, index)
            while not link.imu.empty():
                link.imu.get_nowait()
            samples = []
            t0 = env.clock.now_ns()
            while (env.clock.now_ns() - t0) / 1e9 < step.secs:
                env.sleep(0.02)
                while not link.imu.empty():
                    samples.append(link.imu.get_nowait())
            takes.setdefault(step.label, []).append(samples)
            fixtures.save_take(fixtures_dir, step.label, index, samples)
            out(f"  {step.label} {index + 1}/{step.count}: {len(samples)} samples")
    return takes


def _longest_run(values, target):
    best = run = 0
    for v in values:
        run = run + 1 if v == target else 0
        best = max(best, run)
    return best


def _clip_stats(max_takes, channel):
    """Largest raw value on the gyro ("g") or accelerometer ("a") in the max-effort takes, and whether it plateaus."""
    peaks = [(int(np.max(np.abs([getattr(s, channel)[k] for s in take]))), k, take)
             for take in max_takes for k in range(3)]
    top = max(p[0] for p in peaks)
    run = 0
    for value, k, take in peaks:
        if value == top:
            run = max(run, _longest_run([abs(getattr(s, channel)[k]) for s in take], top))
    return {"max_raw": top, "plateau": run >= 3, "fs_raw": top if run >= 3 else None}


def _quiet_accel(takes, quiet_dps=12.0, min_s=0.3, min_windows=6):
    """Counts per g from every moment the hub lay still (gyro quiet for 0.3 s) in ANY take: at rest |a| = 1 g."""
    magnitudes = []
    for label_takes in takes.values():
        for take in label_takes:
            g = np.array([s.g for s in take], dtype=float)
            a = np.array([s.a for s in take], dtype=float)
            t = np.array([s.t_ns for s in take], dtype=float) / 1e9
            if len(t) < 3:
                continue
            still = np.linalg.norm(g, axis=1) < quiet_dps            # the gyro's rest offset is a few counts at most
            i = 0
            while i < len(t):
                if not still[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(t) and still[j + 1]:
                    j += 1
                if t[j] - t[i] >= min_s:
                    magnitudes.append(float(np.linalg.norm(a[i:j + 1].mean(axis=0))))
                i = j + 1
    if len(magnitudes) < min_windows:
        return None
    mean = float(np.median(magnitudes))
    cv = float(np.std(magnitudes) / mean) if mean else float("inf")
    return {"accel_per_g": mean, "cv": cv, "ok": cv <= 0.03, "source": f"{len(magnitudes)} quiet moments"}


def analyse(takes):
    """Pure analysis of recorded takes -> measurements (only for the labels present)."""
    m = {}
    if takes.get("rate"):
        m["rate"] = benchstats.rate_stats([s.t_ns for s in takes["rate"][0]])
    if takes.get("faces"):
        means = [tuple(float(np.mean([s.a[k] for s in take])) for k in range(3)) for take in takes["faces"]]
        m["accel"] = dict(benchstats.accel_scale_from_faces(means), source="the six faces")
        if not m["accel"]["ok"]:                             # the faces were not still (it is hard to flip a hub in 1 s)
            quiet = _quiet_accel(takes)
            if quiet is not None and quiet["ok"]:
                m["accel"] = quiet
    if takes.get("turns"):
        scales = []
        for take in takes["turns"]:
            t0 = take[0].t_ns
            per_axis = [benchstats.gyro_scale_from_turn(
                [((s.t_ns - t0) / 1e9, s.g[k]) for s in take], 360.0) for k in range(3)]
            scales.append(max(per_axis))          # the turned axis integrates to the largest value
        mean = float(np.mean(scales))
        spread = (max(scales) - min(scales)) / mean if mean else float("inf")
        m["gyro"] = {"gyro_per_dps": mean, "spread": spread, "ok": len(scales) >= 2 and spread <= 0.05}
    if takes.get("max"):
        # HUB_FS_RAW is the GYRO's full scale (the swing detector's clip flag): the real hub's accelerometer
        # saturates at 8011 (~8 g) in a hard swing, which says nothing about the gyro, so they are judged apart.
        m["clip"] = _clip_stats(takes["max"], "g")
        m["accel_clip"] = _clip_stats(takes["max"], "a")
    return m


def apply(m, config_path):
    """Write the trustworthy measurements to config_local.json; return what was written."""
    updates = {}
    if "rate" in m and m["rate"]["n"] > 1:
        updates.update(HUB_RATE_HZ=round(m["rate"]["hz"], 1), HUB_WORST_GAP_MS=round(m["rate"]["worst_gap_ms"], 1),
                       HUB_P999_GAP_MS=round(m["rate"]["p999_gap_ms"], 1))
    if m.get("accel", {}).get("ok"):
        updates["ACCEL_PER_G"] = round(m["accel"]["accel_per_g"], 2)
    if m.get("gyro", {}).get("ok"):
        updates["GYRO_PER_DPS"] = round(m["gyro"]["gyro_per_dps"], 3)
    if m.get("clip", {}).get("plateau"):
        updates["HUB_FS_RAW"] = int(m["clip"]["fs_raw"])
    if updates:
        config.write_local(updates, config_path)
    return updates


def report(m):
    lines = []
    if "rate" in m:
        r = m["rate"]
        lines.append(f"rate   : {r['hz']:.1f} Hz, worst gap {r['worst_gap_ms']:.0f} ms, p99.9 {r['p999_gap_ms']:.0f} ms "
                     f"-> {benchstats.rate_verdict(r['hz'])}")
    if "accel" in m:
        a = m["accel"]
        lines.append(f"accel  : {a['accel_per_g']:.1f} counts/g from {a.get('source', 'the six faces')}, spread {a['cv'] * 100:.1f}% "
                     f"-> {'OK' if a['ok'] else 'NOT TRUSTED (>3%): rest the hub flat and still, re-run --only faces'}")
    if "gyro" in m:
        g = m["gyro"]
        lines.append(f"gyro   : {g['gyro_per_dps']:.2f} counts per deg/s, turn spread {g['spread'] * 100:.1f}% "
                     f"-> {'OK' if g['ok'] else 'NOT TRUSTED (>5%): turn smoothly and exactly 360 degrees, re-run --only turns'}")
    if "clip" in m:
        c = m["clip"]
        lines.append(f"clip   : gyro max raw {c['max_raw']}, plateau {'YES -> HUB_FS_RAW saved' if c['plateau'] else 'no (the gyro did not clip)'}")
    if m.get("accel_clip", {}).get("plateau"):
        lines.append(f"         the accelerometer saturates at {m['accel_clip']['max_raw']} (about 8 g): fine, nothing uses that range")
    return lines


def _selftest():
    from pingpong.sources_fake import FakeEnv

    env = FakeEnv(hz=62.0)
    link = env.make_hub(15, {"card_color": 5, "card_serial": "0997"})
    link.connect()
    faces = [(980, 0, 0), (-980, 0, 0), (0, 980, 0), (0, -980, 0), (0, 0, 980), (0, 0, -980)]

    def prompt(step, index):
        t0 = env.clock.now_ns()

        def scenario(now):
            t = (now - t0) / 1e9
            if step.label == "faces":
                return (*faces[index], 0, 0, 0)
            if step.label == "turns":
                return (0, 0, 1000, 0, 0, 900 if 1.0 <= t < 5.0 else 0)
            if step.label == "max":
                return (0, 0, 1000, 32767 if 1.0 <= t < 1.2 else 0, 0, 0)
            return (0, 0, 1000, 0, 0, 0)

        env.scenario = scenario

    quick = [Step(s.label, s.count, min(s.secs, 8.0) if s.label == "rate" else s.secs, s.text)
             for s in STEPS if s.label in ("rate", "faces", "turns", "max")]
    with tempfile.TemporaryDirectory() as tmp:
        takes = run_steps(env, link, quick, prompt, out=lambda *_: None, fixtures_dir=Path(tmp) / "fx")
        m = analyse(takes)
        written = apply(m, Path(tmp) / "c.json")
    assert m["accel"]["ok"] and m["gyro"]["ok"] and m["clip"]["plateau"], m
    assert {"ACCEL_PER_G", "GYRO_PER_DPS", "HUB_FS_RAW", "HUB_RATE_HZ"} <= set(written), written
    print("bench_hub selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=config.CARD_COLOR)
    ap.add_argument("--card-serial", default=config.CARD_SERIAL)
    ap.add_argument("--only", default="", help="comma list of step labels (default: all)")
    ap.add_argument("--guided", action="store_true", help="(default behaviour)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    from pingpong import hostcheck

    hostcheck.require_host("Bluetooth")
    if not args.card_color or not args.card_serial:
        print("No hub card configured. Run ./pp scan_hubs, then ./pp env_check --card-color C --card-serial NNNN "
              "(it saves them), or pass --card-color/--card-serial.", file=sys.stderr)
        return 1
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    steps = [s for s in STEPS if not only or s.label in only]
    unknown = only - {s.label for s in STEPS}
    if unknown:
        print(f"unknown step(s): {sorted(unknown)}; choose from {[s.label for s in STEPS]}", file=sys.stderr)
        return 1

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    from pingpong.realenv import RealEnv

    env = RealEnv()
    link = env.make_hub(config.NOTIFY_MS, card_kwargs(args.card_color, args.card_serial))
    try:
        link.connect()
        print(f"connected; battery {link.battery_pct()}%  ({sum(s.count for s in steps)} takes)")
        prompt = make_prompt(env, link, set(), print)
        takes = run_steps(env, link, steps, prompt, out=print,
                          fixtures_dir=config.HERE / "recordings" / "fixtures")
        m = analyse(takes)
        written = apply(m, config.LOCAL_PATH)
        print("\n".join(["", *report(m)]))
        print("saved to config_local.json:", written or "nothing (re-run the untrusted steps)")
        return 0
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
