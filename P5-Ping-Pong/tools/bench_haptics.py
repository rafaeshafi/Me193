"""Haptics bench: which motor recipes you feel, and how long the hub's own IMU rings after a pulse.

Usage (Terminal.app only -- Bluetooth):
    ./pp bench_haptics                       # hit_perfect, hit_early, fault: 3 pulses each, then you rate them
    ./pp bench_haptics --recipes record,hit_late --repeats 2
    ./pp bench_haptics --selftest

Hold the hub in your fist as you will play.  After each recipe you rate how strongly you felt it
(1 = nothing, 5 = strong).  The hub's own IMU is recorded around every pulse: how long it keeps
shaking after the motors stop becomes BLANK_AFTER_PULSE_S in config_local.json (the swing
detector ignores that long after each pulse, so the motors cannot fake a swing).  If nothing is
felt at 3/5 the game still works on beep + light; the report says how to feel more.
"""

import argparse
import json
import signal
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pingpong import benchstats, haptics  # noqa: E402
from pingpong.haptics import PATTERNS, ActuatorCore  # noqa: E402

from tools.bench_cam import CheckResult  # noqa: E402

DEFAULT_RECIPES = ("hit_perfect", "hit_early", "fault")
FELT = 3.0


def motor_recipes(names):
    bad = [n for n in names if n not in PATTERNS or not any(s.kind == "motors" for s in PATTERNS[n].steps)]
    if bad:
        raise ValueError(f"no motor pulses in: {', '.join(bad)} (those are beep + light only)")
    return tuple(names)


def _sleep(env, hub, collected, seconds):
    """Sleep in small steps, collecting every IMU sample that arrives."""
    end = env.clock.now_ns() + round(seconds * 1e9)
    while env.clock.now_ns() < end:
        env.sleep(0.02)
        while not hub.imu.empty():
            s = hub.imu.get_nowait()
            collected.append((s.t_ns, s.g, s.a))


def run(env, *, card, recipes=DEFAULT_RECIPES, repeats=3, config_path, report_path, prompt, rate, out):
    results = []

    def add(name, status, detail):
        results.append(CheckResult(name, status, detail))
        out(f"[{status:<4}] {name}: {detail}")

    def finish(code):
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps({"results": [asdict(r) for r in results]}, indent=2) + "\n")
        return results, code

    hub = env.make_hub(config.NOTIFY_MS, card)
    try:
        hub.connect()
    except ConnectionError as exc:
        add("hub_connect", "FAIL", str(exc))
        return finish(1)
    pulses, core = [], ActuatorCore(hub.dev, clock=env.clock)
    core.on_blank = lambda start, end: pulses.append(start + haptics.BLANK_LEAD_NS)     # the write time
    rings, ratings = [], {}
    try:
        for recipe in recipes:
            last_ms = [s for s in PATTERNS[recipe].steps if s.kind == "motors"][-1].ms
            prompt(f"Recipe '{recipe}': hold the hub in your fist as you will play. {repeats} pulses follow, "
                   "each after a quiet second. Press Enter.")
            ring_here, peaks_g, felt = [], [], False
            for _ in range(repeats):
                collected = []
                while not hub.imu.empty():
                    hub.imu.get_nowait()
                pulses.clear()
                _sleep(env, hub, collected, 1.0)                       # the quiet baseline
                core.submit(recipe)
                while core.process(env.clock.now_ns()) is not None:
                    _sleep(env, hub, collected, 0.01)
                _sleep(env, hub, collected, 1.6)                       # the hub rings, then settles
                if not pulses:
                    continue
                try:
                    r = benchstats.pulse_response(collected, pulses[-1], last_ms)
                except ValueError:
                    continue
                peaks_g.append(r["peak_gyro"])
                if r["felt_by_imu"]:
                    felt = True
                    ring_here.append(r["ringdown_s"])
            rings += ring_here
            ratings[recipe] = rate(recipe)
            detail = (f"you rated it {ratings[recipe]:.1f}/5; the hub's gyro moved by up to "
                      f"{max(peaks_g, default=0.0):.0f} dps" + (f" and rang for {max(ring_here) * 1000:.0f} ms"
                                                              if ring_here else "; its IMU did not notice it"))
            add(f"haptic_{recipe}", "PASS" if ratings[recipe] >= FELT else "WARN", detail)
    finally:
        core.close()
        hub.close()
    blank = benchstats.blank_after_pulse_s(rings)
    if blank is None:
        add("haptic_blanking", "WARN", f"the IMU did not feel any pulse (fist too loose?), so BLANK_AFTER_PULSE_S "
            f"stays {config.BLANK_AFTER_PULSE_S:.2f} s")
    else:
        config.write_local({"BLANK_AFTER_PULSE_S": round(blank, 3)}, config_path)
        add("haptic_blanking", "PASS", f"longest ring-down {max(rings) * 1000:.0f} ms: BLANK_AFTER_PULSE_S="
            f"{blank:.3f} written")
    best = max(ratings, key=ratings.get) if ratings else None
    if best is not None and ratings[best] >= FELT:
        add("haptic_summary", "PASS", f"felt well enough: best recipe is '{best}' ({ratings[best]:.1f}/5)")
    else:
        add("haptic_summary", "WARN", "nothing felt at 3/5: the game still works on beep + light, and the write-up "
            "says so. To feel more, tape coins or nuts to the shaft ends (axle-stopped, taped well), or strap the "
            "hub to your forearm, then re-run")
    return finish(0)


def _selftest():
    import math

    from pingpong.sources_fake import FakeEnv

    S = 1_000_000_000
    env, pulses = FakeEnv(hz=66.0), []
    dev = env.hub_device
    original = dev.motor_run_for_time

    def motor_run_for_time(time_ms, **kw):
        original(time_ms, **kw)
        pulses.append((env.clock.now_ns(), time_ms))

    dev.motor_run_for_time = motor_run_for_time

    def imu(now):
        shake, flip = 0.0, 1 if (now // 15_000_000) % 2 else -1
        for t_write, ms in pulses:
            end = t_write + ms * 1_000_000
            if end <= now < end + 0.1 * S:
                shake = max(shake, 800.0 * math.exp(-(now - end) / S / 0.0333))
            elif t_write <= now < end:
                shake = 800.0
        return (0, 0, 1000 + round(flip * shake / 4), round(flip * shake), 0, 0)

    env.scenario = imu
    with tempfile.TemporaryDirectory() as tmp:
        _, code = run(env, card={}, repeats=2, config_path=Path(tmp) / "c.json", report_path=Path(tmp) / "r.json",
                      prompt=lambda text: None, rate=lambda recipe: 4.0, out=lambda *_: None)
        blank = json.loads((Path(tmp) / "c.json").read_text())["BLANK_AFTER_PULSE_S"]
    assert code == 0 and 0.08 < blank < 0.22, (code, blank)
    print(f"bench_haptics selftest OK: ring-down measured, BLANK_AFTER_PULSE_S={blank:.3f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=config.CARD_COLOR)
    ap.add_argument("--card-serial", default=config.CARD_SERIAL)
    ap.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    from pingpong import hostcheck, live
    from pingpong.realenv import RealEnv

    try:
        card = live.require_card(args)
        recipes = motor_recipes(tuple(r.strip() for r in args.recipes.split(",") if r.strip()))
    except (live.LiveSetupError, ValueError) as exc:
        print(f"cannot run the haptics bench: {exc}", file=sys.stderr)
        return 2
    hostcheck.require_host("Bluetooth")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    _, code = run(RealEnv(), card=card, recipes=recipes, repeats=args.repeats, config_path=config.LOCAL_PATH,
                  report_path=config.HERE / "recordings" / "bench_haptics.json",
                  prompt=lambda text: input(text + "\n> "),
                  rate=lambda recipe: float(input(f"How strongly did you feel '{recipe}'? 1 (nothing) to 5 (strong)\n> ") or 1),
                  out=print)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
