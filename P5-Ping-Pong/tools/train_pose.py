"""Train the pose model on your own recordings: a steadier filter, a prediction of where the hand is going, and the
pose model that reads you best.

Usage:
    cd ~/ME193/P5-Ping-Pong
    ./pp train_pose --player rafae --record     # from Terminal.app (the camera): a ~95 s guided take, then trains on it
    ./pp train_pose --player rafae              # train on the sessions and the take already recorded
    ./pp train_pose --player rafae --dry-run    # show what it would choose, save nothing
    ./pp train_pose --selftest                  # no hardware needed

The take is the one thing worth doing on purpose: stand where you play with the hub in your fist and follow the screen
(still, slide, lift, follow a ball, swing).  It is saved as a video with the time of every frame, so the SAME footage can be run
through each pose model (the light one and the full one) and the one whose readings are cleanest, while still keeping up with 30
frames a second, is chosen for you.  Every game you play also keeps its hand readings (before and after the filter); the filter
and the hand predictor are trained on those and on the take, and the predictor is only used if it clearly beats holding the hand
where it was.

The result is data/players/<name>/pose_model.json, which `./pp play --player <name>` loads by itself.  Training works from any
terminal; only --record needs the camera, which has to be Terminal.app (not the Claude app).
"""

import argparse
import datetime
import math
import shutil
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

import config  # noqa: E402
from pingpong import canvas, live, posemodel, posetake, posetrain, preview, profile, recorder  # noqa: E402
from pingpong.body import BodyTracker  # noqa: E402
from pingpong.pose_features import LANDMARKERS  # noqa: E402

TITLE = "P5 pose take  (Q to cancel)"
POSITION_HOLD_S = 2.0              # in position (head to hips in the picture) this long, and the script starts
AMBER, GREEN, GREY = (40, 170, 255), (80, 220, 80), (170, 170, 170)
ENTER_KEYS = (13, 10, 32)
QUIT_KEYS = (ord("q"), 27)
FRAME_S = 1 / 30
POSITION_TEXT = "Stand where you play with the hub in your fist; get your head, shoulders and hips in the picture"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", default="rafae")
    ap.add_argument("--hand", choices=("right", "left"), default=None, help="default: the hand in the player's calibration")
    ap.add_argument("--record", action="store_true", help="first record a guided take with the camera (~95 s, Terminal.app)")
    ap.add_argument("--dry-run", action="store_true", help="work out and show the pose model, but do not save it")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def player_setup(player, hand, player_root=None):
    """-> (hand, shoulder width) from the player's calibration (either swing source); --hand overrides the hand."""
    cal = profile.load(player, root=player_root) or profile.load(player, root=player_root, source="pose")
    return hand or (cal.hand if cal else "right"), (cal.shoulder_w if cal else None)


# --- recording a take -----------------------------------------------------------------------------------------------------------
def render(frame, landmarks, caption, headline, progress, width):
    """One window frame: the mirrored camera with the joints the game reads, what to do, and how far the take is."""
    image = preview.compose(frame, caption, landmarks, width=width)
    height = image.shape[0]
    canvas.draw_text(image, headline, (width // 2, 100), max(0.5, width / 700), AMBER, 2, anchor="center")
    cv2.rectangle(image, (20, height - 62), (width - 20, height - 48), GREY, 1)
    cv2.rectangle(image, (20, height - 62), (20 + int((width - 40) * progress), height - 48), GREEN, -1)
    return image


def _take_loop(env, vision, cap, show, wait_key, width):
    """-> True when the whole script was recorded, False when the player cancelled."""
    total = sum(step.seconds for step in posetake.SCRIPT)
    frame_ns = round(1e9 * FRAME_S)
    held_since = started = last_shown = None
    while True:
        if not env.threaded:
            vision.step()                                             # a synchronous environment has no camera thread
        now = env.clock.now_ns()
        in_position = not preview.missing_parts(vision.last_landmarks)
        key = 255
        if started is None:
            if not in_position:
                held_since = None
            elif held_since is None:
                held_since = now
            held = 0.0 if held_since is None else (now - held_since) / 1e9
            caption = POSITION_TEXT
            headline = "GET IN POSITION" if held_since is None else f"HOLD STILL  {math.ceil(POSITION_HOLD_S - held)}"
            progress = 0.0
            if held_since is not None and held >= POSITION_HOLD_S:
                started, cap.recording = now, True
        else:
            at = posetake.where((now - started) / 1e9)
            if at is None:
                cap.recording = False
                return True
            index, step, left = at
            caption, progress = step.text, (now - started) / 1e9 / total
            headline = f"STEP {index + 1}/{len(posetake.SCRIPT)}  {step.name.upper()}  {math.ceil(left)} s left"
        frame = vision.latest_frame()
        if frame is not None and (last_shown is None or now - last_shown >= frame_ns):
            show(render(frame, vision.last_landmarks, caption, headline, progress, width))
            last_shown, key = now, wait_key(1) & 0xFF
        if key in QUIT_KEYS:
            return False
        if key in ENTER_KEYS and started is None:                      # you are at the laptop: no need to wait for the hold
            started, cap.recording = now, True
        env.sleep(0.01 if env.threaded else FRAME_S)


def record_take(env, args, hand, ref_width, *, root, show, wait_key, out=print, width=preview.WIDTH):
    """The player follows the on-screen script while the camera is recorded -> the take's folder, or None (cancelled, or
    the camera or the disk would not do)."""
    from pingpong.vision import VisionWorker

    capture = env.open_camera(config.CAMERA_INDEX)
    if not capture.isOpened():
        out("cannot record: could not open the camera (allow Camera for this terminal in System Settings > Privacy & "
            "Security, quit apps using it, and turn Continuity Camera off on your iPhone)")
        return None
    live.request_720p(capture)
    directory = posetake.take_dir(args.player, root)
    cap = posetake.TakeCapture(capture, env.clock, directory, recording=False)
    if cap.failed:
        cap.release()
        shutil.rmtree(directory, ignore_errors=True)
        out(f"cannot record: could not write a video to {directory}")
        return None
    vision = VisionWorker(cap, env.make_landmarker(), clock=env.clock, hand=hand, body=BodyTracker(ref=ref_width),
                          to_image=getattr(env, "to_image", None))
    done = False
    try:
        if env.threaded:
            vision.start()
        done = _take_loop(env, vision, cap, show, wait_key, width)
    finally:
        vision.stop()                                                 # also finishes the video
        if not done or cap.failed:
            shutil.rmtree(directory, ignore_errors=True)
    if not done:
        return None
    if cap.failed:
        out(f"the take could not be written to {directory} (is the disk full?)")
        return None
    out(f"take saved: {directory}")
    return directory


# --- training -------------------------------------------------------------------------------------------------------------------
def _progress(out):
    """Print the analysis' progress every quarter."""
    last = [-1]

    def report(i, n):
        quarter = i * 4 // max(1, n) * 25
        if quarter != last[0]:
            last[0] = quarter
            out(f"    {quarter}%")

    return report


def compare_landmarkers(env, take, hand, ref_width, out):
    """Run each pose model over the take -> (the one to use, {name: its numbers}, why, its Run)."""
    runs = {}
    for name in LANDMARKERS:
        out(f"  running the {name} pose model over the take ...")
        landmarker = None
        try:
            landmarker = env.make_landmarker(model=name)
            runs[name] = posetake.analyze(take, landmarker, hand=hand, ref_width=ref_width,
                                          to_image=getattr(env, "to_image", None), progress=_progress(out))
        except Exception as exc:                  # a pose model that will not start must not stop the other one
            out(f"  the {name} pose model could not run: {exc}")
        finally:
            if landmarker is not None and hasattr(landmarker, "close"):
                landmarker.close()
    if not runs:
        return "lite", None, None, None
    for name, run in runs.items():
        out(f"    {name:<5} noise {run.noise:.3f} shoulder widths, {run.glitches:.1%} landmark flips, "
            f"found you in {run.detection:.0%} of frames, {run.infer_ms_p95:.0f} ms a frame")
    chosen, why = posetake.choose(runs)
    numbers = {name: {"noise": run.noise, "glitches": run.glitches, "detection": run.detection,
                      "infer_ms_p95": run.infer_ms_p95} for name, run in runs.items()}
    return chosen, numbers, why, runs[chosen]


def train(env, args, *, player_root=None, record_root=None, out=print):
    """Gather what is recorded, compare the pose models on the take, tune the filter, train the hand predictor, save the
    result -> 0 done, 2 nothing to train on."""
    hand, ref_width = player_setup(args.player, args.hand, player_root)
    root = record_root or recorder.default_root()
    tracks = posetrain.load_tracks(args.player, root=root)
    sessions, takes = len(tracks), posetake.list_takes(args.player, root=root)
    out(f"Pose training for {args.player} ({hand} hand): {sessions} recorded session{'s' if sessions != 1 else ''}"
        + (f" and the take {takes[-1].name}" if takes else ", no take"))
    landmarker, numbers, why, take = "lite", None, None, None
    if takes:
        landmarker, numbers, why, run = compare_landmarkers(env, takes[-1], hand, ref_width, out)
        take = takes[-1].name
        if run is not None and run.track is not None:
            tracks.append(run.track)
        if why:
            out(f"Pose model: {landmarker}: {why}")
    if not tracks:
        out("There is nothing to train on yet. Record a take (about 95 seconds, from Terminal.app):\n"
            f"    ./pp train_pose --player {args.player} --record\n"
            "or play a few games first: every game keeps the hand readings this learns from.")
        return 2
    params, filter_report = posemodel.FilterParams(), None
    if any(t.ru is not None for t in tracks):
        try:
            params, filter_report = posetrain.tune_filter(tracks)
        except ValueError as exc:
            out(f"Filter: kept as it was: {exc}")
        else:
            if params == posemodel.FilterParams():
                out("Filter: the settings stay as they were: none follows your hand better without jittering a still hand more")
            else:
                out(f"Filter: min_cutoff {params.min_cutoff:g}, beta {params.beta:g}, d_cutoff {params.d_cutoff:g}: it follows a "
                    f"zero-phase smoothing of your raw readings to {filter_report['rmse_tuned']:.3f} shoulder widths "
                    f"(the settings before: {filter_report['rmse_default']:.3f}), and a still hand jitters "
                    f"{filter_report['shimmer_tuned']:.4f} shoulder widths a frame (before: {filter_report['shimmer_default']:.4f})")
    else:
        out("Filter: kept as it was: no unfiltered readings recorded yet (record a take, or play a game)")
    if params != posemodel.FilterParams():
        kept = [posetrain.refilter(t, params) for t in tracks if t.ru is not None]
        older = len(tracks) - len(kept)
        if older:
            out(f"  {older} older session{'s' if older != 1 else ''} without unfiltered readings {'are' if older != 1 else 'is'} "
                "left out of the predictor's training: they were smoothed with the old settings")
        tracks = kept
    predictor, predictor_report = None, None
    try:
        candidate, report = posetrain.fit_predictor(tracks)
    except ValueError as exc:
        out(f"Hand prediction: not used: {exc}")
    else:
        cv = report["cv"]
        shipped, reason = posetrain.worth_shipping(cv)
        predictor = candidate if shipped else None
        predictor_report = {"shipped": shipped, "why": reason, "lambda": report["lambda"], "sessions": report["sessions"],
                            "cv": {k: float(v) for k, v in cv.items()}}
        out(f"Hand prediction ({cv['lead_s'] * 1000:.0f} ms ahead, on tracks it was not trained on): holding the hand where "
            f"it was is {cv['rmse_hold']:.3f} shoulder widths off, the model {cv['rmse_model']:.3f}: "
            + ("used: " if shipped else "not used: ") + reason)
    meta = {"trained": datetime.datetime.now().isoformat(timespec="seconds"), "hand": hand, "sessions": sessions,
            "take": take, "filter": filter_report, "predictor": predictor_report, "landmarkers": numbers,
            "landmarker_why": why}
    model = posemodel.PoseModel(filter=params, predictor=predictor, landmarker=landmarker, meta=meta)
    if args.dry_run:
        out("Result not saved (--dry-run).")
    else:
        path = posemodel.save_for(args.player, model, root=player_root)
        out(f"Saved {path}: ./pp play --player {args.player} uses it.")
    return 0


# --- the selftest -----------------------------------------------------------------------------------------------------------------
def _selftest():
    import random
    import tempfile
    from types import SimpleNamespace

    from pingpong.sources_fake import FakeEnv, FakeLandmarker, body_landmarks

    s = 1_000_000_000

    def hand_at(t):                                       # a hand that follows any script: slides, lifts and quick swings
        quick = 0.5 * math.sin(2 * math.pi * 1.3 * t) * (1 if int(t / 4) % 2 else 0.2)
        return 0.7 * math.sin(2 * math.pi * 0.45 * t) + quick, 0.3 * math.sin(2 * math.pi * 0.3 * t)

    class Noisy:                                          # a pose model that reads the same hand with some noise
        def __init__(self, sigma, flip_every=0, seed=1):
            self.sigma, self.flip_every, self.rnd, self.n = sigma, flip_every, random.Random(seed), 0

        def detect_for_video(self, image, ts_ms):
            u, v = hand_at(self.n / 30.0)
            u += self.rnd.gauss(0, self.sigma) + (1.5 if self.flip_every and self.n % self.flip_every == 7 else 0.0)
            v += self.rnd.gauss(0, self.sigma)
            self.n += 1
            return SimpleNamespace(pose_landmarks=[body_landmarks(u, v, full_body=True)])

    env = FakeEnv()
    t0 = env.clock.now_ns()
    env.make_landmarker = lambda model="lite": FakeLandmarker(lambda t_ns: hand_at((t_ns - t0) / s), env.clock, full_body=True)
    args = parse_args(["--player", "selftest", "--record"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        take = record_take(env, args, "right", None, root=tmp / "rec", show=lambda image: None, wait_key=lambda ms: 255,
                           out=lambda *_: None, width=160)
        assert take is not None, "the fake player's take was not recorded"
        total = sum(step.seconds for step in posetake.SCRIPT)
        times = posetake.VideoFrames(take).times
        assert abs((times[-1] - times[0]) / s - total) < 1.0, f"the take is not the script's {total:.0f} s long"
        env.make_landmarker = lambda model="lite": Noisy(0.05, flip_every=45) if model == "lite" else Noisy(0.012, seed=2)
        lines = []
        code = train(env, args, player_root=tmp / "players", record_root=tmp / "rec", out=lines.append)
        model = posemodel.load_for("selftest", root=tmp / "players")
        assert code == 0 and model is not None, "\n".join(lines)
        assert model.landmarker == "full", f"the steadier pose model was not chosen: {model.meta['landmarkers']}"
        assert model.filter != posemodel.FilterParams() or model.meta["filter"]["rmse_tuned"] <= model.meta["filter"]["rmse_default"]
        assert train(env, parse_args(["--player", "selftest", "--dry-run"]), player_root=tmp / "players2",
                     record_root=tmp / "rec", out=lambda *_: None) == 0 and posemodel.load_for("selftest", root=tmp / "players2") is None
    print(f"train_pose selftest OK: a {total:.0f} s take, chose the {model.landmarker} pose model "
          f"(noise {model.meta['landmarkers']['full']['noise']:.3f} against {model.meta['landmarkers']['lite']['noise']:.3f}), "
          f"filter {'tuned' if model.filter != posemodel.FilterParams() else 'kept'}, "
          f"hand prediction {'used' if model.predictor is not None else 'not used'}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return _selftest()
    from pingpong.realenv import RealEnv

    env = RealEnv()
    try:
        if not args.record:
            return train(env, args)
        from pingpong import hostcheck

        hostcheck.require_host("Camera")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        hand, ref_width = player_setup(args.player, args.hand)
        cv2.namedWindow(TITLE)
        try:
            take = record_take(env, args, hand, ref_width, root=recorder.default_root(),
                               show=lambda image: cv2.imshow(TITLE, image), wait_key=cv2.waitKey)
        finally:
            cv2.destroyAllWindows()
        return 1 if take is None else train(env, args)
    except ValueError as exc:                             # a damaged calibration file: the message says how to fix it
        print(f"cannot train: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
