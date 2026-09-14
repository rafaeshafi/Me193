"""Record examples of yourself holding each pose, to train a classifier on.

This is the data-collection half of training. You hold a pose, it records what
your body looked like, and every sample is one labelled row in a CSV. Nothing
is learned here — see train_poses.py for that.

    python collect_poses.py

Press the number key for a class, get into the pose during the countdown, hold
still-ish while it records, then press any key to stop. Repeat for every class.
Samples append to pose_data.csv, so you can collect over several sittings.
"""

import argparse
import csv
import time
from collections import Counter

import cv2

import pose_features as pf

COUNTDOWN_SECONDS = 3


def existing_counts(path):
    """Samples already on disk, so a second session adds to the first."""
    if not path.exists():
        return Counter()
    with path.open(newline="") as handle:
        return Counter(row["label"] for row in csv.DictReader(handle))


def append_samples(path, rows):
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["label", *pf.FEATURE_NAMES])
        writer.writerows(rows)


def draw_panel(frame, classes, counts, recording, countdown, session_count):
    height = frame.shape[0]
    for slot, name in enumerate(classes):
        total = counts[name]
        active = recording == name
        colour = (0, 220, 0) if active else (255, 255, 255) if total else (120, 120, 120)
        cv2.putText(frame, f"[{slot + 1}] {name:<9} {total:>4}", (20, 40 + slot * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2 if active else 1)

    if countdown is not None:
        text = str(countdown) if countdown > 0 else "GO"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
        origin = ((frame.shape[1] - size[0]) // 2, (height + size[1]) // 2)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 220, 255), 8)
        cv2.putText(frame, "get into the pose", (origin[0] - 90, origin[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    elif recording:
        cv2.circle(frame, (frame.shape[1] - 40, 40), 12, (0, 0, 255), -1)
        cv2.putText(frame, f"REC {recording}  +{session_count}",
                    (frame.shape[1] - 260, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "press any key to stop", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "press a number to record that pose  |  q to finish",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", default=",".join(pf.DEFAULT_CLASSES),
                        help="comma-separated pose class names")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--out", type=lambda p: pf.HERE / p, default=pf.DATA_PATH,
                        help="CSV to append samples to")
    args = parser.parse_args()

    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not 1 < len(classes) <= 9:
        print("Need between 2 and 9 classes (one per number key).")
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    counts = existing_counts(args.out)
    if counts:
        print("Already recorded:", dict(counts))

    pending = []            # samples this session, written out at the end
    recording = None        # class currently being recorded
    countdown_ends = None   # when the countdown finishes
    session_count = 0
    start = time.monotonic()

    print("Keys:", ", ".join(f"{i + 1}={name}" for i, name in enumerate(classes)), "| q=finish")
    try:
        with pf.make_landmarker() as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera stopped delivering frames.")
                    break

                # Mirror to match pose_car.py. Mirroring flips which side of the
                # frame each arm appears on, so training and driving MUST agree.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                now = time.monotonic()
                result = landmarker.detect_for_video(pf.to_mp_image(frame),
                                                     int((now - start) * 1000))

                pts = None
                features = None
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    geometry = pf.pose_geometry(landmarks, width, height)
                    if geometry is not None:
                        pts = geometry[0]
                        features = pf.features_from_landmarks(landmarks, width, height)

                countdown = None
                if countdown_ends is not None:
                    remaining = countdown_ends - now
                    if remaining <= 0:
                        countdown_ends = None
                    else:
                        countdown = int(remaining) + 1

                # Only record once the countdown is done and the pose is readable.
                if recording and countdown_ends is None and features is not None:
                    pending.append([recording, *features])
                    counts[recording] += 1
                    session_count += 1

                pf.draw_upper_body(frame, pts)
                if features is None:
                    cv2.putText(frame, "no usable pose", (20, height - 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
                draw_panel(frame, classes, counts, recording, countdown, session_count)
                cv2.imshow("collect poses", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue
                if recording:
                    # Any key stops the take.
                    print(f"  {recording}: +{session_count} samples ({counts[recording]} total)")
                    recording, session_count = None, 0
                    continue
                if key in (ord("q"), 27):
                    break
                if ord("1") <= key <= ord("9"):
                    slot = key - ord("1")
                    if slot < len(classes):
                        recording = classes[slot]
                        countdown_ends = now + COUNTDOWN_SECONDS
                        session_count = 0
                        print(f"recording '{recording}' in {COUNTDOWN_SECONDS}s...")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if pending:
        append_samples(args.out, pending)
        print(f"\nWrote {len(pending)} samples to {args.out.name}")
    else:
        print("\nNo samples recorded.")

    print("Totals:", dict(counts) or "(none)")
    thin = [name for name in classes if counts[name] < 50]
    if thin:
        print(f"Thin classes (under 50 samples): {', '.join(thin)} — record more of these.")
    else:
        print(f"Now train on it:  my_env/bin/python train_poses.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
