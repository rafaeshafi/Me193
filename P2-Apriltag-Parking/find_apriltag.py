"""Find AprilTags in a live video stream and show where they are.

Usage:
    python3 scripts/find_apriltag.py [source] [--family FAMILY] [--id ID]

<source> is a camera index (default 0), a video file, or a stream URL
(rtsp://..., http://...).  By default every AprilTag family that
make_apriltag.py can produce is searched; --family 36h11 (etc.) restricts it
to one, which is faster.  --id limits the search to one tag.

Each detected tag is outlined in the window with its id and center, and
printed to the terminal as:  frame, family, id, center (x, y), angle (deg).
Press d to toggle debug view (red = square candidates that did not decode
as a tag).  Press q or Esc to quit.

Requires opencv-python (cv2.aruco); no other dependencies.
"""

import math
import sys

import cv2
from cv2 import aruco

from make_apriltag import FAMILIES


def make_detectors(families):
    params = aruco.DetectorParameters()
    # Also accept white-on-black tags (e.g. shown on a dark-mode screen).
    params.detectInvertedMarker = True
    # Sub-pixel corners give steadier center/angle readings.
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return {fam: aruco.ArucoDetector(aruco.getPredefinedDictionary(FAMILIES[fam]), params)
            for fam in families}


def find_tags(detectors, frame):
    """Return ([(family, id, corners 4x2, center (x, y), angle_deg), ...],
    rejected candidate corners) for one frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags, rejected = [], []
    for fam, detector in detectors.items():
        corners, ids, rej = detector.detectMarkers(gray)
        rejected = rej  # same candidates for every family; keep one set
        if ids is None:
            continue
        for tag_id, c in zip(ids.flatten(), corners):
            pts = c.reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            # Angle of the tag's top edge (corner 0 -> corner 1).
            dx, dy = pts[1] - pts[0]
            tags.append((fam, int(tag_id), pts, (float(cx), float(cy)),
                         math.degrees(math.atan2(dy, dx))))
    return tags, rejected


def draw_tags(frame, tags):
    for fam, tag_id, pts, (cx, cy), angle in tags:
        cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
        cv2.circle(frame, tuple(pts[0].astype(int)), 5, (0, 0, 255), -1)  # top-left
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"{fam} id {tag_id}  {angle:.0f} deg",
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    argv = sys.argv[1:]
    family, want_id = "all", None
    if "--family" in argv:
        i = argv.index("--family")
        family = argv[i + 1]
        del argv[i:i + 2]
    if "--id" in argv:
        i = argv.index("--id")
        want_id = int(argv[i + 1])
        del argv[i:i + 2]
    if family != "all" and family not in FAMILIES:
        raise SystemExit(f"Unknown family {family!r}; choose from {list(FAMILIES)} or all")
    families = list(FAMILIES) if family == "all" else [family]

    source = argv[0] if argv else "0"
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {source!r}")

    detectors = make_detectors(families)
    print(f"searching {', '.join(families)}  (d = debug view, q = quit)")

    frame_no, debug = 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of file or stream dropped
        frame_no += 1

        if frame_no == 30 and frame.max() < 10:
            print("warning: camera frames are black -- check the camera "
                  "permission for your terminal / VS Code in System Settings")

        tags, rejected = find_tags(detectors, frame)
        if want_id is not None:
            tags = [t for t in tags if t[1] == want_id]
        for fam, tag_id, _, (cx, cy), angle in tags:
            print(f"frame {frame_no}: {fam} id {tag_id} at ({cx:.0f}, {cy:.0f}), {angle:.0f} deg")

        if debug:
            aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))
        draw_tags(frame, tags)
        cv2.putText(frame, f"{len(tags)} tag(s)  [{family}]" + ("  DEBUG" if debug else ""), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("AprilTag finder (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("d"):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
