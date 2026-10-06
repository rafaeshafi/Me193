"""Generate an AprilTag marker image using OpenCV's aruco module.

Usage:
    python3 scripts/make_apriltag.py [id] [size_px] [output.png] [--family FAMILY]

Defaults: id=0, size_px=200, output=apriltag.png, family=36h11 (AprilTag's own
default -- best inter-tag distance of the standard families, and what
apriltag-ros / most robotics stacks expect).

Requires opencv-python (cv2.aruco); no other dependencies.
"""

import sys

import cv2
import numpy as np
from cv2 import aruco

FAMILIES = {
    "16h5": aruco.DICT_APRILTAG_16h5,
    "25h9": aruco.DICT_APRILTAG_25h9,
    "36h10": aruco.DICT_APRILTAG_36h10,
    "36h11": aruco.DICT_APRILTAG_36h11,
}


def make_tag(tag_id, size_px, family="36h11", quiet_zone=True):
    """Return a grayscale marker image (numpy array), with a white margin.

    generateImageMarker() draws only the tag's own black border, which
    butts straight up against the image edge.  Detectors need a white
    "quiet zone" around that border to find the marker's outer boundary --
    without it, detectMarkers() finds nothing, even though the tag itself
    is correct.  The margin is sized like AprilTag's own printable tags:
    one border-width of white on each side.
    """
    if family not in FAMILIES:
        raise SystemExit(f"Unknown family {family!r}; choose from {list(FAMILIES)}")
    dictionary = aruco.getPredefinedDictionary(FAMILIES[family])
    if tag_id >= dictionary.bytesList.shape[0]:
        raise SystemExit(
            f"id {tag_id} out of range for {family} "
            f"(0-{dictionary.bytesList.shape[0] - 1})"
        )
    tag = aruco.generateImageMarker(dictionary, tag_id, size_px)
    if not quiet_zone:
        return tag

    pad = max(size_px // 5, 1)  # one border-cell width, roughly
    padded = np.full((size_px + 2 * pad, size_px + 2 * pad), 255, dtype=np.uint8)
    padded[pad : pad + size_px, pad : pad + size_px] = tag
    return padded


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    family = "36h11"
    if "--family" in sys.argv:
        family = sys.argv[sys.argv.index("--family") + 1]
    quiet_zone = "--no-quiet-zone" not in sys.argv

    tag_id = int(args[0]) if len(args) > 0 else 0
    size_px = int(args[1]) if len(args) > 1 else 200
    out = args[2] if len(args) > 2 else "apriltag.png"

    img = make_tag(tag_id, size_px, family, quiet_zone)
    cv2.imwrite(out, img)
    print(f"saved: {out}  ({img.shape[1]}x{img.shape[0]}, family {family}, id {tag_id})")

    dictionary = aruco.getPredefinedDictionary(FAMILIES[family])
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(img)
    if ids is not None and tag_id in ids.flatten():
        print(f"self-check: detector reads it back as id {tag_id} -- OK")
    else:
        print("self-check: WARNING -- detector could not read this image back "
              "(try without --no-quiet-zone, or a larger size_px)")


if __name__ == "__main__":
    main()
