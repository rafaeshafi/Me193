"""Print-ready AprilTag cards for the ping-pong game (36h11, 15 cm tag).

Usage:
    ./pp make_cards [id ...] [--out docs/cards]     # default ids 0 1 2 3
    ./pp make_cards --selftest

Card ids: 0 = START, 1 = ROOKIE, 2 = CLUB, 3 = PRO.

Each card is one US Letter page at 300 dpi: the tag's black border is exactly
15 cm wide (8 cells x 221 px), surrounded by a white quiet zone of at least
2 cm (detectors find nothing without it), plus a printed 15 cm ruler bar so you
can confirm "Actual size" with a ruler (15 cm +-3 mm).  Print the PDF at
100% / Actual size -- NOT "fit to page" -- on matte paper and mount it on foam
board.  P2's make_tag() pads by size//5, which would print a 21 cm card, so
this tool does its own layout.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from cv2 import aruco  # noqa: E402

DPI = 300
PX_PER_CM = DPI / 2.54
PAGE_W, PAGE_H = 2550, 3300          # US Letter at 300 dpi
CELL_PX = 221                        # 36h11 = 8 x 8 cells including the black border
TAG_PX = 8 * CELL_PX                 # 1768 px = 14.97 cm
MARGIN_PX = round(2.0 * PX_PER_CM)   # 2 cm white quiet zone around the tag
RULER_CM = 15.0
RULER_PX = round(RULER_CM * PX_PER_CM)
FAMILY = aruco.DICT_APRILTAG_36h11
N_IDS = 587

LABELS = {
    0: "START  (id 0)",
    1: "ROOKIE  (id 1)",
    2: "CLUB  (id 2)",
    3: "PRO  (id 3)",
}

_TAG_TOP = 300                       # px from the top of the page to the tag's top edge


def make_card_page(tag_id):
    """Return the whole card as a grayscale Letter-size array."""
    if not 0 <= tag_id < N_IDS:
        raise ValueError(f"id {tag_id} out of range for 36h11 (0-{N_IDS - 1})")
    dictionary = aruco.getPredefinedDictionary(FAMILY)
    tag = aruco.generateImageMarker(dictionary, tag_id, TAG_PX)

    page = np.full((PAGE_H, PAGE_W), 255, dtype=np.uint8)
    x0 = (PAGE_W - TAG_PX) // 2
    page[_TAG_TOP:_TAG_TOP + TAG_PX, x0:x0 + TAG_PX] = tag

    # Everything below stays at least the quiet-zone width away from the tag.
    y = _TAG_TOP + TAG_PX + MARGIN_PX + 60
    label = LABELS.get(tag_id, f"id {tag_id}")
    cv2.putText(page, label, (x0, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 3.2, 0, 8, cv2.LINE_AA)

    y += 230
    rx0 = (PAGE_W - RULER_PX) // 2
    cv2.line(page, (rx0, y), (rx0 + RULER_PX, y), 0, 6)
    for rx in (rx0, rx0 + RULER_PX):
        cv2.line(page, (rx, y - 28), (rx, y + 28), 0, 6)
    cv2.putText(page, f"{RULER_CM:.0f} cm - measure this bar: print at ACTUAL SIZE (100%)",
                (rx0, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    cv2.putText(page, "36h11 AprilTag, matte paper, mount on foam board",
                (rx0, y + 165), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 90, 2, cv2.LINE_AA)
    return page


def save_card(tag_id, out_dir):
    """Write card<id>.png and card<id>.pdf (both 300 dpi); return their paths."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(make_card_page(tag_id))
    png = out_dir / f"card{tag_id}.png"
    pdf = out_dir / f"card{tag_id}.pdf"
    img.save(png, dpi=(DPI, DPI))
    img.convert("RGB").save(pdf, "PDF", resolution=float(DPI))
    return png, pdf


def decodes_as(page, tag_id):
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(FAMILY), aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(page)
    return ids is not None and ids.flatten().tolist() == [tag_id]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        with tempfile.TemporaryDirectory() as tmp:
            for tag_id in range(4):
                png, pdf = save_card(tag_id, tmp)
                assert decodes_as(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), tag_id), tag_id
                assert pdf.read_bytes().startswith(b"%PDF")
        print("make_cards selftest OK")
        return 0

    out = Path("docs/cards")
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    ids = [int(a) for a in argv] or [0, 1, 2, 3]
    for tag_id in ids:
        png, pdf = save_card(tag_id, out)
        ok = decodes_as(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), tag_id)
        print(f"{LABELS.get(tag_id, tag_id)}: {pdf}  (self-check {'OK' if ok else 'FAILED'})")
        if not ok:
            return 1
    print("Print each PDF at ACTUAL SIZE (100%), then check the ruler bar reads 15 cm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
