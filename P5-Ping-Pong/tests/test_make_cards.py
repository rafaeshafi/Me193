"""Printable AprilTag cards: physical size, margin and decodability.

The plan prints 36h11 tags 15 cm wide with a 2 cm white margin on US Letter,
at "Actual size".  These tests pin the geometry in pixels at 300 dpi so a
printed card can be checked with a ruler (15 cm +-3 mm).
"""

import cv2
import numpy as np
import pytest
from cv2 import aruco

from tools import make_cards

DPI = 300
PX_PER_CM = DPI / 2.54


def _detect(img):
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11),
        aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(img)
    return corners, ids


def test_page_is_us_letter_at_300_dpi():
    page = make_cards.make_card_page(0)
    assert page.shape == (3300, 2550)
    assert page.dtype == np.uint8


@pytest.mark.parametrize("tag_id", [0, 1, 2, 3])
def test_tag_decodes_as_requested_id(tag_id):
    page = make_cards.make_card_page(tag_id)
    _, ids = _detect(page)
    assert ids is not None
    assert ids.flatten().tolist() == [tag_id]


def test_tag_edge_is_15_cm_within_one_millimetre():
    page = make_cards.make_card_page(1)
    corners, ids = _detect(page)
    pts = corners[0].reshape(4, 2)
    edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    for edge_px in edges:
        edge_cm = edge_px / PX_PER_CM
        assert abs(edge_cm - 15.0) <= 0.1, f"edge {edge_cm:.3f} cm"


def test_white_margin_is_at_least_2_cm_on_every_side_of_the_tag():
    page = make_cards.make_card_page(2)
    corners, _ = _detect(page)
    pts = corners[0].reshape(4, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    margin_px = 2.0 * PX_PER_CM - 4  # 2 cm minus corner-detection slack
    # the square around the tag, grown by the margin, must be pure white
    xa, ya = int(x0 - margin_px), int(y0 - margin_px)
    xb, yb = int(x1 + margin_px), int(y1 + margin_px)
    ring = page[ya:yb, xa:xb].copy()
    ring[int(y0 - ya) - 2 : int(y1 - ya) + 3, int(x0 - xa) - 2 : int(x1 - xa) + 3] = 255
    assert ring.min() == 255


def test_ruler_bar_is_exactly_15_cm_long():
    # A printed scale bar lets you verify "Actual size" without a CAD tool.
    assert make_cards.RULER_CM == 15.0
    assert abs(make_cards.RULER_PX - 15.0 * PX_PER_CM) <= 1.0


def test_save_card_writes_png_and_pdf_with_300_dpi(tmp_path):
    from PIL import Image

    png, pdf = make_cards.save_card(3, tmp_path)
    assert png.exists() and pdf.exists()
    with Image.open(png) as im:
        assert im.size == (2550, 3300)
        assert round(im.info["dpi"][0]) == DPI
    assert pdf.read_bytes().startswith(b"%PDF")


def test_labels_cover_start_and_three_levels():
    assert make_cards.LABELS[0].startswith("START")
    assert [make_cards.LABELS[i].split()[0] for i in (1, 2, 3)] == ["ROOKIE", "CLUB", "PRO"]


def test_out_of_range_id_is_rejected():
    with pytest.raises(ValueError):
        make_cards.make_card_page(587)  # 36h11 has ids 0..586
