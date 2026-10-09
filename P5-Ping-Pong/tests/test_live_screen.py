"""What the screen is given by the live rig: the camera's picture in the corner, and how steady the pictures were."""

import numpy as np
import pytest

from test_live import Rig


def test_the_display_frame_is_the_mirror_image_of_the_camera_frame():
    r = Rig()
    assert r.rig.display_frame() is None
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 0] = 255                                        # a bright left edge
    r.vision.frame = frame
    shown = r.rig.display_frame()
    assert shown[0, 5, 0] == 255 and shown[0, 0, 0] == 0     # ... appears on the right: like a mirror


def test_the_display_frame_comes_from_the_small_picture_when_the_camera_offers_one():
    r = Rig()
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = np.zeros((360, 640, 3), dtype=np.uint8)
    small[:, 0] = 255
    r.vision.frame = big
    r.vision.latest_picture = lambda: small
    shown = r.rig.display_frame()
    assert shown.shape == (360, 640, 3) and shown[0, 639, 0] == 255      # the small one, mirrored: not a 1080p frame to flip every turn


def test_the_time_between_the_pictures_shown_is_kept_for_the_report():
    r = Rig()
    assert "fps" not in r.rig.loop_stats()                         # nothing was shown yet (a replay never shows anything)
    for t in (0.000, 0.017, 0.033, 0.050, 0.067, 0.120, 0.137):     # one picture came late: 53 ms after the one before
        r.rig.note_picture(t)
    stats = r.rig.loop_stats()
    assert stats["pictures"] == 7
    assert stats["fps"] == pytest.approx(6 / 0.137, rel=0.01)
    assert stats["frame_p50_ms"] == pytest.approx(17.0, abs=0.5) and stats["frame_max_ms"] == pytest.approx(53.0, abs=0.5)
    assert stats["over_25ms"] == pytest.approx(1 / 6)
