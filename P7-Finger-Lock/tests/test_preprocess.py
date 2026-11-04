import numpy as np

from lock import preprocess as pp


def test_roi_box_is_square_inside_frame():
    x0, y0, x1, y1 = pp.roi_box((480, 640, 3), [0.3, 0.2, 0.55])
    assert x1 - x0 == y1 - y0 == round(0.55 * 480)
    assert 0 <= x0 and x1 <= 640 and 0 <= y0 and y1 <= 480


def test_roi_box_clamped_to_frame():
    x0, y0, x1, y1 = pp.roi_box((480, 640), [0.9, 0.9, 0.5])
    assert x1 == 640 and y1 == 480


def test_to_input_shape_and_range_cv2_and_pil():
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = pp.crop(frame, [0.3, 0.2, 0.55])
    a = pp.to_input(crop, 64)
    b = pp.to_input(crop, 64, use_cv2=False)
    for x in (a, b):
        assert x.shape == (64, 64) and x.dtype == np.float32
        assert 0 <= x.min() and x.max() <= 1
    assert abs(float(a.mean()) - float(b.mean())) < 0.02


def test_gray_of_gray_is_identity():
    g = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert (pp.to_gray(g) == g).all()
