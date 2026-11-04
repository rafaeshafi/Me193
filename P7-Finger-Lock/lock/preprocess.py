"""From a camera frame to the net's input: crop the square region of interest, grayscale, resize, scale to 0..1.

The ROI is [x, y, size] as fractions of the frame, so the same config.json crops the same desk square on the Mac and
on the board. OpenCV does the resize when it is installed (both machines); PIL otherwise.
"""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the board and the Mac both have it; tests cover the PIL path explicitly
    cv2 = None

SAVE_SIZE = 128  # the crops on disk are kept bigger than the net's input so the input size can change later


def roi_box(frame_shape, roi):
    """Pixel box (x0, y0, x1, y1) of the ROI in a frame of shape (h, w[, c]); always square and inside the frame."""
    h, w = frame_shape[:2]
    x, y, s = roi
    side = int(round(s * min(w, h)))
    x0, y0 = int(round(x * w)), int(round(y * h))
    x0, y0 = min(x0, w - side), min(y0, h - side)
    return x0, y0, x0 + side, y0 + side


def crop(frame, roi):
    x0, y0, x1, y1 = roi_box(frame.shape, roi)
    return frame[y0:y1, x0:x1]


def to_gray(img):
    """uint8 (h, w) from uint8 (h, w) or (h, w, 3) BGR (OpenCV's order; the camera gives BGR on both machines)."""
    if img.ndim == 2:
        return img
    b, g, r = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    return np.clip(0.114 * b + 0.587 * g + 0.299 * r + 0.5, 0, 255).astype(np.uint8)


def resize(img, size, use_cv2=True):
    if img.shape[0] == size and img.shape[1] == size:
        return img
    if cv2 is not None and use_cv2:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def to_input(crop_img, size=64, use_cv2=True):
    """(size, size) float32 in 0..1 from a crop (uint8 gray or BGR)."""
    g = resize(to_gray(crop_img), size, use_cv2)
    return g.astype(np.float32) / 255.0


def standardise(x, mean, std):
    return (x - float(mean)) / float(std)


def frame_to_input(frame, roi, size=64):
    return to_input(crop(frame, roi), size)
