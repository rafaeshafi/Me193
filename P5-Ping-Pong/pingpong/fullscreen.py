"""The game's window: full screen on launch, with the picture shown in the window's own shape.

OpenCV's macOS window keeps a picture's shape but scales it to the window's width and sits it at the bottom, so a 16:9 picture on a
16:10 screen (a laptop's, with the menu bar kept) leaves a band across the top in the window's own colour.  The picture is therefore
padded with black to the shape of the window before it is shown, and the window is asked now and then how big it is (OpenCV cannot
say: its own answers are the picture's size).  The mouse callback reports positions in the pixels of what was shown, so the padding
is taken off again for the mouse that stands in for the hand in --fake.
"""

import ctypes
import platform
import sys
import time

import numpy as np

ASK_EVERY_S = 1.0               # how often the window is asked how big it is (it changes when the player leaves full screen)
SAME_SHAPE = 0.003              # a window this close to the picture's own shape (width / height) is shown the picture as it is


def pad_to(picture, aspect):
    """The picture centred on black in a frame of this shape (width / height), its own pixels untouched: -> (frame, (left, top))."""
    h, w = picture.shape[:2]
    if aspect <= 0 or abs(aspect - w / h) < SAME_SHAPE:
        return picture, (0, 0)
    frame_w, frame_h = (round(h * aspect), h) if aspect > w / h else (w, round(w / aspect))
    left, top = (frame_w - w) // 2, (frame_h - h) // 2
    frame = np.zeros((frame_h, frame_w) + picture.shape[2:], dtype=picture.dtype)
    frame[top:top + h, left:left + w] = picture
    return frame, (left, top)


class _CGRect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("width", ctypes.c_double), ("height", ctypes.c_double)]


def _ask_cocoa(title):
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    appkit = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
    objc.sel_registerName.restype, objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]

    def ask(receiver, selector, *args, returns=ctypes.c_void_p):
        send = objc.objc_msgSend
        send.restype, send.argtypes = returns, [ctypes.c_void_p, ctypes.c_void_p] + [type(a) for a in args]
        return send(receiver, objc.sel_registerName(selector), *args)

    app = ctypes.c_void_p.in_dll(appkit, "NSApp").value             # the application OpenCV made for its windows (None before it)
    if not app:
        return None
    windows = ask(app, b"windows")
    for i in range(ask(windows, b"count", returns=ctypes.c_ulong)):
        window = ask(windows, b"objectAtIndex:", ctypes.c_ulong(i))
        if ask(ask(window, b"title"), b"UTF8String", returns=ctypes.c_char_p) == title.encode():
            rect = ask(ask(window, b"contentView"), b"bounds", returns=_CGRect)
            return (rect.width, rect.height) if rect.width > 0 and rect.height > 0 else None
    return None


def view_size(title):
    """-> (width, height), in points, of what the window of this title shows its picture in; None when that cannot be known (not a Mac
    with Apple's chip, no such window, anything unexpected).  It only reads."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    try:
        return _ask_cocoa(title)
    except Exception:
        return None


_ask_the_system = view_size


class Window:
    def __init__(self, title, *, size, fullscreen=True, cv2=None, view_size=None, now=time.monotonic):
        """size: the picture's (width, height).  cv2, view_size and now are there to be replaced in a test."""
        if cv2 is None:
            import cv2
        self.title, self.size, self.fullscreen = title, size, fullscreen
        self._cv2, self._view_size, self._now = cv2, view_size or _ask_the_system, now
        self._asked_at = self._aspect = None
        self._offset = (0, 0)                                          # where the picture sits in what was shown last

    def open(self):
        if not self.fullscreen:
            self._cv2.namedWindow(self.title)                          # the window the game always had: the size of the picture
            return
        self._cv2.namedWindow(self.title, self._cv2.WINDOW_NORMAL)
        self._cv2.setWindowProperty(self.title, self._cv2.WND_PROP_FULLSCREEN, self._cv2.WINDOW_FULLSCREEN)
        self._cv2.imshow(self.title, np.zeros((self.size[1], self.size[0], 3), np.uint8))       # black while the camera and the hub connect
        self._cv2.waitKey(1)

    def show(self, picture):
        if self.fullscreen:
            aspect = self._window_aspect()
            picture, self._offset = pad_to(picture, aspect) if aspect else (picture, (0, 0))
        self._cv2.imshow(self.title, picture)

    def picture_xy(self, x, y):
        """A position over what was shown (the mouse callback's) as a position in the picture."""
        return x - self._offset[0], y - self._offset[1]

    def _window_aspect(self):
        now = self._now()
        if self._asked_at is None or now - self._asked_at >= ASK_EVERY_S:
            self._asked_at = now
            size = self._view_size(self.title)
            self._aspect = size[0] / size[1] if size else None
        return self._aspect
