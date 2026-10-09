"""Frames drawn on a thread of their own, while the main thread sits in the window's wait.

On macOS the window system makes cv2.waitKey(1) take about 15 ms (measured: the same with nothing at all to draw, and the same for a
small window), and OpenCV lets go of the interpreter lock while it waits.  Drawing the next picture in that time instead of before
it makes the loop as quick as the wait alone: 25 ms a frame became 15 with the same drawing.  Only the drawing moves: the game, the
keys and the state the picture is made from stay on the main thread, and what is handed over is a finished, unshared picture.
"""

import threading


class FramePipe:
    def __init__(self, draw):
        """draw(*args) -> a picture; it runs on the pipe's thread, so it must only read what it is given."""
        self._draw = draw
        self._cv = threading.Condition()
        self._pending = None                  # the arguments of the picture to draw next (the newest submission wins)
        self._busy = False                    # a picture is being drawn
        self._done = None                     # (picture, error) of the newest finished one, until it is handed over
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="frames", daemon=True)
        self._thread.start()

    def submit(self, *args):
        """Draw this next, at once.  A submission still waiting when a newer one comes is dropped: nobody wants an old picture."""
        with self._cv:
            if not self._closed:
                self._pending = args
                self._cv.notify_all()

    def wait(self, timeout_s):
        """The newest picture, once it is drawn (waiting up to timeout_s for the one in the making); None if there is nothing new.
        An error raised while drawing is raised here, once."""
        with self._cv:
            self._cv.wait_for(lambda: not self._busy and self._pending is None, timeout_s)
            done, self._done = self._done, None
        if done is None:
            return None
        picture, error = done
        if error is not None:
            raise error
        return picture

    def close(self):
        with self._cv:
            self._closed, self._pending = True, None
            self._cv.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                args, self._pending, self._busy = self._pending, None, True
            try:
                result = (self._draw(*args), None)
            except Exception as exc:
                result = (None, exc)
            with self._cv:
                self._done, self._busy = result, False
                self._cv.notify_all()
