"""Interactive image-processing pipeline: threshold, kernel, and morphology.

Usage:
    python3 scripts/threshold_slider.py <source> [output.png]

<source> is anything grayscale_first_image.py accepts: a local image file,
a direct image URL, or a page URL (the first <img> is used).

Pipeline (grayscale -> convolve -> threshold -> morphology), each stage
optional and adjustable live:
  - kernel     : blur / sharpen / edge-detect convolution (see morphology.py)
  - threshold  : slider 0-255, or "Auto (Otsu)"
  - morphology : erosion / dilation / opening / closing / boundary /
                 skeleton / skeleton+prune / corner detection
  - iterations : how many times the morphology op repeats

Press Save to write the full-resolution result.  Standard library plus
Pillow only.
"""

import sys
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from grayscale_first_image import first_image
from morphology import KERNELS, OPS, convolve, otsu, threshold as apply_threshold

PREVIEW_MAX = (900, 675)
DEFAULT_THRESHOLD = 128
MORPH_CHOICES = list(OPS.keys())
KERNEL_CHOICES = list(KERNELS.keys())


class ThresholdApp:
    def __init__(self, root, gray, out):
        self.gray = gray
        self.out = out

        self.preview = gray.copy()
        self.preview.thumbnail(PREVIEW_MAX, Image.LANCZOS)

        root.title(f"Image pipeline — {out}")

        self.canvas = tk.Label(root)
        self.canvas.pack(padx=8, pady=8)

        controls = tk.Frame(root)
        controls.pack(padx=8, pady=(0, 8), fill=tk.X)

        # --- kernel -----------------------------------------------------
        krow = tk.Frame(controls)
        krow.pack(fill=tk.X, pady=2)
        tk.Label(krow, text="kernel:", width=10, anchor="w").pack(side=tk.LEFT)
        self.kernel = tk.StringVar(value="none")
        ttk.Combobox(
            krow, textvariable=self.kernel, values=KERNEL_CHOICES,
            state="readonly", width=16,
        ).pack(side=tk.LEFT)
        self.kernel.trace_add("write", lambda *a: self.redraw())

        # --- threshold ----------------------------------------------------
        # Auto (Otsu) by default -- a fixed cutoff of 128 turns a dark photo
        # almost entirely black, which looks indistinguishable from a
        # broken window.
        self.auto_threshold = tk.BooleanVar(value=True)
        self.slider = tk.Scale(
            controls, from_=0, to=255, orient=tk.HORIZONTAL,
            length=self.preview.width, label="threshold",
            command=lambda v: self.redraw(),
        )
        self.slider.set(DEFAULT_THRESHOLD)
        self.slider.pack(fill=tk.X)
        tk.Checkbutton(
            controls, text="Auto (Otsu)", variable=self.auto_threshold,
            command=self.redraw,
        ).pack(anchor="w")

        # --- morphology ---------------------------------------------------
        mrow = tk.Frame(controls)
        mrow.pack(fill=tk.X, pady=2)
        tk.Label(mrow, text="morphology:", width=10, anchor="w").pack(side=tk.LEFT)
        self.morph = tk.StringVar(value="none")
        ttk.Combobox(
            mrow, textvariable=self.morph, values=MORPH_CHOICES,
            state="readonly", width=20,
        ).pack(side=tk.LEFT)
        self.morph.trace_add("write", lambda *a: self.redraw())

        self.reps = tk.Scale(
            controls, from_=1, to=10, orient=tk.HORIZONTAL,
            length=self.preview.width, label="iterations",
            command=lambda v: self.redraw(),
        )
        self.reps.set(1)
        self.reps.pack(fill=tk.X)

        # --- actions --------------------------------------------------
        row = tk.Frame(controls)
        row.pack(pady=6)
        tk.Button(row, text="Save", command=self.save).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=4)
        self.status = tk.Label(row, text="")
        self.status.pack(side=tk.LEFT, padx=8)

    def _pipeline(self, img):
        """Run kernel -> threshold -> morphology on a grayscale image."""
        step = convolve(img, self.kernel.get())

        cutoff = otsu(step) if self.auto_threshold.get() else self.slider.get()
        step = apply_threshold(step, cutoff)
        # Grey out the slider's readout when Otsu overrides it, but leave it
        # enabled -- a disabled tk.Scale silently ignores programmatic
        # .set() calls, which would strand it at a stale value.
        self.slider.configure(fg="#999" if self.auto_threshold.get() else "black")

        morph_name = self.morph.get()
        step = OPS[morph_name](step, self.reps.get())
        return step, cutoff

    def redraw(self):
        bw, cutoff = self._pipeline(self.preview)
        # Keep a reference on self or Tk garbage-collects the image.
        self.photo = ImageTk.PhotoImage(bw)
        self.canvas.configure(image=self.photo)
        mode = "auto" if self.auto_threshold.get() else "manual"
        self.status.configure(text=f"threshold {cutoff} ({mode})")

    def reset(self):
        self.kernel.set("none")
        self.morph.set("none")
        self.auto_threshold.set(False)
        self.slider.set(DEFAULT_THRESHOLD)
        self.reps.set(1)

    def save(self):
        # Morphology at full resolution (skeleton especially) can take real
        # time -- show something before the UI freezes, not after.
        self.status.configure(text="working…")
        self.canvas.configure(cursor="watch")
        self.canvas.master.update_idletasks()

        bw, cutoff = self._pipeline(self.gray)
        bw.save(self.out)

        self.canvas.configure(cursor="")
        self.status.configure(text=f"saved at threshold {cutoff} → {self.out}")
        print(f"saved: {self.out} (threshold {cutoff})")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = sys.argv[2] if len(sys.argv) > 2 else "processed.png"

    src, img = first_image(sys.argv[1])
    print(f"image: {src}")

    root = tk.Tk()
    app = ThresholdApp(root, img.convert("L"), out)
    app.redraw()
    root.mainloop()


if __name__ == "__main__":
    main()
