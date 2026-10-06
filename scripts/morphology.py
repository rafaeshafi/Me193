"""Binary morphology and convolution kernels, built on Pillow only.

The pipeline these functions are meant to be used in:

    grayscale  ->  convolve (kernel)  ->  threshold  ->  morphology

Binary images here are mode "L" with only 0 (background) and 255
(foreground), which is what PIL.ImageMorph expects.
"""

from PIL import Image, ImageChops, ImageFilter, ImageMorph

# --- thresholding ----------------------------------------------------------


def threshold(img, cutoff):
    """Grayscale -> binary.  Pixels above cutoff become 255, the rest 0."""
    return img.convert("L").point(lambda p: 255 if p > cutoff else 0)


def otsu(img):
    """Pick a cutoff automatically by maximising between-class variance."""
    hist = img.convert("L").histogram()
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))

    best, best_var = 0, -1.0
    w_bg, sum_bg = 0, 0
    for t, count in enumerate(hist):
        w_bg += count
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * count
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best, best_var = t, var
    return best


# --- structuring-element operations ----------------------------------------
#
# Pattern syntax (see PIL.ImageMorph): "4:(000 .1. 111)->0" means "in all 4
# rotations, where the neighbourhood matches, set the centre pixel to 0".
# '1' = must be on, '0' = must be off, '.' = don't care.

ERODE_8 = ["4:(... .1. .0.)->0", "4:(... .1. ..0)->0"]
ERODE_4 = ["4:(... .1. .0.)->0"]
DILATE_8 = ["4:(... .0. .1.)->1", "4:(... .0. ..1)->1"]
DILATE_4 = ["4:(... .0. .1.)->1"]

# --- thinning (Zhang-Suen) -------------------------------------------------
#
# ImageMorph indexes its 512-entry lookup table by the 3x3 neighbourhood,
# row-major from the top-left, with bit 4 as the centre pixel.  Building the
# table directly lets the thinning conditions run at C speed.

_BITS = {  # Zhang-Suen's P2..P9, clockwise from north
    "P2": 1, "P3": 2, "P4": 5, "P5": 8,
    "P6": 7, "P7": 6, "P8": 3, "P9": 0,
}


def _zhang_suen_lut(step):
    """Lookup table for one Zhang-Suen sub-iteration (step 1 or 2)."""
    lut = bytearray(512)
    for index in range(512):
        centre = (index >> 4) & 1
        lut[index] = centre
        if not centre:
            continue

        ring = [(index >> _BITS[f"P{n}"]) & 1 for n in range(2, 10)]
        neighbours = sum(ring)
        # 0 -> 1 transitions going once around the ring
        transitions = sum(
            1 for a, b in zip(ring, ring[1:] + ring[:1]) if a == 0 and b == 1
        )
        p2, p3, p4, p5, p6, p7, p8, p9 = ring

        if not (2 <= neighbours <= 6 and transitions == 1):
            continue
        if step == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
            lut[index] = 0
        elif step == 2 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
            lut[index] = 0
    return lut


_THIN_LUTS = [_zhang_suen_lut(1), _zhang_suen_lut(2)]


# A foreground pixel with exactly one 8-neighbour is a line end.
ENDPOINTS = ["4:(000 010 .1.)->0", "4:(000 010 ..1)->0"]


def _apply(img, patterns, reps=1):
    """Run a pattern set over the image, returning (image, pixels_changed).

    ImageMorph leaves the outermost pixel ring untouched, so each pass runs
    on a zero-padded copy and is cropped back.  That treats everything
    outside the frame as background, the usual convention, and lets shapes
    grow right up to the edge.
    """
    op = ImageMorph.MorphOp(patterns=patterns)
    w, h = img.size
    out, changed = img, 0
    for _ in range(reps):
        padded = Image.new("L", (w + 2, h + 2), 0)
        padded.paste(out, (1, 1))
        count, padded = op.apply(padded)
        out = padded.crop((1, 1, w + 1, h + 1))
        changed += count
    return out, changed


def _thin_cycle(img):
    """One thinning sweep: both Zhang-Suen sub-iterations, in order."""
    w, h = img.size
    out, changed = img, 0
    for lut in _THIN_LUTS:
        padded = Image.new("L", (w + 2, h + 2), 0)
        padded.paste(out, (1, 1))
        n, padded = ImageMorph.MorphOp(lut=lut).apply(padded)
        out = padded.crop((1, 1, w + 1, h + 1))
        changed += n
    return out, changed


def erode(img, reps=1, connectivity=8):
    """Shrink foreground: a pixel survives only if its neighbours are on."""
    return _apply(img, ERODE_8 if connectivity == 8 else ERODE_4, reps)[0]


def dilate(img, reps=1, connectivity=8):
    """Grow foreground: a pixel turns on if any neighbour is on."""
    return _apply(img, DILATE_8 if connectivity == 8 else DILATE_4, reps)[0]


def opening(img, reps=1):
    """Erode then dilate: drops specks, keeps the big shapes' size."""
    return dilate(erode(img, reps), reps)


def closing(img, reps=1):
    """Dilate then erode: fills pinholes and gaps, keeps the outline."""
    return erode(dilate(img, reps), reps)


def hit_or_miss(img, patterns, reps=1):
    """Match an explicit shape template.  patterns use the syntax above."""
    return _apply(img, patterns, reps)[0]


def boundary(img, reps=1):
    """A - erosion(A): the foreground's outline."""
    return ImageChops.subtract(img, erode(img, reps))


def skeleton(img, max_passes=600):
    """Thin to 1-pixel-wide centrelines, preserving topology.

    Passes needed scales with the thickest region's half-width (a filled
    500px-wide shape needs on the order of 250 passes), so this can be slow
    on full-resolution photos -- callers doing that synchronously should
    give the user some "working..." feedback first.
    """
    out = img
    for i in range(max_passes):
        out, changed = _thin_cycle(out)
        if not changed:
            break
    else:
        print(f"skeleton: stopped after {max_passes} passes, still changing "
              f"({changed} px last pass) -- result may be incomplete", file=__import__("sys").stderr)
    return out


def prune(img, reps=3):
    """Trim short spurs left behind by thinning."""
    return _apply(img, ENDPOINTS, reps)[0]


# --- convolution kernels ---------------------------------------------------
#
# A kernel is a small matrix slid over the image: multiply each pixel by the
# weight above it, sum, and that sum becomes the new centre pixel.

KERNELS = {
    "none": None,
    "box blur": ((1, 1, 1, 1, 1, 1, 1, 1, 1), 9, 0),
    "gaussian blur": ((1, 2, 1, 2, 4, 2, 1, 2, 1), 16, 0),
    "sharpen": ((0, -1, 0, -1, 5, -1, 0, -1, 0), 1, 0),
    "laplacian": ((0, 1, 0, 1, -4, 1, 0, 1, 0), 1, 128),
    "sobel x": ((-1, 0, 1, -2, 0, 2, -1, 0, 1), 1, 128),
    "sobel y": ((-1, -2, -1, 0, 0, 0, 1, 2, 1), 1, 128),
    "emboss": ((-2, -1, 0, -1, 1, 1, 0, 1, 2), 1, 128),
}


def convolve(img, name):
    """Apply a named 3x3 kernel to a grayscale image."""
    spec = KERNELS[name]
    if spec is None:
        return img
    weights, scale, offset = spec
    return img.convert("L").filter(ImageFilter.Kernel((3, 3), weights, scale, offset))


# --- op registry used by the GUI -------------------------------------------

# Convex corner of a filled region: centre on, two sides off.
CORNERS = ["1:(... ... ...)->0", "4:(00. 01. ...)->1"]

OPS = {
    "none": lambda im, n: im,
    "erosion": erode,
    "dilation": dilate,
    "opening": opening,
    "closing": closing,
    "boundary": boundary,
    "skeleton": lambda im, n: skeleton(im),
    "skeleton + prune": lambda im, n: prune(skeleton(im), n),
    "corners (hit-or-miss)": lambda im, n: hit_or_miss(im, CORNERS, n),
}
