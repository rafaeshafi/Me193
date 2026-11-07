"""The stop-action movie: numbered PNG frames, hand-drawn style, 3 frames a second, using the real trained net.

    ../../.venv/bin/python frames.py [--out frames] [--sample path.png] [--quick]

Every scene draws a few frames and holds them, so the pictures move in visible steps like a stop-motion film. The
voice-over text and the frame ranges go to script.md; build.sh turns the frames into movie_silent.mp4.
"""
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
from lock.cnn_np import NumpyNet, softmax  # noqa: E402
from lock.preprocess import to_input  # noqa: E402

FPS = 3
try:  # the net's input side, so the words match the deployed model (64 at first, 128 since the 3-vs-4 live test)
    SIZE = int(json.load(open(ROOT / "models" / "meta.json"))["input"]["size"])
except Exception:
    SIZE = 64
W, H, DPI = 12.8, 7.2, 100
INK, PAPER, RED, GREEN, BLUE, ORANGE, GREY = "#222222", "#fbf7ee", "#d1495b", "#3a9d5d", "#2f6fb0", "#e8871e", "#8a8a8a"
warnings.filterwarnings("ignore", message=".*font.*")


class Movie:
    def __init__(self, out_dir, quick=False):
        self.out, self.n, self.quick, self.scenes = Path(out_dir), 0, quick, []
        self.out.mkdir(parents=True, exist_ok=True)
        for p in self.out.glob("f*.png"):
            p.unlink()

    def scene(self, title, words):
        self.scenes.append({"title": title, "words": words, "start": self.n})

    def frame(self, fig, hold=1):
        """Save the figure as the next frame and repeat it hold times (copies, not re-renders)."""
        hold = 1 if self.quick else max(1, int(hold))
        path = self.out / f"f{self.n:05d}.png"
        fig.savefig(path, dpi=DPI, facecolor=PAPER)
        plt.close(fig)
        self.n += 1
        for _ in range(hold - 1):
            shutil.copy(path, self.out / f"f{self.n:05d}.png")
            self.n += 1

    def seconds(self, s):
        return max(1, int(round(s * FPS)))

    def script(self, path):
        lines = ["# Voice-over script", "", f"*{self.n} frames at {FPS} a second = {self.n / FPS:.0f} s. Play `movie_silent.mp4` and read "
                 "each paragraph while its scene is on screen (QuickTime > File > New Screen Recording with the microphone on, "
                 "or Zoom with 'record'). The times are where each scene starts.*", ""]
        for i, sc in enumerate(self.scenes):
            end = self.scenes[i + 1]["start"] if i + 1 < len(self.scenes) else self.n
            t0, t1 = sc["start"] / FPS, end / FPS
            lines += [f"## {i + 1}. {sc['title']}  ({t0 // 60:.0f}:{t0 % 60:02.0f} to {t1 // 60:.0f}:{t1 % 60:02.0f})", "", sc["words"], ""]
        Path(path).write_text("\n".join(lines))


def busy_patch(arr, k=8):
    """(r0, c0) of the k x k window of a 2-D array with the largest spread, so the numbers on screen are not all alike."""
    best, best_s = (0, 0), -1.0
    for r in range(0, arr.shape[0] - k + 1, 2):
        for c in range(0, arr.shape[1] - k + 1, 2):
            sd = float(arr[r:r + k, c:c + k].std())
            if sd > best_s:
                best, best_s = (r, c), sd
    return best


def mixed_patch(arr, k=8):
    """(r0, c0) of the k x k window with the most even mix of negative and positive numbers (for the ReLU scene)."""
    best, best_s = (0, 0), -1
    for r in range(0, arr.shape[0] - k + 1, 2):
        for c in range(0, arr.shape[1] - k + 1, 2):
            w = arr[r:r + k, c:c + k]
            score = min(int((w < -0.05).sum()), int((w > 0.05).sum()))
            if score > best_s:
                best, best_s = (r, c), score
    return best


def busy_channel(maps):
    """The channel whose map has the largest spread."""
    return int(np.argmax([m.std() for m in maps]))


def fig_ax(title=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 128), ax.set_ylim(0, 72), ax.axis("off")
    if title:
        ax.text(64, 67, title, ha="center", va="center", fontsize=26, color=INK)
    return fig, ax


def say(ax, x, y, text, size=15, color=INK, ha="left"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center")


def imshow_box(ax, img, x, y, w, h, vmin=None, vmax=None, cmap="gray", edge=INK, lw=2):
    ax.imshow(img, extent=(x, x + w, y, y + h), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=lw, ec=edge))


def grid_numbers(ax, arr, x, y, cell, fmt="{:.0f}", size=9, color=INK, highlight=None, hcolor=ORANGE):
    """Draw a small array as a grid with its numbers; highlight = (r0, c0, k) box."""
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x + c * cell, y + (rows - 1 - r) * cell), cell, cell, fill=False, lw=0.8, ec=GREY))
            ax.text(x + c * cell + cell / 2, y + (rows - 1 - r) * cell + cell / 2, fmt.format(arr[r, c]), ha="center",
                    va="center", fontsize=size, color=color)
    if highlight:
        r0, c0, k = highlight
        ax.add_patch(Rectangle((x + c0 * cell, y + (rows - r0 - k) * cell), k * cell, k * cell, fill=False, lw=3, ec=hcolor))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=20, lw=lw, color=color))


def bubble(ax, x, y, w, h, text, size=14, color=INK, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color)


# ----------------------------------------------------------------------------------------------- scenes
def scene_title(m, sample):
    m.scene("Title", "This is how my finger lock sees. A camera looks at a square on the desk, a small neural network "
            "counts the fingers in it, and a lock policy decides whether the safe opens. I built the network myself in "
            "PyTorch, trained it on my own hand, and then moved it onto the Arduino UNO Q. Let me walk through what "
            "happens to one picture.")
    for i in range(3):
        fig, ax = fig_ax()
        say(ax, 64, 50, "How my finger lock sees", 40, ha="center")
        say(ax, 64, 38, "a convolutional neural net, one picture at a time", 18, GREY, ha="center")
        if i >= 1:
            imshow_box(ax, sample, 52, 8, 24, 24)
        if i >= 2:
            say(ax, 80, 20, "← this hand", 16, RED)
        m.frame(fig, m.seconds(2))


def scene_pixels(m, sample):
    m.scene("From a picture to numbers", "The webcam gives a colour picture. I only keep the square where the hand goes, "
            f"turn it to grey and shrink it to {SIZE} by {SIZE}. To the network a picture is nothing but a grid of {SIZE * SIZE:,} numbers, "
            "one per pixel, 0 for black and 1 for white. Everything that follows is arithmetic on these numbers.")
    pr, pc = busy_patch(sample, 8)
    patch = sample[pr:pr + 8, pc:pc + 8]
    for step in range(5):
        fig, ax = fig_ax("A picture is just numbers")
        imshow_box(ax, sample, 6, 14, 40, 40)
        say(ax, 26, 10, f"{SIZE} x {SIZE} grey pixels", 14, GREY, ha="center")
        if step >= 1:
            ax.add_patch(Rectangle((6 + pc * 40 / SIZE, 14 + (SIZE - pr - 8) * 40 / SIZE), 8 * 40 / SIZE, 8 * 40 / SIZE, fill=False, lw=3, ec=RED))
            arrow(ax, 46, 36, 58, 36, RED)
        if step >= 2:
            rows = min(8, 2 * (step - 1))
            shown = np.full_like(patch, np.nan)
            shown[:rows] = patch[:rows]
            for r in range(8):
                for c in range(8):
                    v = shown[r, c]
                    ax.add_patch(Rectangle((60 + c * 7, 14 + (7 - r) * 5.4), 7, 5.4, fill=True, fc=str(0.3 + 0.7 * (0 if np.isnan(v) else v)), ec=GREY, lw=0.8))
                    if not np.isnan(v):
                        ax.text(60 + c * 7 + 3.5, 14 + (7 - r) * 5.4 + 2.7, f"{v:.2f}", ha="center", va="center", fontsize=10,
                                color=INK if v > 0.5 else "white")
            say(ax, 88, 60, "8 x 8 of them, zoomed in", 14, GREY, ha="center")
        m.frame(fig, m.seconds(2.5 if step else 2))


def scene_convolution(m, sample, net):
    m.scene("Convolution", "The first layer is a convolution. A tiny 3 by 3 window of weights, called a kernel or filter, "
            "slides over the picture one pixel at a time. At every position it multiplies the nine pixels under it by "
            "its nine weights and adds them up; that single number goes into a new picture called a feature map. The "
            "weights were learned, not chosen by me: this filter happens to respond to edges running one way. The first "
            "layer has sixteen different filters, so it makes sixteen feature maps, each looking for something "
            "different.")
    k_f = int(np.argmax(np.abs(net.layers[0][1]).sum(axis=(1, 2, 3))))  # the strongest first-layer filter
    w = net.layers[0][1][k_f, 0]  # (3, 3)
    b = float(net.layers[0][2][k_f])
    x = (sample - net.mean) / net.std
    pr, pc = busy_patch(x, 8)
    patch = x[pr:pr + 8, pc:pc + 8]
    out = np.full((6, 6), np.nan)
    positions = [(r, c) for r in range(6) for c in range(6)]
    for step, (r, c) in enumerate(positions):
        val = float((patch[r:r + 3, c:c + 3] * w).sum() + b)
        out[r, c] = val
        if step >= 7 and step % 5 != 0 and step != len(positions) - 1:
            continue  # the first positions one by one, then in jumps
        fig, ax = fig_ax("Convolution: a 3x3 window of weights slides over the pixels")
        grid_numbers(ax, patch, 4, 12, 6, "{:.1f}", 9, highlight=(r, c, 3))
        say(ax, 28, 8, "the pixels (standardised)", 12, GREY, ha="center")
        grid_numbers(ax, w, 58, 34, 6, "{:+.2f}", 11, ORANGE)
        say(ax, 67, 54, "the filter (learned weights)", 12, ORANGE, ha="center")
        say(ax, 67, 30, "x", 18, INK, ha="center")
        if step < 7:
            terms = " + ".join(f"{patch[r + i, c + j]:.1f}x{w[i, j]:+.2f}" for i in range(3) for j in range(3))
            say(ax, 64, 22, f"{terms[:62]}\n{terms[62:]} + bias = {val:+.2f}", 9, INK, ha="center")
        else:
            say(ax, 64, 22, f"multiply the 9 pairs, add them: {val:+.2f}", 12, INK, ha="center")
        grid_numbers(ax, np.nan_to_num(out, nan=0.0), 92, 20, 5.5, "{:+.1f}", 8, BLUE, highlight=(r, c, 1), hcolor=BLUE)
        for rr in range(6):
            for cc in range(6):
                if np.isnan(out[rr, cc]):
                    ax.add_patch(Rectangle((92 + cc * 5.5, 20 + (5 - rr) * 5.5), 5.5, 5.5, fill=True, fc=PAPER, ec=GREY, lw=0.8))
        say(ax, 108, 56, "the feature map", 12, BLUE, ha="center")
        arrow(ax, 44, 36, 56, 36), arrow(ax, 80, 36, 90, 36)
        m.frame(fig, m.seconds(2.0 if step < 3 else 1.2))
    acts = dict(net.activations(sample))
    maps = acts["conv1"]
    for k in (1, 4, 16):
        fig, ax = fig_ax("16 filters → 16 feature maps (the real ones for this hand)")
        imshow_box(ax, sample, 4, 22, 24, 24)
        arrow(ax, 30, 34, 36, 34)
        for i in range(k):
            r, c = divmod(i, 8)
            mp = maps[i]
            imshow_box(ax, mp, 38 + c * 11, 40 - r * 22, 10, 18, vmin=mp.min(), vmax=mp.max(), cmap="viridis", edge=GREY, lw=1)
        m.frame(fig, m.seconds(2.5))


def scene_relu(m, sample, net):
    m.scene("ReLU", "After each convolution comes ReLU, the rectified linear unit. It is the simplest rule there is: if "
            "a number is negative, make it zero; otherwise leave it alone. Without this bend the whole network would "
            "collapse into one big multiplication and could only draw straight lines. The bend is what lets it learn "
            "shapes.")
    acts = dict(net.activations(sample))
    ch = busy_channel(acts["conv1"])
    before = acts["conv1"][ch]
    rr, rc = mixed_patch(before, 8)
    for step in range(4):
        fig, ax = fig_ax("ReLU: negatives become zero")
        xs = np.linspace(-3, 3, 50)
        ax.plot(20 + xs * 5, 30 + np.maximum(xs, 0) * 6, color=RED, lw=3)
        ax.plot([5, 35], [30, 30], color=GREY, lw=1), ax.plot([20, 20], [26, 50], color=GREY, lw=1)
        say(ax, 20, 54, "y = max(0, x)", 16, RED, ha="center")
        say(ax, 20, 22, "x in, y out", 12, GREY, ha="center")
        sub_b = before[rr:rr + 8, rc:rc + 8]
        shown = sub_b.copy()
        rows = 2 * step
        shown[:rows] = np.maximum(shown[:rows], 0)
        grid_numbers(ax, sub_b, 44, 14, 5, "{:+.1f}", 8, INK)
        say(ax, 64, 58, "before", 13, GREY, ha="center")
        arrow(ax, 86, 34, 92, 34, RED)
        grid_numbers(ax, shown, 94, 14, 4, "{:.1f}", 8, BLUE)
        say(ax, 110, 58, "after ReLU", 13, BLUE, ha="center")
        m.frame(fig, m.seconds(3))


def scene_pool(m, sample, net):
    m.scene("Max pooling", "Then max pooling. The feature map is cut into 2 by 2 blocks and only the biggest number in "
            "each block survives. The map shrinks to half its width and half its height. We lose exact positions, which "
            "is the point: a finger is a finger whether it sits two pixels left or right, and the next layer gets four "
            "times less to look at.")
    acts = dict(net.activations(sample))
    ch = busy_channel(acts["relu1"])
    mp = acts["relu1"][ch]
    rr, rc = busy_patch(mp, 4)
    sub = mp[rr:rr + 4, rc:rc + 4]
    pooled = np.full((2, 2), np.nan)
    blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for step in range(5):
        fig, ax = fig_ax("Max pooling: keep the biggest of every 2x2 block")
        grid_numbers(ax, sub, 10, 16, 8, "{:.1f}", 12, INK, highlight=(blocks[min(step, 3)][0] * 2, blocks[min(step, 3)][1] * 2, 2) if step < 4 else None)
        if step > 0:
            for i in range(min(step, 4)):
                r, c = blocks[i]
                pooled[r, c] = sub[2 * r:2 * r + 2, 2 * c:2 * c + 2].max()
        shown = np.nan_to_num(pooled, nan=0.0)
        grid_numbers(ax, shown, 56, 24, 8, "{:.1f}", 12, BLUE)
        for r in range(2):
            for c in range(2):
                if np.isnan(pooled[r, c]):
                    ax.add_patch(Rectangle((56 + c * 8, 24 + (1 - r) * 8), 8, 8, fill=True, fc=PAPER, ec=GREY))
        arrow(ax, 44, 32, 54, 32)
        say(ax, 26, 10, "4 x 4", 13, GREY, ha="center"), say(ax, 64, 18, "2 x 2", 13, BLUE, ha="center")
        imshow_box(ax, mp, 84, 30, 20, 20, cmap="viridis", edge=GREY, lw=1)
        say(ax, 94, 54, "whole map 32 x 32", 11, GREY, ha="center")
        if step >= 4:
            m2 = acts["maxpool1"][ch]
            imshow_box(ax, m2, 108, 35, 10, 10, cmap="viridis", edge=GREY, lw=1)
            arrow(ax, 105, 40, 107, 40)
            say(ax, 113, 48, "16 x 16", 11, BLUE, ha="center")
        m.frame(fig, m.seconds(3))


def scene_stack(m, sample, net, meta):
    m.scene("Three layers deep", "My network does this three times. Convolution, ReLU, pooling, with sixteen filters, "
            "then thirty-two, then sixty-four. Each stage sees a bigger piece of the hand: the first finds edges, the "
            "second joins edges into strokes, the third tells a finger-shaped stroke from a palm. By the end the picture "
            "is sixty-four tiny 8 by 8 maps. That whole stack has twenty-three thousand weights; the picture went in as "
            "four thousand numbers and comes out as four thousand numbers that mean something.")
    acts = net.activations(sample)
    stages = [(f"input\n1 x {SIZE} x {SIZE}", acts[0][1][0], 22), (f"conv 16 + ReLU + pool\n16 x {SIZE // 2} x {SIZE // 2}", dict(acts)["maxpool1"][2], 17),
              (f"conv 32 + ReLU + pool\n32 x {SIZE // 4} x {SIZE // 4}", dict(acts)["maxpool2"][5], 13), (f"conv 64 + ReLU + pool\n64 x {SIZE // 8} x {SIZE // 8}", dict(acts)["maxpool3"][7], 9)]
    for k in range(1, 5):
        fig, ax = fig_ax("The whole stack: conv → ReLU → pool, three times")
        for i in range(k):
            label, img, size = stages[i]
            x = 6 + i * 31
            for d in range(min(3, 1 + 2 * i)):
                imshow_box(ax, img, x + 2 * d, 26 - size / 2 + 2 * d, size, size, cmap="gray" if i == 0 else "viridis", edge=GREY, lw=1)
            say(ax, x + size / 2 + 3, 50, label, 11, INK, ha="center")
            if i:
                arrow(ax, x - 8, 26, x - 1, 26)
        if k == 4:
            say(ax, 64, 8, f"{meta_params(meta, 'conv'):,} weights in the three convolutions; the input 4,096 numbers → 4,096 numbers that mean something", 11, GREY, ha="center")
        m.frame(fig, m.seconds(4))


def meta_params(meta, which):
    return 160 + 4640 + 18496 if which == "conv" else meta.get("n_params", 0)


def scene_linear(m, sample, net):
    m.scene("Flatten, linear layer, softmax", "Now the decision. The 64 little maps are unrolled into one long line of "
            "4,096 numbers. A linear layer takes a weighted sum of all of them for each of 64 hidden units, then one "
            "more linear layer turns those into six scores, one per answer: zero to five fingers. Softmax squashes the "
            "six scores into probabilities that add up to one. The biggest probability is the network's answer. For "
            "this hand it says three fingers, and you can read how sure it is.")
    acts = dict(net.activations(sample))
    flat, hidden, logits = acts["flatten1"], acts["relu4"], acts["linear2"]
    probs = softmax(logits)
    for step in range(5):
        fig, ax = fig_ax("Flatten → linear → linear → softmax")
        for i in range(8):
            r, c = divmod(i, 4)
            imshow_box(ax, acts["maxpool3"][i], 4 + c * 6, 36 - r * 6, 5, 5, cmap="viridis", edge=GREY, lw=0.8)
        say(ax, 14, 24, "64 maps of 8x8", 11, GREY, ha="center")
        if step >= 1:
            arrow(ax, 30, 36, 36, 36)
            ax.imshow(flat[None, :256], extent=(38, 62, 34, 38), cmap="viridis", aspect="auto")
            ax.add_patch(Rectangle((38, 34), 24, 4, fill=False, ec=GREY))
            say(ax, 50, 42, "flatten: 4,096 numbers in a row", 11, GREY, ha="center")
        if step >= 2:
            arrow(ax, 64, 36, 70, 36)
            ax.imshow(hidden[None, :], extent=(72, 84, 34, 38), cmap="viridis", aspect="auto")
            ax.add_patch(Rectangle((72, 34), 12, 4, fill=False, ec=GREY))
            say(ax, 78, 42, "64 hidden\n(weighted sums + ReLU)", 10, GREY, ha="center")
        if step >= 3:
            arrow(ax, 86, 36, 92, 36)
            for i, z in enumerate(logits):
                ax.add_patch(Rectangle((94, 16 + i * 6), max(0.3, 6 * (z - logits.min()) / (np.ptp(logits) + 1e-6)), 4, fc=GREY, ec=INK))
                say(ax, 93, 18 + i * 6, f"{i}", 11, INK, ha="right")
            say(ax, 100, 54, "6 scores", 11, GREY, ha="center")
        if step >= 4:
            arrow(ax, 104, 36, 108, 36)
            for i, p in enumerate(probs):
                ax.add_patch(Rectangle((110, 16 + i * 6), 16 * p, 4, fc=GREEN if i == probs.argmax() else GREY, ec=INK))
                say(ax, 112 + 16 * p, 18 + i * 6, f"{100 * p:.0f}%", 10, INK)
            say(ax, 116, 54, f"softmax → {probs.argmax()} fingers", 12, GREEN, ha="center")
        m.frame(fig, m.seconds(5 if step == 4 else 3))


def scene_loss(m, sample, net):
    m.scene("Cross-entropy loss", "How does it learn? During training the right answer is known, because I typed it "
            "when I took the picture. The loss is cross entropy: minus the log of the probability the network gave the "
            "true answer. If it was sure and right, the loss is near zero. If it gave the true answer only 10 percent, "
            "the loss is 2.3. One number says how wrong the whole network was on that picture.")
    probs = softmax(dict(net.activations(sample))["linear2"])
    true = int(probs.argmax())
    for step, p_true in enumerate([probs[true], 0.5, 0.1, 0.9]):
        fig, ax = fig_ax("Cross-entropy: how wrong was it?")
        say(ax, 64, 52, "loss = −log( probability of the true answer )", 20, ha="center")
        ps = np.linspace(0.02, 1, 60)
        ax.plot(20 + ps * 50, 10 + (-np.log(ps)) * 7, color=RED, lw=3)
        ax.plot([20, 70], [10, 10], color=GREY), ax.plot([20, 20], [10, 40], color=GREY)
        say(ax, 45, 6, "probability given to the true answer", 11, GREY, ha="center")
        ax.plot([20 + p_true * 50], [10 + (-np.log(p_true)) * 7], "o", color=BLUE, ms=12)
        say(ax, 95, 30, f"p = {p_true:.2f}\nloss = {-np.log(p_true):.2f}", 22, BLUE, ha="center")
        if step == 0:
            say(ax, 95, 16, f"(this hand, true = {true})", 12, GREY, ha="center")
        m.frame(fig, m.seconds(3.5))


def scene_gradient(m, meta):
    m.scene("Gradient descent", "Every weight gets nudged to make that loss smaller. Picture the loss as a bowl and the "
            "weights as a ball somewhere on its wall. The gradient is the slope under the ball. Gradient descent takes "
            "a small step downhill: new weight equals old weight minus the learning rate times the slope. Do that for "
            "all 286 thousand weights, on every batch of pictures, for forty passes over the data, and the loss curve "
            "falls like this. The real curve from my training run is drawn here, one epoch per step.")
    xs = np.linspace(-3, 3, 100)
    pos = [2.6]
    for _ in range(9):
        pos.append(pos[-1] - 0.25 * 2 * pos[-1])
    for step in range(10):
        fig, ax = fig_ax("Gradient descent: step downhill")
        ax.plot(30 + xs * 8, 14 + xs ** 2 * 4, color=INK, lw=3)
        px = pos[step]
        ax.plot([30 + px * 8], [14 + px ** 2 * 4 + 1.5], "o", color=RED, ms=18)
        if step < 9:
            arrow(ax, 30 + px * 8, 14 + px ** 2 * 4 + 1.5, 30 + pos[step + 1] * 8, 14 + pos[step + 1] ** 2 * 4 + 1.5, RED, 3)
        say(ax, 30, 58, "loss", 14, GREY, ha="center"), say(ax, 58, 12, "a weight →", 12, GREY)
        say(ax, 95, 44, "w ← w − lr · ∂loss/∂w", 20, ha="center")
        say(ax, 95, 34, "slope under the ball: " + f"{2 * px:+.2f}", 14, RED, ha="center")
        say(ax, 95, 26, "learning rate 0.001, Adam, batch 64", 11, GREY, ha="center")
        m.frame(fig, m.seconds(1.5 if step else 4))
    hist = meta["history"].get("augmented") or next(iter(meta["history"].values()))
    ep = [h["epoch"] for h in hist]
    tl, vl, va = [h["train_loss"] for h in hist], [h["val_loss"] for h in hist], [h["val_acc"] for h in hist]
    for k in range(2, len(ep) + 1, 2):
        fig, ax = fig_ax("My real training run: loss per epoch")
        ax2 = fig.add_axes([0.1, 0.12, 0.5, 0.7])
        ax2.set_facecolor(PAPER)
        ax2.plot(ep[:k], tl[:k], "o-", color=BLUE, label="training loss")
        ax2.plot(ep[:k], vl[:k], "s-", color=RED, label="validation loss")
        ax2.set_xlim(0, ep[-1] + 1), ax2.set_ylim(0, max(tl + vl) * 1.1)
        ax2.set_xlabel("epoch (one pass over all pictures)"), ax2.set_ylabel("cross-entropy loss"), ax2.legend()
        for s in ax2.spines.values():
            s.set_color(INK)
        say(ax, 95, 40, f"epoch {ep[k - 1]}\nvalidation accuracy {100 * (va[k - 1] or 0):.0f}%", 20, GREEN, ha="center")
        m.frame(fig, m.seconds(0.7))
    m.frame(fig_last(ep, tl, vl, va), m.seconds(3))


def fig_last(ep, tl, vl, va):
    fig, ax = fig_ax("My real training run: loss per epoch")
    ax2 = fig.add_axes([0.1, 0.12, 0.5, 0.7])
    ax2.set_facecolor(PAPER)
    ax2.plot(ep, tl, "o-", color=BLUE, label="training loss"), ax2.plot(ep, vl, "s-", color=RED, label="validation loss")
    ax2.set_xlabel("epoch"), ax2.set_ylabel("cross-entropy loss"), ax2.legend()
    say(ax, 95, 40, f"after {ep[-1]} epochs\nvalidation accuracy {100 * (va[-1] or 0):.0f}%", 20, GREEN, ha="center")
    return fig


def scene_backprop(m):
    m.scene("Back-propagation", "Where do the slopes come from? Back-propagation. The loss is computed at the end, "
            "then the chain rule walks backwards through the layers: how much did the last layer's weights change the "
            "loss, then the layer before, all the way back to the first filter. Each layer only needs the gradient "
            "handed to it from the layer after it. PyTorch does the bookkeeping; the one line in my code is "
            "loss dot backward.")
    names = ["conv 1", "conv 2", "conv 3", "linear", "linear", "loss"]
    for step in range(7):
        fig, ax = fig_ax("Back-propagation: the chain rule, backwards")
        for i, n in enumerate(names):
            x = 8 + i * 20
            col = RED if (6 - step) <= i < 6 or (step > 0 and i == 5) else INK
            bubble(ax, x, 30, 14, 10, n, 13, col)
            if i < 5:
                arrow(ax, x + 15, 35, x + 19, 35, GREY)
        for i in range(5, 6 - step, -1):
            x = 8 + i * 20
            arrow(ax, x, 27, x - 6, 27, RED, 3)
        say(ax, 64, 52, "forward: picture → … → loss", 14, GREY, ha="center")
        say(ax, 64, 16, "backward: ∂loss/∂(each weight), layer by layer", 14, RED, ha="center")
        say(ax, 64, 9, "loss.backward()   then   optimizer.step()", 14, INK, ha="center")
        m.frame(fig, m.seconds(3))


def scene_augment(m, sample):
    m.scene("More pictures for free", "I took around a thousand pictures, but the network wants more. So every time "
            "a picture is used, it is rotated a little, shifted, scaled, flipped and re-lit, differently each time. The "
            "label stays the same: a rotated three is still a three. This is augmentation, and it is what makes the lock "
            "forgive a hand that is not exactly where the training hand was.")
    from dataset import augment
    rng = np.random.default_rng(3)
    for step in range(5):
        fig, ax = fig_ax("Augmentation: one picture, many pictures")
        imshow_box(ax, sample, 6, 22, 24, 24)
        say(ax, 18, 18, "the picture I took", 12, GREY, ha="center")
        for i in range(min(step * 2, 8)):
            r, c = divmod(i, 4)
            imshow_box(ax, augment(sample, rng), 40 + c * 21, 36 - r * 23, 18, 18, edge=GREY, lw=1)
        if step:
            arrow(ax, 32, 34, 38, 34)
            say(ax, 80, 60, "rotate · shift · scale · flip · brighten · noise", 13, ORANGE, ha="center")
        m.frame(fig, m.seconds(3))


def scene_policy(m):
    m.scene("The lock policy", "Finally the lock. Press the button and it listens. Ten times a second the network "
            "says what it sees. A digit counts only when the same answer comes back for eight frames in a row with "
            "high confidence, so a flicker is ignored; then the hand has to leave before the next digit starts. "
            "Three digits make a code. If they match, the servo opens the latch for five seconds. If not, red light, "
            "and after three wrong tries the safe ignores the button for half a minute.")
    seq = [("2", 0.3), ("2", 0.7), ("2", 1.0), ("5", 0.5), ("5", 1.0), ("3", 0.6), ("3", 1.0)]
    for step, (d, prog) in enumerate(seq + [("ok", 1.0)]):
        fig, ax = fig_ax("The lock policy: hold to accept, release, three digits")
        done = ["2"] * (step >= 3) + ["5"] * (step >= 5) + ["3"] * (step >= 7)
        for i in range(3):
            bubble(ax, 20 + i * 22, 36, 14, 14, done[i] if i < len(done) else "_", 30, GREEN if i < len(done) else GREY)
        if d != "ok":
            say(ax, 90, 50, f"net sees {d}  ({prog * 8:.0f}/8 frames)", 15, BLUE, ha="center")
            ax.add_patch(Rectangle((74, 40), 32 * prog, 4, fc=ORANGE, ec=INK))
            ax.add_patch(Rectangle((74, 40), 32, 4, fill=False, ec=INK))
            say(ax, 90, 34, "hold …", 13, GREY, ha="center")
        else:
            say(ax, 90, 44, "2 5 3 = code → UNLOCKED", 18, GREEN, ha="center")
            say(ax, 90, 34, "servo opens 5 s, green LED", 13, GREY, ha="center")
        say(ax, 64, 14, "wrong code → red, 3 wrong → 30 s lockout", 13, RED, ha="center")
        m.frame(fig, m.seconds(2.5))


def scene_end(m, meta):
    m.scene("The end", "That is the whole network: numbers in, convolutions that look for shapes, pooling that forgives "
            "position, a linear layer that votes, and a loss that teaches it all through gradient descent. The same "
            "weights now run on the UNO Q in plain numpy, and the safe opens for my hand.")
    fig, ax = fig_ax()
    say(ax, 64, 46, "FingerNet", 36, ha="center")
    say(ax, 64, 34, f"{meta.get('n_params', 0):,} weights · 3 convolutions · trained in PyTorch", 16, GREY, ha="center")
    acc = meta.get("val_accuracy")
    say(ax, 64, 26, f"held-out accuracy {100 * acc:.0f}%" if acc else "", 16, GREEN, ha="center")
    say(ax, 64, 16, "runs on the Arduino UNO Q in numpy", 16, GREY, ha="center")
    m.frame(fig, m.seconds(5))


def load_sample(path, net):
    if path and Path(path).exists():
        import cv2
        return to_input(cv2.imread(str(path), cv2.IMREAD_COLOR), net.size)
    import synth
    return to_input(synth.hand(3, np.random.default_rng(7)), net.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE / "frames")
    ap.add_argument("--sample", default=None, help="a data/raw picture (default: the first class-3 picture, else synthetic)")
    ap.add_argument("--quick", action="store_true", help="no holds (fast check)")
    a = ap.parse_args()
    net = NumpyNet.load(ROOT / "models" / "fingers.json")
    meta = json.load(open(ROOT / "models" / "meta.json"))
    sample_path = a.sample
    if sample_path is None:
        real = sorted((ROOT / "data" / "raw" / "3").glob("*.png")) if (ROOT / "data" / "raw" / "3").exists() else []
        sample_path = real[0] if real else None
    sample = load_sample(sample_path, net)
    plt.xkcd(scale=1.0, length=100, randomness=2)
    plt.rcParams["font.family"] = ["Comic Sans MS", "DejaVu Sans"]  # DejaVu supplies the arrows Comic Sans lacks
    m = Movie(a.out, a.quick)
    scene_title(m, sample)
    scene_pixels(m, sample)
    scene_convolution(m, sample, net)
    scene_relu(m, sample, net)
    scene_pool(m, sample, net)
    scene_stack(m, sample, net, meta)
    scene_linear(m, sample, net)
    scene_loss(m, sample, net)
    scene_gradient(m, meta)
    scene_backprop(m)
    scene_augment(m, sample)
    scene_policy(m)
    scene_end(m, meta)
    m.script(HERE / "script.md")
    print(f"{m.n} frames in {a.out} ({m.n / FPS:.0f} s at {FPS} fps); script.md written")


if __name__ == "__main__":
    main()
