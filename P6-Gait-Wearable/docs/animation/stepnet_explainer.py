"""How the step counter works, as one Manim scene (~75 s, captions only, no LaTeX).

  docs/animation/render.sh            # -> docs/animation/stepnet_explainer.mp4 (1080p) and _small.mp4 (480p, for Notion)
"""
import math
import textwrap

import numpy as np
from manim import (BLUE, ChangeDecimalToValue, DOWN, GREEN, GREY_B, LEFT, ORANGE, RED, RIGHT, UP, WHITE, YELLOW, Arrow, Axes, Create,
                   DecimalNumber, Dot, FadeIn, FadeOut, Line, Rectangle, RoundedRectangle, Scene, Text, Transform,
                   ValueTracker, VGroup, VMobject, Write, always_redraw, config, rate_functions)

config.background_color = "#0b0b10"
Text.set_default(font="Helvetica Neue")
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
CH_COLORS = [RED, GREEN, BLUE, ORANGE, YELLOW, "#c084fc"]
STEP_F, STEP_T0, N_STEPS = 1.5, 0.6, 7  # the demo walk in part 4: 7 steps at 1.5 steps/s


def caption(txt, size=26, chars=72):
    """Bottom caption, wrapped to lines and shrunk if it would still overflow the frame."""
    t = Text("\n".join(textwrap.wrap(txt, chars)), font_size=size, color=GREY_B, line_spacing=0.9)
    if t.width > config.frame_width - 1.0:
        t.scale_to_fit_width(config.frame_width - 1.0)
    return t.to_edge(DOWN, buff=0.35)


def num(value, **kw):
    """A DecimalNumber drawn with Text glyphs (the default MathTex needs LaTeX)."""
    return DecimalNumber(value, mob_class=Text, **kw)


def title(txt, size=40):
    t = Text(txt, font_size=size)
    if t.width > config.frame_width - 1.0:
        t.scale_to_fit_width(config.frame_width - 1.0)
    return t.to_edge(UP, buff=0.4)


def trace(fn, x0, x1, n=200, **kw):
    return VMobject(**kw).set_points_as_corners([np.array([x, fn(x), 0]) for x in np.linspace(x0, x1, n)])


def step_rate(t):
    """One raised-cosine bump per step, each with area exactly 1, so the integral is the step count."""
    u = t - STEP_T0
    if u < 0 or u > N_STEPS / STEP_F:
        return 0.0
    return STEP_F * (1 - math.cos(2 * math.pi * STEP_F * u)) ** 2 / 1.5


def steps_so_far(t):
    u = min(max(t - STEP_T0, 0.0), N_STEPS / STEP_F)
    w = 2 * math.pi * STEP_F
    # integral of f(1 - cos wu)^2 / 1.5 = f/1.5 * (1.5u - 2 sin(wu)/w + sin(2wu)/(4w))
    return STEP_F / 1.5 * (1.5 * u - 2 * math.sin(w * u) / w + math.sin(2 * w * u) / (4 * w))


class StepNetExplainer(Scene):
    def construct(self):
        self.part_wearable()
        self.part_prepare()
        self.part_net()
        self.part_rate_to_count()
        self.part_loss()
        self.part_backprop()
        self.part_ship()

    def clear_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

    def swap_caption(self, old, txt):
        new = caption(txt)
        self.play(FadeOut(old), FadeIn(new), run_time=0.6)
        return new

    # 1. the wearable: a thigh with the board, six traces scrolling out of it
    def part_wearable(self):
        ttl = title("A thigh-worn step counter", 44)
        thigh = RoundedRectangle(corner_radius=0.6, width=1.6, height=4, color=GREY_B).move_to(LEFT * 5.2 + UP * 0.3)
        board = Rectangle(width=1.0, height=1.4, color=BLUE, fill_opacity=0.3).move_to(thigh.get_center() + UP * 0.4)
        chip = Rectangle(width=0.4, height=0.4, color=WHITE, fill_opacity=0.6).move_to(board.get_center() + DOWN * 0.4)
        lbl = Text("UNO Q + MPU6050", font_size=22).next_to(thigh, DOWN, buff=0.2)
        self.play(Write(ttl), Create(thigh), FadeIn(board), FadeIn(chip), Write(lbl))
        rows = VGroup()
        for i, (name, col) in enumerate(zip(CH, CH_COLORS)):
            y = 2.15 - i * 0.78
            f, amp = (1.8, 0.26) if i < 3 else (0.9, 0.3)
            tr = trace(lambda x, y=y, a=amp, ff=f, i=i: y + a * math.sin(2 * math.pi * ff * x + i), 0, 5.5,
                       color=col, stroke_width=3).shift(RIGHT * -2.0)
            tag = Text(name, font_size=22, color=col).next_to(tr, LEFT, buff=0.2)
            rows.add(VGroup(tag, tr))
        self.play(*[Create(r[1]) for r in rows], *[FadeIn(r[0]) for r in rows], run_time=3)
        cap = caption("Six channels at 100 Hz: 3 accelerations (g) and 3 rotation rates (deg/s). "
                      "Every one of them goes into the net.")
        self.play(FadeIn(cap))
        self.wait(2)
        self.clear_all()

    # 2. decimate 4 -> 1 and standardise
    def part_prepare(self):
        ttl = title("Prepare: 100 Hz to 25 Hz, then standardise")
        self.play(Write(ttl))
        dots = VGroup(*[Dot(point=np.array([-5 + i * 0.5, 1.3 + 0.6 * math.sin(i), 0]), color=RED) for i in range(20)])
        self.play(FadeIn(dots))
        cap = caption("Average every 4 samples into 1. Walking lives below a few Hz, so 25 Hz keeps all of it, "
                      "and the board does a quarter of the work.")
        self.play(FadeIn(cap))
        for b in range(5):
            grp = dots[4 * b:4 * b + 4]
            m = Dot(point=np.mean([d.get_center() for d in grp], axis=0), color=RED, radius=0.14)
            self.play(*[Transform(d, m.copy()) for d in grp], run_time=0.35)
        self.wait(0.5)
        cap = self.swap_caption(cap, "Standardise each channel: subtract its mean, divide by its spread. "
                                     "Now a g and a deg/s weigh the same.")
        axis = Line(LEFT * 5.5 + DOWN * 1.2, RIGHT * 5.5 + DOWN * 1.2, color=GREY_B)
        zero = Text("0", font_size=22, color=GREY_B).next_to(axis, LEFT, buff=0.15)
        formula = Text("z = (x - mean) / std", font_size=34, color=YELLOW).move_to(RIGHT * 2.5 + UP * 0.2)
        self.play(Create(axis), FadeIn(zero), Write(formula))
        self.play(dots.animate.scale(0.8).move_to(DOWN * 1.2 + LEFT * 0.2), run_time=1.5)
        self.wait(2)
        self.clear_all()

    # 3. the net: three dilated causal conv layers and the head
    def layers_drawing(self):
        cols = []
        spec = [(6, "6 inputs"), (12, "conv 12, k5\ndilation 1"), (12, "conv 12, k5\ndilation 2"),
                (12, "conv 12, k5\ndilation 4"), (1, "rate")]
        for li, (n, label) in enumerate(spec):
            x = -5.2 + li * 2.6
            units = VGroup(*[Dot(point=np.array([x, 0.45 + (n - 1) / 2 * 0.26 - k * 0.26, 0]), radius=0.08,
                                 color=CH_COLORS[k] if li == 0 else (YELLOW if li == 4 else WHITE)) for k in range(n)])
            tag = Text(label, font_size=20, color=GREY_B, line_spacing=0.8)
            tag.move_to(np.array([x, -1.45, 0]))
            cols.append(VGroup(units, tag))
        links = VGroup()
        for a, b in zip(cols[:-1], cols[1:]):
            for ua in a[0]:
                for ub in b[0]:
                    links.add(Line(ua.get_center(), ub.get_center(), stroke_width=0.6, color=GREY_B, stroke_opacity=0.35))
        return VGroup(*cols), links

    def part_net(self):
        ttl = title("The net: causal dilated 1-D convolutions")
        self.play(Write(ttl))
        cols, links = self.layers_drawing()
        self.play(Create(links, lag_ratio=0.0), *[FadeIn(c) for c in cols], run_time=2)
        cap = caption("Each layer looks only at the present and the past (causal). Kernel 5 with dilations "
                      "1, 2, 4 lets the output see the last 1.2 s of movement.")
        self.play(FadeIn(cap))
        # one time step: the input window lights up, the activity runs left to right, one rate value comes out
        out = cols[-1][0][0]
        for _ in range(2):
            for c in cols[1:-1]:
                self.play(c[0].animate.set_color(ORANGE), run_time=0.15)
            self.play(out.animate.set_color(ORANGE).scale(1.6), run_time=0.2)
            self.play(*[c[0].animate.set_color(WHITE) for c in cols[1:-1]], out.animate.set_color(YELLOW).scale(1 / 1.6),
                      run_time=0.25)
        cap = self.swap_caption(cap, "ReLU after each conv; a Softplus head keeps the output at or above zero: "
                                     "a step rate in steps per second. About 1,800 weights in all.")
        self.wait(3)
        self.clear_all()

    # 4. rate curve -> area -> count
    def part_rate_to_count(self):
        ttl = title("Rate to count: the area under the curve")
        self.play(Write(ttl))
        ax = Axes(x_range=[0, 6, 1], y_range=[0, 4.5, 1], x_length=9, y_length=3.2,
                  axis_config={"color": GREY_B, "include_numbers": False}).shift(LEFT * 0.8 + UP * 0.45)
        xl = Text("time (s)", font_size=22).next_to(ax.x_axis, DOWN, buff=0.15)
        yl = Text("steps / s", font_size=22).rotate(math.pi / 2).next_to(ax.y_axis, LEFT, buff=0.2)
        self.play(Create(ax), Write(xl), Write(yl))
        full = ax.plot(step_rate, x_range=[0, 6, 0.01], color=YELLOW)
        t = ValueTracker(0.01)
        curve = always_redraw(lambda: ax.plot(step_rate, x_range=[0, t.get_value(), 0.01], color=YELLOW))
        area = always_redraw(lambda: ax.get_area(full, x_range=[0, t.get_value()], color=YELLOW, opacity=0.35))
        counter = num(0, num_decimal_places=1, font_size=56, color=WHITE)
        counter.add_updater(lambda m: m.set_value(steps_so_far(t.get_value())))
        counter.move_to(RIGHT * 5.4 + UP * 0.7)
        clbl = Text("steps", font_size=24, color=GREY_B).next_to(counter, DOWN)
        cap = caption("count = sum of rate x 0.04 s (one sample at 25 Hz). Each step is one bump of area 1, "
                      "so the running area ticks up by one per step.")
        self.play(FadeIn(cap), FadeIn(counter), FadeIn(clbl))
        self.add(area, curve)
        self.play(t.animate.set_value(6), run_time=6, rate_func=rate_functions.linear)
        counter.clear_updaters()
        self.wait(1.5)
        self.clear_all()

    # 5. loss: predicted count vs the typed count
    def part_loss(self):
        ttl = title("Training signal: the number you counted")
        self.play(Write(ttl))
        pred = VGroup(Text("StepNet", font_size=28, color=GREY_B),
                      num(31.6, num_decimal_places=1, font_size=64, color=YELLOW)).arrange(DOWN)
        true = VGroup(Text("you counted", font_size=28, color=GREY_B),
                      num(40, num_decimal_places=0, font_size=64, color=GREEN)).arrange(DOWN)
        pred.move_to(LEFT * 3 + UP * 1.0)
        true.move_to(RIGHT * 3 + UP * 1.0)
        self.play(FadeIn(pred), FadeIn(true))
        cap = caption("No per-step labels needed. Each take has one number: the steps you counted out loud. "
                      "The loss is the gap.")
        self.play(FadeIn(cap))
        gap = Line(pred.get_right() + RIGHT * 0.3, true.get_left() + LEFT * 0.3, color=RED, stroke_width=4)
        loss = Text("loss = smooth-L1(31.6, 40) = 7.9", font_size=34, color=RED).move_to(DOWN * 0.9)
        self.play(Create(gap), Write(loss))
        self.wait(2)
        cap = self.swap_caption(cap, "Zero-step takes (standing, fidgeting, air-biking) teach it to hold a flat zero "
                                     "when you are not walking.")
        self.wait(2)
        self.clear_all()

    # 6. back-propagation: the error flows backwards, the weights nudge
    def part_backprop(self):
        ttl = title("Back-propagation: the error flows backwards")
        self.play(Write(ttl))
        cols, links = self.layers_drawing()
        self.play(FadeIn(links), FadeIn(cols))
        loss = num(7.9, num_decimal_places=2, font_size=40, color=RED)
        lgrp = VGroup(Text("loss", font_size=22, color=GREY_B), loss).arrange(DOWN, buff=0.15)
        lgrp.next_to(cols[-1][0], RIGHT, buff=0.35)
        loss.add_updater(lambda m: m.next_to(lgrp[0], DOWN, buff=0.15))  # stay centred as digits change
        self.play(FadeIn(lgrp))
        cap = caption("PyTorch's autograd works out how the loss changes with every weight. "
                      "Colour = gradient sign: red pushes a weight down, blue pushes it up.")
        self.play(FadeIn(cap))
        rng = np.random.default_rng(0)
        y_arrow = cols[1][0].get_top()[1] + 0.35
        xs = [c[0].get_center()[0] for c in cols]
        for val in [4.2, 2.1, 1.0, 0.6]:
            arrows = VGroup(*[Arrow(np.array([xs[i] - 0.15, y_arrow, 0]), np.array([xs[i - 1] + 0.15, y_arrow, 0]),
                                    color=RED, buff=0, stroke_width=4) for i in range(4, 0, -1)])
            self.play(Create(arrows, lag_ratio=0.3), run_time=0.8)
            picked = [links[int(k)] for k in rng.choice(len(links), 60, replace=False)]
            recol = [l.animate.set_stroke(color=RED if rng.random() < 0.5 else BLUE, opacity=0.9, width=2) for l in picked]
            self.play(*recol, ChangeDecimalToValue(loss, val), run_time=0.6)
            self.play(FadeOut(arrows), *[l.animate.set_stroke(color=GREY_B, opacity=0.35, width=0.6) for l in picked], run_time=0.4)
        cap = self.swap_caption(cap, "Adam nudges each weight against its gradient. 300 passes over ~25 takes, with "
                                     "rotation, gain, time-stretch and paired takes so it generalises.")
        self.wait(3)
        self.clear_all()

    # 7. ship the weights to the board, the phone shows the count
    def part_ship(self):
        ttl = title("Ship it: the weights become a JSON file on the board", 38)
        self.play(Write(ttl))
        js = RoundedRectangle(corner_radius=0.2, width=3.4, height=2.0, color=YELLOW).move_to(LEFT * 4.6 + UP * 0.4)
        jtxt = Text('stepnet.json\n{"layers": [...],\n "mean": [...],\n "std": [...]}', font_size=20,
                    line_spacing=0.8).move_to(js)
        board = Rectangle(width=1.6, height=1.2, color=BLUE, fill_opacity=0.3).move_to(UP * 0.4)
        blbl = Text("UNO Q", font_size=22).next_to(board, DOWN, buff=0.15)
        phone = RoundedRectangle(corner_radius=0.3, width=2.2, height=3.6, color=WHITE).move_to(RIGHT * 4.6 + UP * 0.4)
        count = num(0, num_decimal_places=0, font_size=64, color=WHITE).move_to(phone.get_center() + UP * 0.5)
        count.add_updater(lambda m: m.move_to(phone.get_center() + UP * 0.5))
        pm = Text("± 2 steps", font_size=26, color=GREY_B).next_to(count, DOWN)
        a1 = Arrow(js.get_right(), board.get_left(), color=GREY_B, buff=0.15)
        a2 = Arrow(board.get_right(), phone.get_left(), color=GREY_B, buff=0.15)
        self.play(Create(js), Write(jtxt))
        self.play(Create(a1), FadeIn(board), FadeIn(blbl))
        self.play(Create(a2), Create(phone), FadeIn(count), FadeIn(pm))
        cap = caption("The UNO Q runs the same convolutions in plain Python, 25 times a second, and the phone shows "
                      "the count with its measured uncertainty.")
        self.play(FadeIn(cap))
        self.play(ChangeDecimalToValue(count, 38), run_time=4, rate_func=rate_functions.linear)
        self.wait(2)
        end = caption("ME 193 · P6 step counter · trained in PyTorch, counted on a thigh", size=26)
        self.play(FadeOut(cap), run_time=0.5)
        self.play(FadeIn(end))
        self.wait(2)
