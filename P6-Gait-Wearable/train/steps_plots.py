"""Figures for the step-counter write-up: LORO scatter, error histogram, one take's traces + rate + count, training loss."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stepnet import DT  # noqa: E402
from steps import rates  # noqa: E402

COLORS = {"WALK": "#16a034", "RUN": "#ff7800", "STAND": "#555555", "BIKE": "#0093e7", "MIXED": "#b03ac0"}   # the phone page's
INK, MUTED, MAIN = "#222222", "#888888", "#0093e7"
CHANNELS = ("#4c78a8", "#e0872b", "#6b6b6b")   # x, y, z: three distinct, non-activity hues


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(path, results, m):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    hi = max([r["true"] for r in results] + [r["pred"] for r in results] + [10]) * 1.1
    ax.plot([0, hi], [0, hi], color="#999", lw=1, label="perfect")
    xs = np.linspace(0, hi, 50)
    if m["pm_pct95"] is not None:   # None when no take had steps
        ax.fill_between(xs, xs * (1 - m["pm_pct95"] / 100), xs * (1 + m["pm_pct95"] / 100), color="#ddd", alpha=0.6, label="±%.0f %% (95th pct)" % m["pm_pct95"])
    for act in sorted({r["activity"] for r in results}):
        pts = [r for r in results if r["activity"] == act]
        ax.scatter([r["true"] for r in pts], [r["pred"] for r in pts], color=COLORS.get(act, "k"), label=act, zorder=3, s=36, edgecolor="white", linewidth=0.8, clip_on=False)
    ax.set_xlabel("counted steps (true)")
    ax.set_ylabel("StepNet steps (leave-one-recording-out)")
    ax.set_title("MAE %.1f steps, MAPE %.1f %%" % (m["mae_steps"], m["mape_pct"]))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    style(ax)
    save(fig, path)


def errors(path, results):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    e = [r["pred"] - r["true"] for r in results]
    ax.hist(e, bins=max(5, len(e) // 2), color=MAIN, edgecolor="white")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("predicted − true (steps)")
    ax.set_ylabel("takes")
    ax.set_title("Leave-one-recording-out errors")
    style(ax)
    save(fig, path)


def rate_curve(path, rec, model, mean, std):
    r = rates(model, mean, std, rec.x)
    t = np.arange(rec.x.shape[1]) * DT
    fig, axs = plt.subplots(4, 1, figsize=(7, 7.5), sharex=True)
    for ax, (chs, unit, off) in zip(axs[:2], ((("ax", "ay", "az"), "accel (g)", 0), (("gx", "gy", "gz"), "gyro (deg/s)", 3))):
        for i, (name, c) in enumerate(zip(chs, CHANNELS)):
            ax.plot(t, rec.x[off + i], color=c, lw=0.8, label=name)
        ax.set_ylabel(unit)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))   # headroom so the legend never sits on the trace
        ax.legend(fontsize=7, ncol=3, loc="upper right", frameon=False)
    axs[0].set_title("%s: %d counted steps" % (rec.name, rec.steps))
    axs[2].fill_between(t, 0, r, color="#ffde00", alpha=0.85)
    axs[2].plot(t, r, color="#c9a800", lw=0.8)
    axs[2].set_ylabel("step rate (steps/s)")
    axs[3].plot(t, np.cumsum(r) * DT, color=MAIN, lw=2, label="count = area under the rate")
    axs[3].axhline(rec.steps, color=MUTED, ls="--", lw=1, label="counted: %d" % rec.steps)
    axs[3].set_ylabel("steps")
    axs[3].set_xlabel("time (s)")
    axs[3].legend(fontsize=7, loc="upper left", frameon=False)
    for ax in axs:
        style(ax)
    save(fig, path)


def training(path, losses):
    fig, ax = plt.subplots(figsize=(5.2, 3))
    ax.plot(losses, color=MAIN)
    ax.set_xlabel("epoch")
    ax.set_ylabel("smooth-L1 loss (steps)")
    ax.set_yscale("log")
    ax.set_title("Training loss")
    style(ax)
    save(fig, path)


def figures(out_dir, results, m, recs, model, mean, std, losses):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scatter(out / "steps_scatter.png", results, m)
    errors(out / "steps_errors.png", results)
    walks = [r for r in recs if r.activity == "WALK" and r.steps > 0] or recs
    rate_curve(out / "steps_rate.png", max(walks, key=lambda r: r.steps), model, mean, std)
    training(out / "steps_training.png", losses)
