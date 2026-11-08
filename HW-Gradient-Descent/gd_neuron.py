"""Gradient descent for a single neuron, by hand (numpy only).

Part 1: a linear neuron  y_hat = m*x + b          fitted to spring force-vs-extension data.
Part 2: a ReLU neuron    y_hat = ReLU(m*x + b)    fitted to motor speed-vs-voltage data with a dead band.

Both use mean-squared-error loss L = (1/N) * sum (y_hat - y)^2 and plain (batch) gradient descent:
    m <- m - lr * dL/dm,   b <- b - lr * dL/db.
Prints the first 5 iterations (the same numbers as gradient_descent_5_iterations.xlsx) and saves
figures/<name>_fit.png (data + fit, with the first iterations drawn in grey) and figures/<name>_loss.png.

Run:  .venv/bin/python gd_neuron.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

# ---- data -----------------------------------------------------------------------------------------
# Spring: extension x (cm) -> force y (N). Roughly y = 2x + 1 (k = 2 N/cm, 1 N preload) plus noise.
SPRING_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
SPRING_Y = np.array([3.1, 4.9, 7.2, 8.8, 11.1, 13.0, 14.9, 17.2])

# DC motor: voltage x (V) -> shaft speed y (rad/s). Nothing moves below ~3 V (friction dead band),
# then roughly 2 rad/s per volt: y = ReLU(2x - 6) plus noise.
MOTOR_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
MOTOR_Y = np.array([0.0, 0.0, 0.0, 0.1, 2.1, 3.9, 6.2, 7.9, 10.1])

# ---- the neuron -----------------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def forward(x, m, b, activation):
    z = m * x + b
    return relu(z) if activation == "relu" else z


def loss_and_grads(x, y, m, b, activation):
    """MSE loss and its partial derivatives with respect to m and b."""
    n = len(x)
    z = m * x + b
    y_hat = relu(z) if activation == "relu" else z
    err = y_hat - y                                    # dL/dy_hat = 2*err/n
    gate = (z > 0).astype(float) if activation == "relu" else np.ones(n)  # dReLU/dz = 1 if z>0 else 0
    delta = err * gate                                 # dL/dz (times n/2)
    loss = float(np.mean(err ** 2))
    dm = float(2.0 / n * np.sum(delta * x))            # dz/dm = x
    db = float(2.0 / n * np.sum(delta))                # dz/db = 1
    return loss, dm, db


def train(x, y, m, b, lr, epochs, activation="linear"):
    """Batch gradient descent. Returns the loss per epoch and the (m, b) before each update."""
    history = []  # (epoch, m, b, loss, dL/dm, dL/db)
    for k in range(epochs):
        loss, dm, db = loss_and_grads(x, y, m, b, activation)
        history.append((k, m, b, loss, dm, db))
        m -= lr * dm
        b -= lr * db
    final_loss = loss_and_grads(x, y, m, b, activation)[0]
    return m, b, final_loss, history


# ---- plots ------------------------------------------------------------------------------------------
DATA_C, FIT_C, ITER_C = "#1F4E79", "#E07B00", "#9AA3AD"


def plot_fit(name, x, y, m, b, history, activation, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 200)
    for k, mk, bk, *_ in history[:6]:  # iterations 0..5, fading in
        ax.plot(xs, forward(xs, mk, bk, activation), color=ITER_C, lw=1.2, alpha=0.25 + 0.12 * k,
                label="iterations 0-5" if k == 0 else None)
    ax.plot(xs, forward(xs, m, b, activation), color=FIT_C, lw=2,
            label=f"fit after {len(history)} epochs: m={m:.3f}, b={b:.3f}")
    ax.scatter(x, y, s=38, color=DATA_C, zorder=3, label="data")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_fit.png")
    plt.close(fig)


def plot_loss(name, history, final_loss, title):
    losses = np.array([h[3] for h in history] + [final_loss])
    epochs = np.arange(len(losses))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=150, sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    zoom = 20  # left panel: the start, where the spreadsheet's iterations live
    for ax, lo, hi, sub in ((ax1, 0, zoom, f"first {zoom} epochs"), (ax2, 0, len(losses) - 1, "whole run")):
        ax.plot(epochs[lo:hi + 1], losses[lo:hi + 1], color=DATA_C, lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(sub, fontsize=10, color="#555555")
        ax.grid(alpha=0.25, which="both")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.scatter(epochs[:6], losses[:6], s=34, color=FIT_C, zorder=3, label="iterations 0-5 (spreadsheet)")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax1.set_ylabel("loss  L = mean (y_hat - y)^2")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_loss.png")
    plt.close(fig)


def report(title, history, m, b, final_loss):
    print(f"\n{title}")
    print(f"{'iter':>4} {'m':>9} {'b':>9} {'loss':>10} {'dL/dm':>10} {'dL/db':>10}")
    for k, mk, bk, loss, dm, db in history[:5]:
        print(f"{k:>4} {mk:>9.4f} {bk:>9.4f} {loss:>10.4f} {dm:>10.4f} {db:>10.4f}")
    k, mk, bk, loss, *_ = history[5]
    print(f"{k:>4} {mk:>9.4f} {bk:>9.4f} {loss:>10.4f}   <- after 5 updates")
    print(f"final: m={m:.4f} b={b:.4f} loss={final_loss:.4f} after {len(history)} epochs")


def main():
    FIG.mkdir(exist_ok=True)

    # Part 1: linear neuron on the spring data
    m, b, L, hist = train(SPRING_X, SPRING_Y, m=0.0, b=0.0, lr=0.02, epochs=2000)
    report("Part 1: linear neuron  y_hat = m*x + b  (spring)", hist, m, b, L)
    m_ls, b_ls = np.polyfit(SPRING_X, SPRING_Y, 1)
    print(f"check: least-squares closed form m={m_ls:.4f} b={b_ls:.4f}")
    plot_fit("linear", SPRING_X, SPRING_Y, m, b, hist, "linear", "extension x (cm)", "force y (N)",
             "Linear neuron fitted by gradient descent (lr = 0.02)")
    plot_loss("linear", hist, L, "Linear neuron: loss vs epoch")

    # Part 2: ReLU neuron on the motor data
    m, b, L, hist = train(MOTOR_X, MOTOR_Y, m=1.0, b=0.0, lr=0.02, epochs=2000, activation="relu")
    report("Part 2: ReLU neuron  y_hat = max(0, m*x + b)  (motor dead band)", hist, m, b, L)
    print(f"check: dead band ends at x = -b/m = {-b / m:.3f} V (data: between 3 and 4 V)")
    plot_fit("relu", MOTOR_X, MOTOR_Y, m, b, hist, "relu", "voltage x (V)", "speed y (rad/s)",
             "ReLU neuron fitted by gradient descent (lr = 0.02)")
    plot_loss("relu", hist, L, "ReLU neuron: loss vs epoch")
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
