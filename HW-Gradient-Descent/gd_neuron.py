"""Gradient descent by hand (numpy only) on the V data from Anna's code.

Three models, all trained with the same loss L = 1/(2N) * sum (y_hat - y)^2 and plain batch gradient descent:
  1. one linear neuron            y_hat = m*x + b                 (a straight line cannot fit a V)
  2. one ReLU neuron              y_hat = ReLU(m*x + b)           (fits one arm of the V, flat on the other)
  3. a ReLU layer of two neurons  y_hat = v1*ReLU(w1*x + b1) + v2*ReLU(w2*x + b2) + c   (fits the whole V)
x and y are standardised first (subtract the mean, divide by the spread), as in the class code; plots are in the
original units. Also prints the first 5 iterations of both Excel sheets (the class line and the V data) as a cross-check.

Run:  .venv/bin/python gd_neuron.py      -> figures/v_fit.png, figures/v_loss.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = HERE / "data" / "annacode-v.tsv"
LR, EPOCHS = 0.1, 2000


def load():
    rows = [line.split("\t") for line in DATA.read_text().splitlines()[1:] if line.strip()]
    return np.array([float(r[1]) for r in rows]), np.array([float(r[2]) for r in rows])


def relu(z):
    return np.maximum(0.0, z)


# ---- one neuron: y_hat = phi(m*x + b) ------------------------------------------------------------------
def neuron_step(x, y, m, b, lr, use_relu):
    z = m * x + b
    y_hat = relu(z) if use_relu else z
    e = y_hat - y                                  # error
    g = (z > 0).astype(float) if use_relu else 1.0  # dReLU/dz: 1 where the neuron is on, 0 where it is off
    loss = float(np.mean(e ** 2) / 2)
    dm = float(np.mean(e * g * x))                 # dL/dm
    db = float(np.mean(e * g))                     # dL/db
    return m - lr * dm, b - lr * db, loss, dm, db


def train_neuron(x, y, m, b, lr, epochs, use_relu):
    hist = []
    for k in range(epochs):
        m_new, b_new, loss, dm, db = neuron_step(x, y, m, b, lr, use_relu)
        hist.append((k, m, b, loss, dm, db))
        m, b = m_new, b_new
    return m, b, hist


# ---- ReLU layer: two hidden neurons, one linear output -------------------------------------------------
def train_layer(x, y, lr, epochs):
    w = np.array([1.0, -1.0])   # one hinge facing right, one facing left
    b = np.array([0.0, 0.0])
    v = np.array([0.5, 0.5])
    c = 0.0
    losses = []
    for _ in range(epochs):
        z = np.outer(x, w) + b          # (N, 2)
        a = relu(z)
        y_hat = a @ v + c
        e = y_hat - y
        losses.append(float(np.mean(e ** 2) / 2))
        d = e / len(x)                  # dL/dy_hat
        gate = (z > 0).astype(float)
        dv = a.T @ d                    # dL/dv_j = sum d_i a_ij
        dc = d.sum()
        dz = np.outer(d, v) * gate      # back through the output weights and the ReLU
        dw = dz.T @ x
        db_ = dz.sum(axis=0)
        w, b, v, c = w - lr * dw, b - lr * db_, v - lr * dv, c - lr * dc
    return (w, b, v, c), losses


def layer_predict(p, x):
    w, b, v, c = p
    return relu(np.outer(x, w) + b) @ v + c


def main():
    FIG.mkdir(exist_ok=True)

    # cross-check for the spreadsheet: x = 1..5, y = 2x, start m = b = 0, lr = 0.1
    xs, ys = np.arange(1.0, 6.0), 2 * np.arange(1.0, 6.0)
    _, _, hist = train_neuron(xs, ys, 0.0, 0.0, 0.1, 6, use_relu=False)
    print("Spreadsheet check (x = 1..5, y = 2x, lr = 0.1)")
    print(f"{'iter':>4} {'m':>8} {'b':>8} {'loss':>9} {'dL/dm':>9} {'dL/db':>9}")
    for k, m, b, loss, dm, db in hist:
        print(f"{k:>4} {m:>8.4f} {b:>8.4f} {loss:>9.4f} {dm:>9.4f} {db:>9.4f}")

    # the V data
    x_raw, y_raw = load()
    xm, xs_, ym, ys_ = x_raw.mean(), x_raw.std(), y_raw.mean(), y_raw.std()
    x, y = (x_raw - xm) / xs_, (y_raw - ym) / ys_
    rmse = lambda pred: float(np.sqrt(np.mean((pred * ys_ + ym - y_raw) ** 2)))

    m1, b1, h1 = train_neuron(x, y, 0.0, 0.0, LR, EPOCHS, use_relu=False)
    m2, b2, h2 = train_neuron(x, y, 1.0, 0.0, LR, EPOCHS, use_relu=True)
    p3, l3 = train_layer(x, y, LR, EPOCHS)
    print("\nSpreadsheet check, sheet 'V data' (ReLU neuron, start m = 1, b = 0, lr = 0.1)")
    for k, m, b, loss, dm, db in h2[:6]:
        print(f"{k:>4} {m:>9.6f} {b:>9.6f} {loss:>9.6f} {dm:>9.6f} {db:>9.6f}")
    print(f"\nV data: {len(x)} points, x {x_raw.min():.1f}..{x_raw.max():.1f}, y {y_raw.min():.1f}..{y_raw.max():.1f}")
    print(f"1 linear neuron : m={m1:.3f} b={b1:.3f}  loss={h1[-1][3]:.4f}  RMSE={rmse(m1 * x + b1):.2f}")
    print(f"1 ReLU neuron   : m={m2:.3f} b={b2:.3f}  loss={h2[-1][3]:.4f}  RMSE={rmse(relu(m2 * x + b2)):.2f}")
    w, b, v, c = p3
    print(f"ReLU layer (2)  : w={np.round(w, 3)} b={np.round(b, 3)} v={np.round(v, 3)} c={c:.3f}  "
          f"loss={l3[-1]:.4f}  RMSE={rmse(layer_predict(p3, x)):.2f}")
    kinks = -b / w * xs_ + xm
    print(f"kinks of the two hinges, in original x: {np.round(kinks, 1)}")

    DATA_C, C1, C2, C3 = "#8A94A0", "#7B5EA7", "#E07B00", "#1F4E79"
    grid_raw = np.linspace(x_raw.min(), x_raw.max(), 300)
    grid = (grid_raw - xm) / xs_
    back = lambda pred: pred * ys_ + ym

    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=150)
    ax.scatter(x_raw, y_raw, s=16, color=DATA_C, label="V data", zorder=2)
    ax.plot(grid_raw, back(m1 * grid + b1), color=C1, lw=1.8, label="1 linear neuron")
    ax.plot(grid_raw, back(relu(m2 * grid + b2)), color=C2, lw=1.8, label="1 ReLU neuron")
    ax.plot(grid_raw, back(layer_predict(p3, grid)), color=C3, lw=2.4, label="ReLU layer, 2 neurons")
    ax.set_xlabel("x"); ax.set_ylabel("output 1")
    ax.set_title(f"Gradient descent fits to the V data ({EPOCHS} epochs, lr = {LR})", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "v_fit.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=150)
    ep = np.arange(EPOCHS)
    ax.plot(ep, [h[3] for h in h1], color=C1, lw=1.8, label="1 linear neuron")
    ax.plot(ep, [h[3] for h in h2], color=C2, lw=1.8, label="1 ReLU neuron")
    ax.plot(ep, l3, color=C3, lw=2.4, label="ReLU layer, 2 neurons")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, EPOCHS)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss  L = mean (y_hat - y)^2 / 2  (standardised units)")
    ax.set_title("Loss as a function of epoch", fontsize=11)
    ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "v_loss.png"); plt.close(fig)
    print(f"\nfigures: {FIG / 'v_fit.png'}, {FIG / 'v_loss.png'}")


if __name__ == "__main__":
    main()
