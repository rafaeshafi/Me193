"""Build gradient_descent_5_iterations.xlsx: five hand-traceable gradient-descent iterations on m and b.

Every number except the data, the learning rate and the starting m, b is an Excel formula, so changing an
input re-runs all five iterations. Sheet 1 is the linear neuron on the spring data, sheet 2 the ReLU
neuron on the motor data (same data and starting points as gd_neuron.py, so the numbers match).

Run:  .venv/bin/python make_spreadsheet.py
"""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as col

import gd_neuron as g

OUT = Path(__file__).resolve().parent / "gradient_descent_5_iterations.xlsx"
FONT = "Arial"
BOLD = Font(name=FONT, bold=True)
PLAIN = Font(name=FONT)
INPUT = Font(name=FONT, color="0000FF")
YELLOW = PatternFill("solid", fgColor="FFFF00")
GREY = PatternFill("solid", fgColor="EDEDED")
NUM = "0.0000"
ITERS = 5
DATA_ROW = 10  # first data row; inputs and headers sit above


def put(ws, ref, value, font=PLAIN, fmt=None, fill=None, align=None):
    c = ws[ref]
    c.value = value
    c.font = font
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    if align:
        c.alignment = Alignment(horizontal=align)
    return c


def header(ws, title, model, x, y, xlab, ylab, m0, b0, lr):
    put(ws, "A1", title, Font(name=FONT, bold=True, size=13))
    put(ws, "A2", model)
    put(ws, "A3", "Blue cells on yellow are the inputs; everything else is a formula. "
                  "Each block is one gradient-descent iteration; its m and b come from the block before it.")
    put(ws, "A5", "learning rate lr", BOLD); put(ws, "B5", lr, INPUT, NUM, YELLOW)
    put(ws, "A6", "initial m", BOLD);        put(ws, "B6", m0, INPUT, NUM, YELLOW)
    put(ws, "A7", "initial b", BOLD);        put(ws, "B7", b0, INPUT, NUM, YELLOW)
    n = len(x)
    last = DATA_ROW + n - 1
    put(ws, "A8", "N (points)", BOLD);       put(ws, "B8", f"=COUNT(A{DATA_ROW}:A{last})", PLAIN, "0")
    put(ws, f"A{DATA_ROW - 1}", xlab, BOLD, fill=GREY)
    put(ws, f"B{DATA_ROW - 1}", ylab, BOLD, fill=GREY)
    for i, (xi, yi) in enumerate(zip(x, y)):
        put(ws, f"A{DATA_ROW + i}", float(xi), INPUT, "0.0", YELLOW)
        put(ws, f"B{DATA_ROW + i}", float(yi), INPUT, "0.0", YELLOW)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 11
    return n, last


def block(ws, k, c0, n, last, columns, grads, activation):
    """One iteration in columns c0.. ; `columns` = (header, formula_template) per column; `grads` = (dm, db)."""
    L = [col(c0 + j) for j in range(len(columns))]
    width = len(columns)
    ws.merge_cells(start_row=5, start_column=c0, end_row=5, end_column=c0 + width - 1)
    put(ws, f"{L[0]}5", f"Iteration {k}", BOLD, align="center", fill=GREY)
    put(ws, f"{L[0]}6", "m", BOLD)
    put(ws, f"{L[0]}7", "b", BOLD)
    if k == 1:
        put(ws, f"{L[1]}6", "=$B$6", PLAIN, NUM)
        put(ws, f"{L[1]}7", "=$B$7", PLAIN, NUM)
    else:
        prev = col(c0 - width)  # value column of the previous block (blocks are width+1 apart)
        put(ws, f"{L[1]}6", f"={prev}{last + 4}", PLAIN, NUM)
        put(ws, f"{L[1]}7", f"={prev}{last + 5}", PLAIN, NUM)
    m, b = f"${L[1]}$6", f"${L[1]}$7"
    for j, (head, _) in enumerate(columns):
        put(ws, f"{L[j]}{DATA_ROW - 1}", head, BOLD, fill=GREY)
        ws.column_dimensions[L[j]].width = 12
    for r in range(DATA_ROW, last + 1):
        for j, (_, tmpl) in enumerate(columns):
            f = tmpl.format(r=r, m=m, b=b, **{f"c{i}": L[i] for i in range(width)})
            put(ws, f"{L[j]}{r}", f, PLAIN, NUM)
    rng = lambda j: f"{L[j]}{DATA_ROW}:{L[j]}{last}"
    e_col = next(j for j, (h, _) in enumerate(columns) if h.startswith("e ="))
    put(ws, f"{L[0]}{last + 1}", "L = Σe²/N", BOLD)
    put(ws, f"{L[1]}{last + 1}", f"=SUMPRODUCT({rng(e_col)},{rng(e_col)})/$B$8", PLAIN, NUM)
    put(ws, f"{L[0]}{last + 2}", "∂L/∂m", BOLD)
    put(ws, f"{L[1]}{last + 2}", f"=2*SUM({rng(grads[0])})/$B$8", PLAIN, NUM)
    put(ws, f"{L[0]}{last + 3}", "∂L/∂b", BOLD)
    put(ws, f"{L[1]}{last + 3}", f"=2*SUM({rng(grads[1])})/$B$8", PLAIN, NUM)
    put(ws, f"{L[0]}{last + 4}", "new m = m − lr·∂L/∂m", BOLD)
    put(ws, f"{L[1]}{last + 4}", f"={L[1]}6-$B$5*{L[1]}{last + 2}", PLAIN, NUM)
    put(ws, f"{L[0]}{last + 5}", "new b = b − lr·∂L/∂b", BOLD)
    put(ws, f"{L[1]}{last + 5}", f"={L[1]}7-$B$5*{L[1]}{last + 3}", PLAIN, NUM)
    ws.column_dimensions[L[0]].width = 20
    return L[1]


def summary(ws, row, value_cols, last, final_loss_formula):
    put(ws, f"A{row}", "Summary", BOLD)
    for j, h in enumerate(("iteration", "m", "b", "loss L")):
        put(ws, f"{col(1 + j)}{row + 1}", h, BOLD, fill=GREY)
    for k, vc in enumerate(value_cols):
        r = row + 2 + k
        put(ws, f"A{r}", k, PLAIN, "0")
        put(ws, f"B{r}", f"={vc}6", PLAIN, NUM)
        put(ws, f"C{r}", f"={vc}7", PLAIN, NUM)
        put(ws, f"D{r}", f"={vc}{last + 1}", PLAIN, NUM)
    r = row + 2 + len(value_cols)
    vc = value_cols[-1]
    put(ws, f"A{r}", len(value_cols), PLAIN, "0")
    put(ws, f"B{r}", f"={vc}{last + 4}", PLAIN, NUM)
    put(ws, f"C{r}", f"={vc}{last + 5}", PLAIN, NUM)
    put(ws, f"D{r}", final_loss_formula.format(m=f"B{r}", b=f"C{r}", last=last), PLAIN, NUM)
    put(ws, f"E{r}", "← after 5 updates", PLAIN)


def linear_sheet(ws):
    n, last = header(ws, "Linear neuron trained by gradient descent (spring: force vs extension)",
                     "Model ŷ = m·x + b.   Loss L = (1/N)·Σ(ŷ − y)².   ∂L/∂m = (2/N)·Σ(ŷ − y)·x,   ∂L/∂b = (2/N)·Σ(ŷ − y).   "
                     "Update m ← m − lr·∂L/∂m,  b ← b − lr·∂L/∂b.",
                     g.SPRING_X, g.SPRING_Y, "x extension (cm)", "y force (N)", m0=0.0, b0=0.0, lr=0.02)
    columns = [("ŷ = m·x + b", "={m}*$A{r}+{b}"),
               ("e = ŷ − y", "={c0}{r}-$B{r}"),
               ("e·x", "={c1}{r}*$A{r}")]
    vcs = []
    c0 = 4
    for k in range(1, ITERS + 1):
        vcs.append(block(ws, k, c0, n, last, columns, grads=(2, 1), activation="linear"))
        c0 += len(columns) + 1
    summary(ws, last + 8, vcs, last,
            "=SUMPRODUCT(({m}*$A${d}:$A${last}+{b}-$B${d}:$B${last})^2)/$B$8".replace("{d}", str(DATA_ROW)))


def relu_sheet(ws):
    n, last = header(ws, "ReLU neuron trained by gradient descent (DC motor: speed vs voltage, dead band)",
                     "Model z = m·x + b,  ŷ = ReLU(z) = max(0, z).   Loss L = (1/N)·Σ(ŷ − y)².   "
                     "dReLU/dz = 1 if z > 0 else 0 (the gate g).   ∂L/∂m = (2/N)·Σ(ŷ − y)·g·x,   ∂L/∂b = (2/N)·Σ(ŷ − y)·g.",
                     g.MOTOR_X, g.MOTOR_Y, "x voltage (V)", "y speed (rad/s)", m0=1.0, b0=0.0, lr=0.02)
    columns = [("z = m·x + b", "={m}*$A{r}+{b}"),
               ("ŷ = max(0, z)", "=MAX(0,{c0}{r})"),
               ("g = 1 if z>0", "=IF({c0}{r}>0,1,0)"),
               ("e = ŷ − y", "={c1}{r}-$B{r}"),
               ("e·g", "={c3}{r}*{c2}{r}"),
               ("e·g·x", "={c4}{r}*$A{r}")]
    vcs = []
    c0 = 4
    for k in range(1, ITERS + 1):
        vcs.append(block(ws, k, c0, n, last, columns, grads=(5, 4), activation="relu"))
        c0 += len(columns) + 1
    summary(ws, last + 8, vcs, last,
            "=SUMPRODUCT((({m}*$A${d}:$A${last}+{b}>0)*({m}*$A${d}:$A${last}+{b})-$B${d}:$B${last})^2)/$B$8"
            .replace("{d}", str(DATA_ROW)))


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Linear neuron"
    linear_sheet(ws)
    relu_sheet(wb.create_sheet("ReLU neuron"))
    for ws in wb.worksheets:
        ws.freeze_panes = f"C{DATA_ROW}"
    wb.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
