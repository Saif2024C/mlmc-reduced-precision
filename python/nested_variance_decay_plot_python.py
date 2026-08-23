"""
nested_variance_decay_plot_python.py -- standalone variance-decay panel.

Reproduces the top-left panel of the six-panel convergence figure as a figure
in its own right: Var(P_k) and Var(P_k - P_{k-1}) on the even super-levels,
log2 vertical axis, with the fitted slope -beta drawn over the levels the
regression uses.

The even super-levels carry the Milstein correction, so this is the decay the
MLMC cost model depends on; the flat upper series is the payoff variance,
which does not decay and is plotted to give the scale.

Usage:
    MPLBACKEND=Agg python3 python/nested_variance_decay_plot_python.py \
        <file.txt> <out.pdf> [title]
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def read_table(path):
    """Parse the mlmc_test convergence table -> (level, var(Pf-Pc), var(Pf))."""
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    k = next(i for i, l in enumerate(lines) if l and l[0] == "-")
    rows = []
    for l in lines[k + 1:]:
        if len(l) <= 10:
            break
        d = np.fromstring(l, sep=" ")
        if d.size < 8:
            break
        rows.append(d)
    r = np.array(rows)
    # cols: l, ave(Pf-Pc), ave(Pf), var(Pf-Pc), var(Pf), kurtosis, check, cost
    return r[:, 0].astype(int), r[:, 3], r[:, 4]


def beta_from_file(path):
    """The beta the harness itself fitted, so the plot and text agree."""
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip().startswith("beta"):
                return float(l.split("=")[1].split()[0])
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else ""

    lev, vdiff, vf = read_table(src)
    beta = beta_from_file(src)

    even = (lev % 2 == 0)
    k = lev[even] // 2
    vd = vdiff[even]
    vp = vf[even]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.plot(k, np.log2(vp), "o-", color="#1f4fd8", ms=4.5, lw=1.2,
            label=r"$\mathbb{V}[P_k]$")
    # k=0 is the level-0 estimate, not a difference, so it is not part of the
    # decay and is dropped from both the curve and the fit.
    ax.plot(k[1:], np.log2(vd[1:]), "s-", color="#d62728", ms=4.5, lw=1.2,
            label=r"$\mathbb{V}[P_k - P_{k-1}]$")

    if beta is not None:
        kk = k[1:]
        c = np.mean(np.log2(vd[1:]) + beta * kk)
        ax.plot(kk, c - beta * kk, "k--", lw=1.1,
                label=rf"slope $-\beta = {-beta:.2f}$")

    ax.set_xlabel(r"grid level $k$")
    ax.set_ylabel(r"$\log_2$ variance")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
