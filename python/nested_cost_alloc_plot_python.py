"""
nested_cost_alloc_plot_python.py -- cost-complexity and sample-allocation
overlays across precision schemes.

Produces two figures from the mlmc_test complexity tables:

  (1) cost complexity:  eps^2 * C against eps, one line per scheme, plus the
      standard Monte Carlo reference.  Multiplying by eps^2 flattens the
      O(eps^-2) ideal, so a horizontal line is the MLMC target rate and any
      upward slope to the left is the cost growing faster than that.

  (2) sample allocation: N_l against super-level l at ONE fixed eps, one line
      per scheme.  Fixing eps is what makes the overlay legible: the usual
      convention plots one line per eps for a single scheme, and this is the
      transpose of that.

Schemes with a non-positive beta never populate a complexity table (the
adaptive sampler cannot hit its target), so they are skipped with a warning
rather than silently dropped.

Usage:
    MPLBACKEND=Agg python3 python/nested_cost_alloc_plot_python.py \
        <dir> <domain> <opt> [eps_for_allocation]

    <dir>     directory holding nested_<domain>_fp16_avx512_<mode>_<opt>.txt
    <domain>  scalar | basket
    <opt>     1 | 2
"""

import os
import sys

import matplotlib.pyplot as plt

# scheme key -> (filename mode, legend label, colour, marker, linestyle)
SCHEMES = [
    ("nokahan",  "fp16, no Kahan",      "#c0392b", "o", "-"),
    ("kahan",    "fp16, Kahan",         "#e8a33d", "s", "-"),
    ("adaptive", "adaptive",            "#7d3c98", "^", "-"),
    ("puref32",  "pure fp32",           "#1e8449", "D", "-"),
]


def parse_complexity(path):
    """Return list of dicts, one per eps row of the complexity table."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as fh:
        lines = fh.readlines()

    start = None
    for i, line in enumerate(lines):
        if "eps" in line and "mlmc_cost" in line:
            start = i + 2          # skip header and the dashed rule
            break
    if start is None:
        return rows

    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 6:
            break
        try:
            rows.append({
                "eps":       float(parts[0]),
                "value":     float(parts[1]),
                "mlmc_cost": float(parts[2]),
                "std_cost":  float(parts[3]),
                "savings":   float(parts[4]),
                "Nl":        [float(x) for x in parts[5:]],
            })
        except ValueError:
            break
    return rows


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    domain    = sys.argv[2]
    opt       = sys.argv[3]
    eps_alloc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    data = {}
    for mode, label, colour, marker, ls in SCHEMES:
        path = os.path.join(
            directory, f"nested_{domain}_fp16_avx512_{mode}_{opt}.txt")
        rows = parse_complexity(path)
        if rows is None:
            print(f"  missing: {path}")
        elif not rows:
            print(f"  no complexity table (non-positive beta?): {mode}")
        else:
            data[mode] = rows

    if not data:
        print("nothing to plot")
        sys.exit(1)

    payoff = {"basket": {"1": "European", "2": "Asian"},
              "scalar": {"1": "Asian",    "2": "Lookback"}}[domain][opt]

    # ---------------------------------------------------------------- (1)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for mode, label, colour, marker, ls in SCHEMES:
        if mode not in data:
            continue
        rows = data[mode]
        eps = [r["eps"] for r in rows]
        y   = [r["eps"] ** 2 * r["mlmc_cost"] for r in rows]
        ax.loglog(eps, y, ls, color=colour, marker=marker,
                  markersize=6, label=label)

    ref = data[next(iter(data))]
    ax.loglog([r["eps"] for r in ref],
              [r["eps"] ** 2 * r["std_cost"] for r in ref],
              "--", color="0.45", marker="x", markersize=6,
              label="standard MC")

    ax.set_xlabel(r"accuracy  $\varepsilon$")
    ax.set_ylabel(r"$\varepsilon^{2}\,C$")
    ax.set_title(f"{domain.capitalize()} {payoff}: cost complexity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out1 = os.path.join(directory, f"nested_{domain}_cost_{opt}.pdf")
    fig.savefig(out1)
    print("wrote", out1)

    # ---------------------------------------------------------------- (2)
    # Even and odd super-levels are separate estimators (Milstein correction
    # and precision correction), so plotting them on one axis produces a
    # zigzag that hides the decay of each.  One panel per family, indexed by
    # grid level k, keeps both monotone.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    plotted = False
    for mode, label, colour, marker, ls in SCHEMES:
        if mode not in data:
            continue
        row = min(data[mode], key=lambda r: abs(r["eps"] - eps_alloc))
        if abs(row["eps"] - eps_alloc) > 1e-9:
            print(f"  {mode}: nearest eps is {row['eps']}, not {eps_alloc}")
        Nl = row["Nl"]
        even = [(l // 2, n) for l, n in enumerate(Nl) if l % 2 == 0]
        odd  = [(l // 2, n) for l, n in enumerate(Nl) if l % 2 == 1]
        for ax, series in ((axes[0], even), (axes[1], odd)):
            ax.semilogy([k for k, _ in series], [n for _, n in series], ls,
                        color=colour, marker=marker, markersize=6, label=label)
        plotted = True

    if plotted:
        axes[0].set_title("Milstein correction (even $l=2k$)")
        axes[1].set_title("precision correction (odd $l=2k+1$)")
        axes[0].set_ylabel(r"samples  $N_l$")
        for ax in axes:
            ax.set_xlabel(r"grid level  $k$")
            ax.grid(True, which="both", alpha=0.3)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{domain.capitalize()} {payoff}: "
                     rf"sample allocation at $\varepsilon={eps_alloc}$")
        fig.tight_layout()
        out2 = os.path.join(directory, f"nested_{domain}_alloc_{opt}.pdf")
        fig.savefig(out2)
        print("wrote", out2)


if __name__ == "__main__":
    main()
